from typing import Any

import numpy as np
import xarray as xr
import yaml
from icecream import ic
from tqdm.auto import tqdm

from unseen_awg.preprocessing.compute_climatology import interpolate_then_rolling_mean
from unseen_awg.snakemake_utils import snakemake_handler


def groups_counts_init_time(
    valid_time: xr.DataArray,
) -> tuple[xr.DataArray, xr.DataArray]:
    """
    Given a 2D xarray DataArray of valid_time (with dimensions init_time and lead_time),
    return unique values of valid_time, their counts, and a lookup table for init_time.

    Parameters
    ----------
    valid_time : xr.DataArray
        Array of valid_times with dimensions (init_time, lead_time).
        Order of dimensions should not matter.

    Returns
    -------
    tuple[xr.DataArray, xr.DataArray]
        Tuple containing:
        - DataArray with counts of each valid_time.
        - DataArray with counts of each valid_time and lookup table for init_time
    """
    # Group by unique values
    gb = valid_time.groupby(valid_time)

    # Get counts
    counts_valid_times = gb.count()

    # Calculate maximum number of samples for efficient memory allocation
    max_n_samples = counts_valid_times.max().data

    # Initialize array to store init_times for each valid_time and sample
    init_times = xr.DataArray(
        np.full(
            (len(counts_valid_times), max_n_samples),
            fill_value=np.datetime64("NaT", "ns"),
        ),
        coords={
            "valid_time": counts_valid_times.valid_time,
            "sample": np.arange(max_n_samples),
        },
    )

    # Populate init_times for each valid_time
    for vt, idcs in gb.groups.items():
        idcs = np.unravel_index(idcs, shape=valid_time.shape)
        i_lead_time = valid_time.dims.index("init_time")
        init_times.loc[
            {"valid_time": vt, "sample": slice(None, len(idcs[i_lead_time]) - 1)}
        ] = valid_time.init_time.isel(init_time=idcs[i_lead_time]).data.astype(
            "datetime64[ns]"
        )

    # sort samples such that for a given valid_time,
    # the init_time increases with increasing sample.
    return counts_valid_times, xr.apply_ufunc(
        np.sort, init_times, input_core_dims=[["sample"]], output_core_dims=[["sample"]]
    )


def get_new_coordinates(
    d: xr.DataArray | xr.Dataset, counts_valid_times: xr.DataArray
) -> dict[str, np.ndarray]:
    """
    Extract new coordinates for restructured data with valid_time and sample dimensions.

    Parameters
    ----------
    d : xr.DataArray | xr.Dataset
        Original dataset or data array including init_time and lead_time dimensions.
    counts_valid_times : xr.DataArray
        DataArray with unique valid_time values (coords) and their counts.

    Returns
    -------
    dict[str, np.ndarray]
        Dictionary of new coordinates for valid_time and sample dimensions.
    """
    new_coords = {}
    new_coords["valid_time"] = counts_valid_times.valid_time.data
    new_coords["sample"] = np.arange(counts_valid_times.max())

    # Include other dimensions except init_time and lead_time
    new_coords = new_coords | {
        dim: d.coords[dim] for dim in d.dims if dim not in ["init_time", "lead_time"]
    }

    return new_coords


def get_indices_of_valid_time(
    valid_time_da: xr.DataArray, valid_time: np.datetime64
) -> dict[str, xr.DataArray]:
    """
    Given an array of valid_time values and a specific valid_time, extract a dict
    of indices (for lead_time, init_time) that can be used to extract all elements
    of valid_time_da with date valid_time.

    Parameters
    ----------
    valid_time_da : xr.DataArray
        Array of valid time steps.
    valid_time : np.datetime64
        Specific valid date to select indices for.

    Returns
    -------
    dict[str, xr.DataArray]
        Dictionary of indices for init_time and lead_time dimensions.

    Raises
    ------
    ValueError
        If valid_time_da contains dimensions not in (init_time, lead_time).
    """
    # Transpose to ensure consistent indexing regardless of dimension order
    valid_time_da = valid_time_da.transpose("init_time", "lead_time")

    # Create boolean mask and sort by init_time
    m = (valid_time_da == valid_time).to_series()
    m = m[m].sort_index()  # Sort in ascending order of init_time

    indices = {}
    indices["init_time"] = xr.DataArray(
        np.empty(len(m), dtype="datetime64[ns]"), coords={"sample": np.arange(len(m))}
    )
    indices["lead_time"] = xr.DataArray(
        np.empty(len(m), dtype="timedelta64[ns]"), coords={"sample": np.arange(len(m))}
    )

    # Extract indices for each dimension
    for i, dim in enumerate(valid_time_da.dims):
        if dim == "init_time":
            indices[dim].data = np.array([idx[i] for idx in m.index]).astype(
                "datetime64[ns]"
            )
        elif dim == "lead_time":
            indices[dim].data = np.array([idx[i] for idx in m.index]).astype(
                "timedelta64[ns]"
            )
        else:
            raise ValueError(
                "Invalid dimension. Dimension name must be in (init_time, lead_time)"
            )
    return indices


def to_valid_time_sample(ds: xr.Dataset) -> xr.Dataset:
    """
    Restructure dataset from (init_time, lead_time) to (valid_time, sample) dimensions.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset with dimensions including init_time and lead_time.

    Returns
    -------
    xr.Dataset
        Restructured dataset with valid_time and sample dimensions.
        Includes init_time variable that allows reconstructing init_time for each sample
    """
    counts_valid_times, init_times = groups_counts_init_time(ds.valid_time)

    coords = get_new_coordinates(ds, counts_valid_times)
    res = xr.Dataset(coords=coords)

    for var in ds.data_vars:
        # Initialize empty data array
        coords_var = get_new_coordinates(ds[var], counts_valid_times)
        res[var] = xr.DataArray(
            np.full([len(c) for c in coords_var.values()], fill_value=np.nan),
            coords=coords_var,
        )

        # Process each valid_time
        for valid_time in res.valid_time:
            indices = get_indices_of_valid_time(ds.valid_time, valid_time)
            slice_samples = slice(
                None, counts_valid_times.sel(valid_time=valid_time) - 1
            )
            loc = {"valid_time": valid_time, "sample": slice_samples}
            dims_out = [dim for dim in res[var].dims if dim != "valid_time"]

            # Select and transpose data
            res[var].loc[loc] = (
                ds[var]
                .sel(init_time=indices["init_time"], lead_time=indices["lead_time"])
                .drop_vars(("init_time"))
                .transpose(*dims_out)
            )
        res[var].attrs = ds[var].attrs

    res["init_time"] = init_times
    return res


def get_mu_sigma(
    da: xr.DataArray, lead_time: np.timedelta64
) -> tuple[xr.DataArray, xr.DataArray]:
    """
    Calculate mean and standard deviation of reforecast dataset
    from data at a given lead time.

    Parameters
    ----------
    da : xr.DataArray
        Data array with ensemble_member dimension.
    lead_time : np.timedelta64
        Lead time for which to calculate statistics.

    Returns
    -------
    tuple[xr.DataArray, xr.DataArray]
        Mean and standard deviation values.
    """
    sigma = np.sqrt(
        da.sel(lead_time=lead_time)
        .std(dim="ensemble_member", ddof=1)  # use adjusted formula
        .mean(("latitude", "longitude", "init_time"))
        .compute()
    )
    mu = da.mean().compute()
    return mu, sigma


def store_mu_sigma(
    mu: dict[str, xr.DataArray],
    sigma_mean: dict[str, xr.DataArray],
    sigma_climatology: dict[str, xr.DataArray],
    out_path: str,
    lead_time: np.timedelta64,
) -> None:
    """
    Store mean and standard deviation values to NetCDF file.

    Parameters
    ----------
    mu : dict[str, xr.DataArray]
        Dictionary of mean values for each variable.
    sigma_mean : dict[str, xr.DataArray]
        Dictionary of mean standard deviation values for each variable.
    sigma_climatology : dict[str, xr.DataArray]
        Dictionary of standard deviation climatology values for each variable.
    out_path : str
        Output file path.
    lead_time : np.timedelta64
        Lead time for which statistics were calculated.
    """
    ds = xr.Dataset(coords={"lead_time": lead_time})

    for k, v in mu.items():
        ds[f"mu_{k}"] = v

    for k, v in sigma_mean.items():
        ds[f"sigma_{k}"] = v

    for k, v in sigma_climatology.items():
        ds[f"sigma_climatology_{k}"] = v
    ds.to_netcdf(out_path)


def get_sigma_climatology(
    da: xr.DataArray, lead_time: np.timedelta64, half_window_size: int = 45
) -> xr.DataArray:
    """
    Calculate climatological standard deviation for a given lead time.

    Parameters
    ----------
    da : xr.DataArray
        Data array with ensemble_member dimension.
    lead_time : np.timedelta64
        Lead time for which to calculate standard deviation climatology.
    half_window_size : int, optional
        Half window size for climatology calculation, by default 45.

    Returns
    -------
    xr.DataArray
        Climatological standard deviation data.
    """
    sigma = np.sqrt(
        da.sel(lead_time=lead_time)
        .std(dim="ensemble_member", ddof=1)
        .mean(("latitude", "longitude"))
        .compute()
    ).rename({"init_time": "time"})

    # Adjust time coordinate to valid_time
    sigma["time"] = sigma["time"] + lead_time

    return interpolate_then_rolling_mean(
        data=sigma.groupby("time.dayofyear").mean(),
        half_window_size=half_window_size,
        interp_method="linear",
    )


def merge_restructure_reforecasts(
    dt_used_in_sim: list[int],
    dt_mean_variance: int,
    path_input: str,
    path_output: str,
    path_output_mu_sigma: str,
    standardization: str | None = None,
) -> None:
    """
    Merge and restructure reforecast data into valid_time/sample format with
    optional standardization.

    This function takes reforecast data in (init_time, lead_time) format and
    restructures it into (valid_time, sample) dimensions that we introduce to keep
    the weather generator's similarity computations memory and compute efficient.
    It optionally applies standardization using mean and standard deviation statistics
    computed over the ensemble members. The function also prepares data for weather
    generators that compute similarities over more than one timestep by introducing a
    `lag` dimension.

    Parameters
    ----------
    dt_used_in_sim : list[int]
        List of time lags (in days) to be used in similarity calculations.
    dt_mean_variance : int
        Lead time (in days) used for computing mean and standard deviation statistics.
    path_input : str
        Path to the input Zarr store containing reforecast data.
    path_output : str
        Path where the restructured dataset will be saved as NetCDF.
    path_output_mu_sigma : str
        Path where the mean and standard deviation statistics will be saved.
    standardization : str | None, optional
        Standardization method to apply. Options are:
        - "constant": Standardize using same mean and standard deviation everywhere.
        - "no_standardization": No standardization applied.
        - None: No standardization, all stats set to NaN.
        Default is None.

    Raises
    ------
    ValueError
        If the difference between lags in `dt_used_in_sim` exceeds the length of the
        `lead_time` dimension in the input dataset, or if 0 is
        not included in `dt_used_in_sim`.
    """
    a_dt_used_in_sim = np.array(dt_used_in_sim)
    # Load data
    ds = xr.open_zarr(path_input, decode_timedelta=True)

    # Validate lead time parameters

    if np.max(a_dt_used_in_sim) - np.min(a_dt_used_in_sim) >= len(ds.lead_time):
        raise ValueError(
            "Difference between lags cannot be larger than lead_time dimension length."
        )
    assert 0 in a_dt_used_in_sim, "0 must be included in 'dt_used_in_sim'."

    ic("Restructure data.")

    # Prepare time arrays
    dt_used_in_sim = np.array(a_dt_used_in_sim) * np.timedelta64(1, "D").astype(
        "timedelta64[ns]"
    )
    sel_lead_times = ds.lead_time.isel(
        lead_time=slice(
            -np.amin(a_dt_used_in_sim),
            len(ds.lead_time) - np.amax(a_dt_used_in_sim),
        )
    )
    assert np.isin(
        sel_lead_times
        + xr.DataArray(np.timedelta64(1, "D") * a_dt_used_in_sim, dims="lag"),
        ds.lead_time,
    ).all(), "Invalid setup. Would require accessing lead_times that are not available."

    # Initialize statistics dictionaries
    mu = {}
    sigma_mean = {}
    sigma_clim = {}

    # Calculate statistics based on standardization method
    lt = (dt_mean_variance * np.timedelta64(1, "D")).astype("timedelta64[ns]")
    assert lt in ds.lead_time, (
        f"Chosen value for dt_mean_variance ({dt_mean_variance}d) is not in dataset."
    )

    ic("Store mu sigma")

    if standardization is None:
        # No standardization - set all to NaN
        for var in ds.data_vars:
            mu[var], sigma_mean[var] = xr.DataArray(np.nan), xr.DataArray(np.nan)
            sigma_clim[var] = xr.DataArray(np.nan)
        store_mu_sigma(
            mu=mu,
            sigma_mean=sigma_mean,
            sigma_climatology=sigma_clim,
            out_path=path_output_mu_sigma,
            lead_time=lt,
        )

        # Restructure data
        ds = ds.assign_coords(valid_time=ds.init_time + ds.lead_time)
        ds_res = to_valid_time_sample((ds.sel(lead_time=sel_lead_times)).compute())
    else:
        # Standardization enabled - calculate statistics
        for var in ds.data_vars:
            mu[var], sigma_mean[var] = get_mu_sigma(
                ds[var],
                lead_time=lt,
            )
            sigma_clim[var] = get_sigma_climatology(ds[var], lead_time=lt)

        # Store statistics
        store_mu_sigma(
            mu=mu,
            sigma_mean=sigma_mean,
            sigma_climatology=sigma_clim,
            out_path=path_output_mu_sigma,
            lead_time=lt,
        )

        # Restructure data with standardization
        ds = ds.assign_coords(valid_time=ds.init_time + ds.lead_time)

        if standardization == "constant":
            ds_res = to_valid_time_sample(
                ((ds.sel(lead_time=sel_lead_times) - mu) / sigma_mean).compute()
            )
        elif standardization == "no_standardization":
            ds_res = to_valid_time_sample(ds.sel(lead_time=sel_lead_times).compute())
        elif standardization == "climatology":
            raise NotImplementedError(
                "Climatological standardization not implemented yet."
            )
        else:
            raise ValueError("Invalid standardization argument.")

    # Calculate lead time and init_time
    lead_time = ds_res.valid_time - ds_res.init_time
    init_time = ds_res.init_time
    m = ~np.isnat(lead_time)
    ic(m)

    # Fill data
    for var in ds_res.data_vars:
        if var != "init_time":
            # The weather generator can compute similarities between
            # groups of dates. Set up arrayset to do this.
            da_list = [ds_res[var]] * len(dt_used_in_sim)
            da_list = [
                da_i.expand_dims({"lag": np.array([lag])}, axis=-3)
                for (da_i, lag) in zip(da_list, dt_used_in_sim)
            ]
            stacked_da = xr.concat(da_list, dim="lag")

            # Prepare original data for lagging
            da = (
                ds[var]
                .drop_vars("valid_time")
                .transpose(
                    "init_time",
                    "lead_time",
                    "ensemble_member",
                    "latitude",
                    "longitude",
                )
            )

            # Fill data
            for lag in tqdm(
                dt_used_in_sim, desc=f"Copy lagged values for variable {var}"
            ):
                ic(lag, lead_time)
                ic(lag + lead_time)
                ic(init_time)
                ic(
                    da.sel(
                        lead_time=lead_time + lag,
                        init_time=init_time,
                        method="nearest",
                    )
                )
                ic(
                    da.sel(
                        lead_time=lead_time + lag,
                        init_time=init_time,
                        method="nearest",
                    ).data
                )
                ic(stacked_da)
                ic(stacked_da.loc[{"lag": lag}])
                ic(stacked_da.loc[{"lag": lag}].data)
                ic(stacked_da.loc[{"lag": lag}].data[m, ...].shape)
                if lag != np.timedelta64(0):
                    stacked_da.loc[{"lag": lag}].data[m, ...] = (
                        da.sel(
                            lead_time=lead_time + lag,
                            init_time=init_time,
                            method="nearest",
                        )
                        .load()
                        .data[m, ...]
                    )
            ds_res[var] = stacked_da

    # Save final result
    ds_res.to_netcdf(path_output)


@snakemake_handler
def main(snakemake: Any) -> None:
    all_params = snakemake.params.all_params.copy()
    with open(snakemake.output.params, "w") as f:
        yaml.dump(
            snakemake.params.tracked_params.copy(),
            f,
            default_flow_style=False,
            sort_keys=False,
        )
    merge_restructure_reforecasts(
        dt_used_in_sim=all_params["merge_restructure_ds.dt_used_in_sim"],
        dt_mean_variance=all_params["merge_restructure_ds.dt_mean_variance"],
        standardization=all_params["merge_restructure_ds.standardization"],
        path_input=snakemake.input.path_zarr,
        path_output=snakemake.output.nc_file,
        path_output_mu_sigma=snakemake.output.nc_mu_sigma,
    )


if __name__ == "__main__":
    main(snakemake)  # noqa: F821
