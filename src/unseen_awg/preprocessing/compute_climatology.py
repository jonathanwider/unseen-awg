"""
Compute climatological means.

This module provides functions to compute climatological means from time series data,
with support for interpolation, rolling window averaging, and handling different
data structures like ERA5 and reforecast datasets.
"""

from typing import Any

import numpy as np
import xarray as xr
import yaml
from icecream import ic
from metpy.units import units
from tqdm.auto import tqdm

from unseen_awg.snakemake_utils import snakemake_handler


def interpolate_then_rolling_mean(
    data: xr.Dataset | xr.DataArray,
    half_window_size: int,
    interp_method: str = "linear",
) -> xr.Dataset | xr.DataArray:
    """
    Temporally interpolate day-of-year data and compute rolling mean for
    climatological data.

    Handles circular interpolation and averaging for day-of-year data.

    Parameters
    ----------
    data : xr.Dataset or xr.DataArray
        Input data with a 'dayofyear' dimension.
    half_window_size : int
        Half the size of the rolling window for mean computation.
    interp_method : str, optional
        Interpolation method, defaults to 'linear'.

    Returns
    -------
    xr.Dataset or xr.DataArray
        Interpolated and rolling-mean processed data.
    """
    # Disable bottleneck for more numerically accurate results
    xr.set_options(use_bottleneck=False)

    # Compute padding needed for circular interpolation
    trailing_leading_nans = (
        (data.dayofyear.min() - 1) + (366 - data.dayofyear.max())
    ).data

    # Ensure full day-of-year range
    data = data.reindex(dayofyear=np.arange(1, 367))

    # Interpolate with circular wrapping
    data_filled = (
        data.pad(pad_width={"dayofyear": trailing_leading_nans + 1}, mode="wrap")
        .interpolate_na(
            method=interp_method, dim="dayofyear", use_coordinate=False, max_gap=None
        )
        .isel(dayofyear=slice(trailing_leading_nans + 1, -(trailing_leading_nans + 1)))
    )

    # Compute rolling mean with circular padding
    return (
        data_filled.pad(pad_width={"dayofyear": half_window_size}, mode="wrap")
        .rolling(dayofyear=2 * half_window_size + 1, center=True)
        .mean()
        .isel(dayofyear=slice(half_window_size, -(half_window_size)))
    )


def get_clim(
    data: xr.Dataset | xr.DataArray,
    half_window_size: int,
    interp_method: str = "linear",
    n_partitions_lon: int = 50,
) -> xr.Dataset | xr.DataArray:
    """
    Compute climatology for temporal data, using longitude partitioning to
    reduce the memory footprint during the computation.

    Parameters
    ----------
    data : xr.Dataset or xr.DataArray
        Input data with 'valid_time' temporal dimension.
        Must not have 'init_time' or 'lead_time' dimensions.
    half_window_size : int
        Half window size of the rolling window for mean computation.
    interp_method : str, optional
        Interpolation method, defaults to 'linear'
    n_partitions_lon : int, optional
        Number of longitude partitions, defaults to 50.

    Returns
    -------
    xr.Dataset or xr.DataArray
        Climatological mean data.
    """
    res = []

    # Validate longitude partitioning
    assert len(data.longitude) >= n_partitions_lon, (
        f"Number of longitude partitions {n_partitions_lon} is larger than "
        f"longitude values {len(data.longitude)}"
    )

    # Compute climatology in longitude chunks
    for lons in tqdm(
        np.array_split(data.longitude, n_partitions_lon),
        desc="Compute climatology in longitude chunk",
        leave=False,
    ):
        # Group by day of year and compute mean
        data_re_group_lon_subset = (
            data.sel(longitude=lons).load().groupby("valid_time.dayofyear").mean()
        )

        # Average across ensemble members if present
        if "ensemble_member" in data_re_group_lon_subset.dims:
            data_re_group_lon_subset = data_re_group_lon_subset.mean("ensemble_member")

        # Interpolate and compute rolling mean
        res.append(
            interpolate_then_rolling_mean(
                data_re_group_lon_subset,
                half_window_size=half_window_size,
                interp_method=interp_method,
            )
        )

    # Combine results from longitude chunks
    res = xr.combine_by_coords(res)

    return res


def compute_climatology(
    data: xr.DataArray | xr.Dataset,
    slice_years: slice,
    half_window_size: int,
    interp_method: str = "linear",
    n_partitions_lon: int = 50,
) -> xr.Dataset | xr.DataArray:
    """
    Compute climatology for a given dataset, handling different data structures.

    Parameters
    ----------
    data : xr.DataArray or xr.Dataset
        Input data with 'init_time' and 'lead_time' dimensions.
    slice_years : slice
        Slice defining the years to include in climatology computation.
    half_window_size : int
        Half window size for rolling mean computation.
    interp_method : str, optional
        Interpolation method, defaults to 'linear'.
    n_partitions_lon : int, optional
        Number of longitude partitions, defaults to 50.

    Returns
    -------
    xr.Dataset or xr.DataArray
        Climatological means with processed temporal dimensions.
    """
    data = data.assign_coords(valid_time=data["init_time"] + data["lead_time"])

    if len(data.init_time) == 1:  # e.g. ERA5 data
        data = data.squeeze().swap_dims({"lead_time": "valid_time"})

        # subset to years we're interested in
        if slice_years.start is not None:
            data = data.where(
                (data.valid_time.dt.year >= slice_years.start),
                drop=True,
            )
        if slice_years.stop is not None:
            data = data.where(
                (data.valid_time.dt.year <= slice_years.stop),
                drop=True,
            )
        clim = get_clim(
            data,
            half_window_size=half_window_size,
            interp_method=interp_method,
            n_partitions_lon=n_partitions_lon,
        )
    else:  # e.g. reforecast data
        clim = []
        for lt in tqdm(data.lead_time, desc="Computing climatology for each lead time"):
            data_single_lt = data.sel(lead_time=lt).swap_dims(
                {"init_time": "valid_time"}
            )
            # subset to years we're interested in
            if slice_years.start is not None:
                data_single_lt = data_single_lt.where(
                    (data_single_lt.valid_time.dt.year >= slice_years.start),
                    drop=True,
                )
            if slice_years.stop is not None:
                data_single_lt = data_single_lt.where(
                    (data_single_lt.valid_time.dt.year <= slice_years.stop),
                    drop=True,
                )
            clim.append(
                get_clim(
                    data_single_lt,
                    half_window_size=half_window_size,
                    interp_method=interp_method,
                    n_partitions_lon=n_partitions_lon,
                ).expand_dims(lead_time=[lt.data])
            )
        clim = xr.combine_by_coords(clim)

    return clim


@snakemake_handler
def main(snakemake: Any) -> None:
    all_params = snakemake.params.all_params.copy()
    tracked_params = snakemake.params.tracked_params.copy()
    with open(snakemake.output.params, "w") as f:
        yaml.dump(tracked_params, f, default_flow_style=False, sort_keys=False)

    # parse_params:
    slice_lead_times_params = all_params["compute_climatology.slice_lead_times"]
    slice_ensemble_members_params = all_params[
        "compute_climatology.slice_ensemble_members"
    ]

    slice_lead_times = slice(
        *[
            entry * np.timedelta64(1, "D") if entry is not None else None
            for entry in slice_lead_times_params
        ],
    )

    # Load the dataset
    ds_in = xr.open_zarr(snakemake.input.zarr_rechunk, decode_timedelta=True).sel(
        lead_time=slice_lead_times,
        ensemble_member=slice(*slice_ensemble_members_params),
    )

    if "tp" in ds_in.data_vars:
        ic("Clamp dry-day precipitation to zero")
        ds_in["tp"] = (
            ds_in["tp"]
            .where(ds_in["tp"].metpy.quantify() > 1 * units.millimeter, 0)
            .metpy.dequantify()
        )

    # Compute climatology
    compute_climatology(
        data=ds_in,
        slice_years=slice(
            all_params["compute_climatology.year_min"],
            all_params["compute_climatology.year_max"],
        ),
        half_window_size=all_params["compute_climatology.half_window_size"],
        interp_method=all_params["compute_climatology.interp_method"],
        n_partitions_lon=all_params["compute_climatology.n_partitions_lon"],
    ).to_netcdf(snakemake.output.nc_clim)


if __name__ == "__main__":
    main(snakemake)  # noqa: F821
