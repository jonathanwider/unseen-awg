import numpy as np
import xarray as xr
import yaml
from icecream import ic
from metpy.units import units
from tqdm.auto import tqdm

from unseen_awg.preprocessing.compute_climatology import get_clim
from unseen_awg.snakemake_utils import snakemake_handler


def process_single_lt(
    data_re_single_lt: xr.DataArray | xr.Dataset,
    data_era: xr.DataArray | xr.Dataset,
    half_window_size: int,
    interp_method: str = "linear",
    split_mode: str = "chronological",
    n_partitions: int = 1,
    n_partitions_lon: int = 50,
    rng: np.random.Generator | None = None,
) -> tuple[xr.DataArray | xr.Dataset, xr.DataArray | xr.Dataset]:
    """
    Compute climatological mean difference between reforecasts and ERA5 data.

    This function calculates the difference in climatological mean between
    reforecasts and ERA5 data for each day of the year. Days-of-year with no values
    are filled using linear interpolation. The dataset can be split into
    subparts along the valid_time dimension to assess bias computation robustness.

    Parameters
    ----------
    data_re_single_lt : xr.DataArray or xr.Dataset
        Reforecasts dataset for a single lead time.
    data_era : xr.DataArray or xr.Dataset
        ERA5 dataset.
    half_window_size : int
        Half the size of the window for climatology computation.
    interp_method : str, optional
        Interpolation method, by default "linear".
    split_mode : str, optional
        Method for splitting the dataset, by default "chronological".
    n_partitions : int, optional
        Number of partitions to split the dataset, by default 1.
    n_partitions_lon : int, optional
        Number of longitude partitions, by default 50.
    rng : Optional[np.random.Generator], optional
        Random number generator for random splits, by default None.

    Returns
    -------
    tuple[xr.DataArray | xr.Dataset, xr.DataArray | xr.Dataset]
        A tuple containing climatological means for reforecasts and ERA5
    """

    # Split into n parts and assess bias on each part to assess how robust the bias is.
    len_init_time = len(data_re_single_lt.init_time)
    if split_mode == "random":
        assert rng is not None, (
            "Must provide a random generator when using random splits."
        )
        indices = np.arange(len_init_time)
        indices = rng.permutation(len_init_time)
        split_indices = np.array_split(indices, n_partitions)
    elif split_mode == "random-years":
        unique_years = np.unique(data_re_single_lt.init_time.dt.year)
        indices_years = rng.permutation(len(unique_years))
        split_i_years = np.array_split(indices_years, n_partitions)
        split_indices = []
        for i_years in split_i_years:
            split_indices.append(
                np.where(
                    np.isin(data_re_single_lt.init_time.dt.year, unique_years[i_years])
                )[0]
            )
    elif split_mode == "chronological":
        indices = np.arange(len_init_time)
        split_indices = np.array_split(indices, n_partitions)
    else:
        raise f"Invalid argument for split_mode: {split_mode}"

    res_era_groups = []
    res_re_groups = []

    for i, indices in enumerate(
        tqdm(split_indices, desc="Computing bias for sub-dataset")
    ):
        data_re_group = data_re_single_lt.isel(init_time=indices)
        data_era_group = data_era.sel(valid_time=data_re_group.valid_time)

        res_re_groups.append(
            get_clim(
                data_re_group,
                half_window_size=half_window_size,
                interp_method=interp_method,
                n_partitions_lon=n_partitions_lon,
            ).expand_dims({"group": [i]})
        )
        res_era_groups.append(
            get_clim(
                data_era_group,
                half_window_size=half_window_size,
                interp_method=interp_method,
                n_partitions_lon=n_partitions_lon,
            ).expand_dims({"group": [i]})
        )
    res_re_groups = xr.combine_by_coords(res_re_groups)
    res_era_groups = xr.combine_by_coords(res_era_groups)

    return res_re_groups, res_era_groups


def store_result_temporary(
    ds_res: xr.Dataset,
    output_path: str,
    lead_time_value: np.timedelta64,
    split_mode: str,
) -> None:
    """
    Write dataset to a temporary zarr file for a specific lead_time and split_mode.

    This function writes the computed bias results to a temporary zarr file
    that will later be merged with other results in a separate step.

    Parameters
    ----------
    ds_res : xr.Dataset
        The dataset subset to be stored.
    output_path : str
        Path to the output zarr file.
    lead_time_value : np.timedelta64
        The lead time value for this computation.
    split_mode : str
        The way the data is split into groups.
    """
    # Add lead_time and split_mode as coordinates
    ds_with_coords = ds_res.expand_dims(
        {
            "lead_time": [lead_time_value],
            "split_mode": [split_mode],
        }
    )

    # Write to temporary zarr file
    ds_with_coords.to_zarr(output_path, mode="w")


def compute_bias(
    path_era5: str,
    path_reforecasts: str,
    path_output_era5: str,
    path_output_reforecasts: str,
    i_lead_time: int,
    delta_lead_time_days: int,
    split_mode: str,
    seed: int,
    half_window_size: int,
    n_partitions: int,
    n_partitions_lon: int,
) -> None:
    """Compute bias between reforecasts and ERA5 data for a specific lead time.

    This function computes the climatological bias between reforecasts and ERA5
    data by calculating the difference in climatological means for each day of
    the year. It supports different splitting modes to assess bias computation
    robustness and stores results to zarr stores.

    Parameters
    ----------
    path_era5 : str
        Path to the zarr store containing ERA5 data.
    path_reforecasts : str
        Path to the zarr store containing reforecast data.
    path_output_era5 : str
        Path where the ERA5 climatological means will be saved.
    path_output_reforecasts : str
        Path where the reforecast climatological means will be saved.
    i_lead_time : int
        Index of the lead time to process.
    delta_lead_time_days: int
        Half-width of window of lead times considered for in the bias computation.
    split_mode : str
        Method for splitting the dataset ("chronological", "random", or "random-years").
    seed : int
        Random seed for reproducible results, used in splitting the dataset.
    half_window_size : int
        Half the size of the window for climatology computation.
    n_partitions : int
        Number of partitions to split the dataset into for robustness assessment.
    n_partitions_lon : int
        Allows partitioning by longitude to reduce memory footprint.

    Raises
    ------
    ValueError
        If an invalid split_mode is provided.
    """
    rng = np.random.default_rng(seed=seed)
    # preprocessing on arrays:
    ds_era5 = xr.open_zarr(
        path_era5,
        decode_timedelta=True,
    ).squeeze()
    ds_re = xr.open_zarr(
        path_reforecasts,
        decode_timedelta=True,
    )
    ds_re = ds_re.assign_coords(valid_time=ds_re["init_time"] + ds_re["lead_time"])
    ds_era5 = ds_era5.assign_coords(
        valid_time=ds_era5["init_time"].drop_vars("init_time") + ds_era5["lead_time"]
    ).drop_vars(("init_time", "lead_time", "ensemble_member"))

    # to keep bias computation comparable to later use, clamp precip below 1 mm to 0.
    if "tp" in ds_era5.data_vars:
        ds_era5["tp"] = (
            ds_era5["tp"]
            .where(ds_era5["tp"].metpy.quantify() > 1 * units.millimeter, 0)
            .metpy.dequantify()
        )
    else:
        raise ValueError("ds_era5 doesn't contain variable 'tp'.")

    if "tp" in ds_re.data_vars:
        ds_re["tp"] = (
            ds_re["tp"]
            .where(ds_re["tp"].metpy.quantify() > 1 * units.millimeter, 0)
            .metpy.dequantify()
        )
    else:
        raise ValueError("ds_re doesn't contain variable 'tp'.")

    ds_era5 = ds_era5.swap_dims({"lead_time": "valid_time"})

    ic(ds_re)
    ic(ds_era5)
    i_lt_start = max(0, i_lead_time - delta_lead_time_days)
    i_lt_end = min(len(ds_re.lead_time) - 1, i_lead_time + delta_lead_time_days)

    res_reforecasts, res_era5 = process_single_lt(
        data_re_single_lt=ds_re.isel(lead_time=slice(i_lt_start, i_lt_end + 1)),
        data_era=ds_era5,
        half_window_size=half_window_size,
        n_partitions=n_partitions,
        n_partitions_lon=n_partitions_lon,
        split_mode=split_mode,
        rng=rng,
    )

    ic(res_reforecasts)
    ic(res_era5)

    lead_time_value = ds_re.lead_time.isel(lead_time=i_lead_time).data

    # Store results to temporary files
    store_result_temporary(
        ds_res=res_reforecasts,
        output_path=path_output_reforecasts,
        lead_time_value=lead_time_value,
        split_mode=split_mode,
    )

    store_result_temporary(
        ds_res=res_era5,
        output_path=path_output_era5,
        lead_time_value=lead_time_value,
        split_mode=split_mode,
    )


@snakemake_handler
def main(snakemake) -> None:
    tracked_params = snakemake.params.tracked_params
    all_params = snakemake.params.all_params
    i_split_mode = int(snakemake.wildcards["i_lead_time"])
    split_mode = snakemake.wildcards["split_mode"]
    tracked_params["i_lead_time"] = i_split_mode
    tracked_params["split_mode"] = split_mode
    tracked_params["n_partitions"] = int(snakemake.wildcards.n_partitions)
    with open(snakemake.output.params, "w") as f:
        yaml.dump(tracked_params, f, default_flow_style=False, sort_keys=False)

    compute_bias(
        path_era5=snakemake.input.path_era5,
        path_reforecasts=snakemake.input.path_reforecasts,
        path_output_reforecasts=snakemake.output.reforecasts,
        path_output_era5=snakemake.output.era5,
        i_lead_time=int(snakemake.wildcards.i_lead_time),
        delta_lead_time_days=all_params["compute_bias.delta_lead_time_days"],
        split_mode=split_mode,
        seed=all_params["compute_bias.seed"],
        half_window_size=all_params["compute_bias.half_window_size"],
        n_partitions=int(snakemake.wildcards.n_partitions),
        n_partitions_lon=all_params["compute_bias.n_partitions_lon"],
    )


if __name__ == "__main__":
    main(snakemake=snakemake)  # noqa: F821
