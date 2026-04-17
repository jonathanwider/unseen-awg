"""Correcting bias in weather reforecasts."""

from typing import Any

import numpy as np
import xarray as xr
import yaml
from icecream import ic
from tqdm.auto import tqdm

from unseen_awg.snakemake_utils import snakemake_handler


def correct_bias(
    path_ds_to_be_corrected: str,
    path_reforecasts: str,
    path_era5: str,
    path_output: str,
    init_time_partition_length: int,
    vars_to_correct: list[str],
) -> None:
    """Simple mean bias correction for reforecasts using ERA5 as reference data.

    This function applies bias correction to weather reforecasts by computing
    differences or rations between reforecast and ERA5 data, then applying these
    corrections to the raw reforecasts grouped by day of year.

    Parameters
    ----------
    path_ds_to_be_corrected : str
        Path to the zarr store containing the raw reforecasts to be corrected.
    path_reforecasts : str
        Path to the zarr store containing a precomputed reforecast climatology.
    path_era5 : str
        Path to the zarr store containing a precomputed ERA5 climatology.
    path_output : str
        Path where the bias-corrected dataset will be saved.
    init_time_partition_length : int
        Load init_time in parts of length init_time_partition_length to manage memory.
    vars_to_correct : list[str]
        Names of variables to apply bias correction to.

    Raises
    ------
    ValueError
        If the 'group' dimension is unexpectedly found in the reforecast or ERA5
        datasets. Computing biases allows splitting this dimension for diagnostic
        purposes but only datasets that aren't split along this dimension can be used to
        correct biases. An error is also raised if an unsupported variable is
        encountered during bias correction.
    """

    # Load uncorrected reforecasts
    ds_raw: xr.Dataset = xr.open_zarr(
        path_ds_to_be_corrected,
        decode_timedelta=True,
    )
    # Assign valid time coordinate
    ds_raw = ds_raw.assign_coords(valid_time=ds_raw["init_time"] + ds_raw["lead_time"])

    # Load bias data
    res_re: xr.Dataset = (
        xr.open_zarr(path_reforecasts).sel(split_mode="chronological").squeeze()
    )[list(vars_to_correct)]

    res_era: xr.Dataset = (
        xr.open_zarr(path_era5).sel(split_mode="chronological").squeeze()
    )[list(vars_to_correct)]
    if "group" in res_re.dims or "group" in res_era.dims:
        raise ValueError("Unexpected 'group' dimension in datasets")

    # Correct bias for each variable
    bias_dict = {}
    for var in res_re.data_vars:
        if var in ["t2m", "mn2t", "mx2t", "geopotential_height"]:
            bias_dict[var] = (res_re[var] - res_era[var]).load()
        elif var == "tp":
            bias_dict[var] = (res_re[var] / res_era[var]).load()

    ic("finished computing bias")

    ds_raw.to_zarr(path_output)

    indices = np.arange(len(ds_raw.init_time))
    init_time_partition_length = 10
    subsets_indices = np.array_split(
        indices, int(np.ceil(len(indices) / init_time_partition_length))
    )

    for idcs in tqdm(subsets_indices):
        ds_subset = ds_raw.isel(init_time=idcs).load()
        for var in res_re.data_vars:
            if var in ["t2m", "mn2t", "mx2t", "geopotential_height"]:
                ds_subset[var] = (
                    ds_subset[var].groupby(ds_subset.valid_time.dt.dayofyear)
                    - bias_dict[var]
                )
            elif var == "tp":
                ds_subset[var] = (
                    ds_subset[var].groupby(ds_subset.valid_time.dt.dayofyear)
                    / bias_dict[var]
                )
                ds_subset[var] = ds_subset[var].where(ds_subset[var] > 1, 0)
            else:
                raise ValueError(f"Unsupported variable for bias correction: {var}")
        region = {"init_time": slice(idcs[0], idcs[-1] + 1)}
        ds_subset.drop_vars(
            [
                "ensemble_member",
                "latitude",
                "lead_time",
                "longitude",
                "surface",
                "pressure",
                "dayofyear",
                "group",
                "split_mode",
            ],
            errors="ignore",
        ).to_zarr(path_output, mode="r+", region=region)


@snakemake_handler
def main(snakemake: Any) -> None:
    all_params = snakemake.params.all_params
    tracked_params = snakemake.params.tracked_params

    with open(snakemake.output.params, "w") as f:
        yaml.dump(tracked_params, f, default_flow_style=False, sort_keys=False)

    correct_bias(
        path_ds_to_be_corrected=snakemake.input.dataset_to_be_corrected,
        path_era5=snakemake.input.era5,
        path_reforecasts=snakemake.input.reforecasts,
        path_output=snakemake.output.path_reforecasts,
        init_time_partition_length=all_params[
            "correct_bias.init_time_partition_length"
        ],
        vars_to_correct=all_params["correct_bias.vars_to_correct"],
    )


if __name__ == "__main__":
    main(snakemake)  # noqa: F821
