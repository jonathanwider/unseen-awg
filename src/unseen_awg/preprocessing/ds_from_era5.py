"""
Preprocessing script to merge ERA5 data files and store them in a Zarr store.

This module combines multiple preprocessed netCDF files containing ERA5 data
into a single dataset, and saves the result in Zarr format.
"""

from typing import Any

import xarray as xr
import yaml

from unseen_awg.snakemake_utils import snakemake_handler


def ds_from_era5(paths_input: list[str], path_output: str) -> None:
    """
    Merge multiple ERA5 netCDF files into a single Zarr store with a format compatible
    to reforecast preprocessing pipeline.

    This function reads multiple preprocessed ERA5 netCDF files, combines them into
    a single xarray Dataset, and restructures the time dimensions to match the
    expected format for weather generator preprocessing pipeline. It converts valid_time
    to lead_time, adds ensemble_member and init_time dimensions, and removes coordinate
    variables that don't correspond to dimensions.

    Parameters
    ----------
    paths_input : list[str]
        List of file paths to the input netCDF files containing ERA5 data.
    path_output : str
        Path where the resulting Zarr store will be saved.
    """
    ds = xr.open_mfdataset(paths_input, join="outer")
    t0 = ds.valid_time[0].data
    ds["valid_time"] = ds.valid_time - t0
    ds = ds.rename({"valid_time": "lead_time"})

    ds = ds.expand_dims({"ensemble_member": [0]}).expand_dims(
        {"init_time": [t0]}, axis=(1)
    )

    ds = ds.assign_coords()

    # Reset init time and lead time and write datasets to files
    old_init_time = ds["init_time"].data
    ds["init_time"] = ds["init_time"] + ds["lead_time"].isel(lead_time=0)
    ds["lead_time"] = ds["lead_time"] + (old_init_time - ds["init_time"].data)

    for var in ds.data_vars:
        if "coordinates" in ds[var].encoding:
            # Remove the encoding coordinates attribute entirely
            del ds[var].encoding["coordinates"]

    # Identify coordinates without dimensions
    # coords_to_drop = [coord for coord in ds.coords if coord not in ds.dims]
    # ds = ds.drop_vars(coords_to_drop)

    ds.attrs = {
        "Conventions": "CF-1.7",
        "Title": f"Daily aggregate ERA5 {'/'.join(ds.data_vars)}",
        "Source": """Contains modified 'ERA5 post-processed daily statistics on single
        levels from 1940 to present' data retrieved from the
        Copernicus Climate Data Store under a CC-BY 4.0 license.""",
    }

    # assign names & units to synthetic dimensions:
    ds["ensemble_member"].attrs["standard_name"] = "realization"
    ds["ensemble_member"].attrs["long_name"] = (
        "ensemble member numerical id (synthetic dim for unseen-awg data format)"
    )
    ds["ensemble_member"].attrs["units"] = "1"

    ds["init_time"].attrs["standard_name"] = "forecast_reference_time"
    ds["init_time"].attrs["long_name"] = (
        "initial time of forecast (synthetic dim for unseen-awg data format)"
    )

    ds["lead_time"].attrs["standard_name"] = "forecast_period"
    ds["lead_time"].attrs["long_name"] = (
        "time since forecast_reference_time (synthetic dim for unseen-awg data format)"
    )

    ds.chunk({"lead_time": 100}).to_zarr(path_output)


@snakemake_handler
def main(snakemake: Any) -> None:
    with open(snakemake.output.params, "w") as f:
        yaml.dump(
            snakemake.params.tracked_params.copy(),
            f,
            default_flow_style=False,
            sort_keys=False,
        )
    ds_from_era5(paths_input=snakemake.input, path_output=snakemake.output.zarr)


if __name__ == "__main__":
    main(snakemake=snakemake)  # noqa: F821
