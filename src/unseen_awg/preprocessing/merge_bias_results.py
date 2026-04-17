"""Merge temporary bias computation results into final zarr stores."""

import os

import xarray as xr

from unseen_awg.data_utils import create_zarr_store_for_dataset
from unseen_awg.snakemake_utils import snakemake_handler


def merge_bias_results(
    tmp_files: list[str],
    output_path: str,
    all_lead_times: xr.DataArray,
) -> None:
    """
    Merge temporary bias result files into a single zarr store.

    Parameters
    ----------
    tmp_files : list[str]
        List of paths to temporary zarr files to merge.
    output_path : str
        Path to the output merged zarr file.
    all_lead_times : xr.DataArray
        Array containing all lead times in the dataset.
    """
    # Load all temporary files
    datasets = []
    for tmp_file in tmp_files:
        if os.path.exists(tmp_file):
            ds = xr.open_zarr(tmp_file)
            datasets.append(ds)

    if not datasets:
        raise ValueError("No temporary files found to merge")

    # Combine all datasets
    combined_ds = xr.combine_by_coords(datasets)

    # Ensure all lead_times and split_modes are present
    combined_ds = combined_ds.reindex(lead_time=all_lead_times, fill_value=float("nan"))

    # Create zarr store configuration
    config_vars = {}
    for v in combined_ds.data_vars:
        config_vars[v] = {
            "shape": combined_ds[v].shape,
            "chunks": combined_ds[v].chunks or combined_ds[v].shape,
            "dims": tuple(combined_ds[v].dims),
            "dtype": combined_ds[v].dtype,
        }

    coords = dict(combined_ds.coords)

    # Create and write to zarr store
    create_zarr_store_for_dataset(
        zarr_path=output_path,
        coords=coords,
        data_vars=config_vars,
    )

    combined_ds.to_zarr(output_path, mode="w")


@snakemake_handler
def main(snakemake) -> None:
    # Load reference dataset to get lead times
    all_lead_times = xr.open_zarr(
        snakemake.input.path_reforecasts,
        decode_timedelta=True,
    ).lead_time

    # Merge reforecasts results
    merge_bias_results(
        tmp_files=snakemake.input.reforecasts_tmp,
        output_path=snakemake.output.reforecasts,
        all_lead_times=all_lead_times,
    )

    # Merge ERA5 results
    merge_bias_results(
        tmp_files=snakemake.input.era5_tmp,
        output_path=snakemake.output.era5,
        all_lead_times=all_lead_times,
    )


if __name__ == "__main__":
    main(snakemake=snakemake)  # noqa: F821
