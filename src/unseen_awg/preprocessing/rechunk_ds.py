from typing import Any

import dask.array
import numpy as np
import xarray as xr
from tqdm.auto import tqdm

from unseen_awg.snakemake_utils import snakemake_handler


def write_single_longitude(
    ds: xr.Dataset, longitude_index: int, output_path: str
) -> None:
    """
    Write a single longitude slice of a dataset to a Zarr store.

    Parameters
    ----------
    ds : xr.Dataset
        Input dataset to slice and write.
    longitude_index : int
        Index of the longitude slice to write.
    output_path : str
        Path to the Zarr store where the slice will be written.
    """
    ds.isel(longitude=slice(longitude_index, longitude_index + 1)).drop_vars(
        ds.coords
    ).compute().to_zarr(
        output_path,
        region={"longitude": slice(longitude_index, longitude_index + 1)},
    )


def rechunk_dataset(path_input: str, path_output: str) -> None:
    """
    Rechunk a Zarr dataset to chunksize 1 along the longitude dimension.

    This function reads a Zarr dataset, creates a new dataset with the same
    coordinates but rechunked along the longitude dimension, and writes it
    to a new Zarr store. Each longitude slice is written individually to
    manage memory usage during the rechunking process.

    Parameters
    ----------
    path_input : str
        Path to the input Zarr store containing the dataset to be rechunked.
    path_output : str
        Path to the output Zarr store where the rechunked dataset will be saved.
    """
    ds = xr.open_zarr(path_input, decode_timedelta=True)

    ds_rechunked = xr.Dataset(coords=ds.coords)

    for variable_name in ds.data_vars:
        coords = {
            dim_name: ds[variable_name].coords[dim_name].data
            for dim_name in ds[variable_name].dims
        }
        longitude_chunks = [
            1 if dim_name == "longitude" else -1 for dim_name in ds[variable_name].dims
        ]
        ds_rechunked[variable_name] = xr.DataArray(
            dask.array.full(
                [len(c) for c in coords.values()], np.nan, chunks=longitude_chunks
            ),
            coords=coords,
        )
        ds_rechunked[variable_name].attrs = ds[variable_name].attrs

    ds_rechunked.reset_coords(drop=True).to_zarr(path_output, compute=False)

    for longitude_index in tqdm(range(len(ds.longitude))):
        write_single_longitude(ds, longitude_index, output_path=path_output)


@snakemake_handler
def main(snakemake: Any) -> None:
    rechunk_dataset(
        path_input=snakemake.input.path_zarr,
        path_output=snakemake.output.path_rechunk,
    )


if __name__ == "__main__":
    main(snakemake=snakemake)  # noqa: F821
