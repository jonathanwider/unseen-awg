"""Utilities for data manipulation and Zarr store creation.

This module provides functions for stacking xarray data structures and creating
Zarr stores for efficient data storage and retrieval.
"""

import os
from typing import Any

import dask.array
import numpy as np
import xarray as xr
from filelock import FileLock
from icecream import ic


def create_zarr_store_for_dataarray(
    zarr_path: str,
    shape: tuple[int, ...],
    chunks: tuple[int, ...],
    dims: tuple[str, ...],
    coords: dict[str, Any],
    dtype: np.dtype,
    variable_name: str,
) -> None:
    """Create a Zarr store for a single DataArray with metadata only.

    Parameters
    ----------
    zarr_path : str
        Path where the Zarr store will be created.
    shape : tuple[int, ...]
        Shape of the array.
    chunks : tuple[int, ...]
        Chunk sizes for each dimension.
    dims : tuple[str, ...]
        Dimension names.
    coords : dict[str, Any]
        Coordinate arrays and metadata.
    dtype : np.dtype
        Data type of the array.
    variable_name : str
        Name of the variable.

    Raises
    ------
    ValueError
        If shape, chunks, and dims have different lengths.
    """
    if not (len(shape) == len(chunks) == len(dims)):
        raise ValueError("The length of shape, chunks, and dims must be the same.")

    lock_path = zarr_path + ".lock"
    lock = FileLock(lock_path)

    with lock:
        if not os.path.exists(zarr_path):
            ic(f"Creating Zarr store metadata at: {zarr_path}")
            # Create a lazy Dask array with the specified properties
            dummy_dask_array = dask.array.empty(shape, chunks=chunks, dtype=dtype)

            # Create a template xarray DataArray
            template_da = xr.DataArray(
                dummy_dask_array, coords=coords, dims=dims, name=variable_name
            )
            # Convert to a Dataset to write to Zarr
            template_ds = template_da.to_dataset()

            # Write metadata only to the Zarr store
            template_ds.to_zarr(zarr_path, compute=False)
            ic("Zarr store metadata created successfully.")
        else:
            ic("Zarr store already exists. Skipping creation.")


def create_zarr_store_for_dataset(
    zarr_path: str,
    coords: dict[str, Any],
    data_vars: dict[str, dict[str, Any]],
) -> None:
    """Create a Zarr store for a Dataset with multiple variables.

    Parameters
    ----------
    zarr_path : str
        Path where the Zarr store will be created.
    coords : dict[str, Any]
        Coordinate arrays and metadata.
    data_vars : dict[str, dict[str, Any]]
        Dictionary mapping variable names to their specifications.
        Each specification should contain 'shape', 'chunks', 'dims', and 'dtype'.

    Raises
    ------
    ValueError
        If any variable has mismatched shape, chunks, and dims lengths.
    """
    lock_path = zarr_path + ".lock"
    lock = FileLock(lock_path)

    with lock:
        if not os.path.exists(zarr_path):
            ic(f"Creating Zarr store metadata at: {zarr_path}")

            data_arrays = {}
            for var_name, var_details in data_vars.items():
                shape = var_details["shape"]
                chunks = var_details["chunks"]
                dims = var_details["dims"]
                dtype = var_details["dtype"]

                if not (len(shape) == len(chunks) == len(dims)):
                    raise ValueError(
                        f"For variable '{var_name}', the length of shape, "
                        "chunks, and dims must be the same. "
                        f"But we have shape:{shape}, chunks:{chunks}, dims:{dims}"
                    )

                # Create a lazy Dask array with the specified properties
                dummy_dask_array = dask.array.empty(shape, chunks=chunks, dtype=dtype)

                # Create a template xarray DataArray
                template_da = xr.DataArray(dummy_dask_array, dims=dims, name=var_name)
                data_arrays[var_name] = template_da

            # Create a template xarray Dataset
            template_ds = xr.Dataset(data_arrays, coords=coords)

            # Write metadata only to the Zarr store
            template_ds.to_zarr(zarr_path, compute=False)
            ic("Zarr store metadata created successfully.")
        else:
            ic("Zarr store already exists. Skipping creation.")
