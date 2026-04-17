from typing import Any

import dask.array
import numpy as np
import xarray as xr
from tqdm.auto import tqdm

from unseen_awg.snakemake_utils import snakemake_handler


def to_year_doy_format(input_path: str, target_path: str) -> None:
    """
    Convert a dataset's time dimension to year and day of year format.

    This function transforms a dataset by creating a new dataset with separate
    year and day of year dimensions, preserving other original coordinates.

    Parameters
    ----------
    input_path : str
        Path to load input dataset from. Should be a .nc file in "restructured" format.
    target_path : str
        Path where the transformed dataset will be saved as a Zarr store.

    Notes
    -----
    - Assumes all relevant variables have similar dimension structures.
    - Fills the new dataset with NaN values initially.
    - Writes data to Zarr store by iterating through days of the year.
    """
    # Set up dataset to be filled
    ds = xr.open_dataset(input_path, decode_timedelta=True)
    ds_new = xr.Dataset(
        coords={
            "dayofyear": xr.DataArray(
                np.arange(1, 367),
                dims="dayofyear",
            ),
            "year": xr.DataArray(
                np.unique(ds.valid_time.dt.year),
                dims="year",
            ),
            "sample": ds.sample.data,
            "ensemble_member": ds.ensemble_member.data,
            "lag": ds.lag.data,
            "latitude": ds.latitude.data,
            "longitude": ds.longitude.data,
        }
    )

    # assume all variables of relevance in ds have
    # the same dimensions and coordinates as the main data set.
    for var in ds.data_vars:
        var_dims = [
            d for d in ds_new.dims if (d in ds[var].dims or d in ["year", "dayofyear"])
        ]
        var_l_dims = [len(ds_new[d]) for d in var_dims]
        var_l_chunks = [len(ds_new[d]) if d != "dayofyear" else 1 for d in var_dims]

        ds_new[var] = (
            var_dims,
            dask.array.full(
                var_l_dims, fill_value=np.nan, chunks=var_l_chunks, dtype=ds[var].dtype
            ),
        )
        ds_new[var].attrs = ds[var].attrs

    ds_new.to_zarr(target_path, compute=False)

    daysofyear = np.arange(1, 367)

    for var in ds.data_vars:
        for doy in tqdm(daysofyear, desc="Iterate over days of the year"):
            group = ds.valid_time.where(ds.valid_time.dt.dayofyear == doy, drop=True)
            coords_single_doy = {
                k: v for (k, v) in ds_new[var].coords.items() if k not in ["dayofyear"]
            }
            l_dims_single_doy = [len(v) for v in coords_single_doy.values()]

            # set up temporary storage for data to be written to zarr.
            da_single_doy = xr.DataArray(
                np.full(l_dims_single_doy, fill_value=np.nan, dtype=ds[var].dtype),
                coords_single_doy,
                name=var,
            )
            da_single_doy.loc[{"year": group.dt.year}] = ds[var].sel(valid_time=group)
            # write to zarr.
            coords_to_drop = [
                c
                for c in [
                    "year",
                    "sample",
                    "ensemble_member",
                    "latitude",
                    "longitude",
                    "lag",
                ]
                if c in da_single_doy.dims
            ]
            da_single_doy.drop_vars(coords_to_drop).expand_dims(
                dim={"dayofyear": [doy]}, axis=0
            ).to_zarr(
                target_path,
                region={"dayofyear": slice(doy - 1, doy)},
            )


@snakemake_handler
def main(snakemake: Any) -> None:
    to_year_doy_format(
        snakemake.input.nc_file,
        snakemake.output.zarr_year_dayofyear,
    )


if __name__ == "__main__":
    main(snakemake=snakemake)  # noqa: F821
