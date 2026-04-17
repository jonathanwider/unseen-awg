"""
Preprocessing script for single ERA5 variable files.

This module processes individual ERA5 netCDF files, performs regridding to
target grids, applies unit conversions, and saves the processed data in
NetCDF format.
"""

import os
from typing import Any, Literal

import metpy.calc as mpcalc
import xarray as xr
import xesmf as xe
from icecream import ic
from metpy.units import units

from unseen_awg.grids import (
    GRID_ANALOGS,
    GRID_ANALOGS_SMALL,
    GRID_ANALOGS_TINY,
    GRID_IMPACTS,
    GRID_IMPACTS_AR6_EU,
    GRID_IMPACTS_AR6_EU_TINY,
)
from unseen_awg.snakemake_utils import snakemake_handler


def ds_from_era5_single(
    variables: dict[str, Any],
    var: str,
    grid: str,
    var_type: Literal["circulation", "impact_variables"],
    dir_era5_raw: str,
    year: str,
    path_output: str,
) -> None:
    """Process invidual daily ERA5 netCDF files and save in NetCDF format.

    This function processes raw ERA5 netCDF files (one per year), performs regridding to
    target grids, applies unit conversions, and saves the processed data
    in NetCDF format.

    Parameters
    ----------
    variables : dict[str, Any]
        Dictionary containing variable configurations.
    var : str
        Name of the variable to process.
    grid : str
        Name of target grid of regridding..
    var_type : Literal["circulation", "impact_variables"]
        Type of variable being processed.
    dir_era5_raw : str
        Directory containing raw ERA5 data.
    year : str
        Year of data to process.
    path_output : str
        Path where output NetCDF file will be saved.

    Raises
    ------
    FileNotFoundError
        If the input file does not exist.
    ValueError
        If invalid grid or variable type is specified.
    """
    used_var = {var: variables[var]}
    ic(used_var)
    # Set up conversions and unit conversions
    conversions: dict[str, Any] = {}
    for k, v in used_var.items():
        if "conversion" in v.keys():
            conversions[k] = getattr(mpcalc, v["conversion"])

    unit_conversions: dict[str, str] = {}
    for k, v in used_var.items():
        unit_conversions[k] = v["units"]

    # Determine target grid based on variable type and grid wildcard
    if var_type == "circulation":
        if grid == "analogs":
            grid = GRID_ANALOGS
        elif grid == "analogs_small":
            grid = GRID_ANALOGS_SMALL
        elif grid == "analogs_tiny":
            grid = GRID_ANALOGS_TINY
        else:
            raise ValueError(f"Invalid target grid for circulation: {grid}")
    elif var_type == "impact_variables":
        if grid == "analogs":
            grid = GRID_ANALOGS
        elif grid == "analogs_small":
            grid = GRID_ANALOGS_SMALL
        elif grid == "analogs_tiny":
            grid = GRID_ANALOGS_TINY
        elif grid == "impacts":
            grid = GRID_IMPACTS
        elif grid == "impacts_AR6_EU":
            grid = GRID_IMPACTS_AR6_EU
        elif grid == "impacts_AR6_EU_tiny":
            grid = GRID_IMPACTS_AR6_EU_TINY
        else:
            raise ValueError(f"Invalid grid for impact variable: {grid}")
    else:
        raise ValueError(f"Invalid variable type {var_type}")

    cf_final_cell_methods = {
        k: v["cf_final_cell_methods"]
        for k, v in used_var.items()
        if "cf_final_cell_methods" in v.keys()
    }

    cf_standard_names = {
        k: v["cf_standard_name"]
        for k, v in used_var.items()
        if "cf_standard_name" in v.keys()
    }

    cf_additional_coordinates = {
        k: v["cf_additional_coordinate"]
        for k, v in used_var.items()
        if "cf_additional_coordinate" in v.keys()
    }

    cf_final_long_names = {
        k: v["cf_final_long_name"]
        for k, v in used_var.items()
        if "cf_final_long_name" in v.keys()
    }

    cf_final_standard_names = {
        k: v["cf_final_standard_name"]
        for k, v in used_var.items()
        if "cf_final_standard_name" in v.keys()
    }

    # Process each variable
    for name, var_config in used_var.items():
        in_file = os.path.join(
            dir_era5_raw,
            var_config["era5"]["file_name"],
            f"{year}.nc",
        )

        # Check if file exists
        if not os.path.exists(in_file):
            raise FileNotFoundError(f"Input file not found: {in_file}")

        ds_0 = xr.open_dataset(in_file)
        # Switch coordinate range from [0,360] to [-180,180]
        ds_0["longitude"] = (ds_0["longitude"] + 180) % 360 - 180
        ds_0 = ds_0.sortby("longitude").sortby("latitude")

        regridder = xe.Regridder(
            ds_0,
            grid,
            method=var_config["regridding_type"],
        )

        da = xr.open_dataset(in_file, decode_timedelta=True)[
            var_config["era5"]["var_name"]
        ].squeeze()  # squeeze to avoid trivial level dimension.
        # Switch coordinate range from [0,360] to [-180,180]
        da["longitude"] = (da["longitude"] + 180) % 360 - 180
        da = da.sortby("longitude").sortby("latitude")
        da = regridder(da, keep_attrs=True).metpy.quantify()

        # Apply conversions and unit transformations
        if "conversion" in var_config.keys():
            da = (
                conversions[name](da)
                .metpy.convert_units(var_config["units"])
                .metpy.dequantify()
            )
            da.attrs["units"] = f"{da.metpy.units:~}"
        else:
            da = da.metpy.convert_units(var_config["units"]).metpy.dequantify()
            da.attrs["units"] = f"{da.metpy.units:~}"

        # Special handling for precipitation data
        if name == "tp":
            da = da.where(
                da.metpy.quantify() > 0 * units.millimeter, 0
            ).metpy.dequantify()

        ic(path_output)
        ds = da.rename(name).to_dataset()

        ds.attrs = {k: v for k, v in ds.attrs.items() if not k.startswith("GRIB")}
        for name in list(ds.variables.keys()):
            ds[name].attrs = {
                k: v for k, v in ds[name].attrs.items() if not k.startswith("GRIB")
            }

        ds = ds.drop_vars(
            ["surface", "isobaricInhPa", "pressure_level", "number"], errors="ignore"
        )

        for var in ds.data_vars:
            if var in cf_final_cell_methods.keys():
                ds[var].attrs["cell_methods"] = cf_final_cell_methods[var]
            if var in cf_standard_names.keys():
                ds[var].attrs["standard_name"] = cf_standard_names[var]
            # lazy - the two seperate attributes were necessary for reforefasts
            if var in cf_final_standard_names.keys():
                ds[var].attrs["standard_name"] = cf_final_standard_names[var]
            if var in cf_additional_coordinates.keys():
                cf_coords = cf_additional_coordinates[
                    var
                ]  # by default this will only be one coord.
                ds[var] = ds[var].assign_coords(
                    {
                        k: xr.DataArray(
                            v["value"],
                            attrs={
                                k_attr: v_attr
                                for k_attr, v_attr in v.items()
                                if k_attr != "value"
                            },
                        )
                        for k, v in cf_coords.items()
                    }
                )
            if var in cf_final_long_names.keys():
                ds[var].attrs["long_name"] = cf_final_long_names[var]
        ds = ds.metpy.dequantify()

    ds.to_netcdf(path_output)


@snakemake_handler
def main(snakemake: Any) -> None:
    ds_from_era5_single(
        variables=snakemake.config["variables"],
        var=snakemake.wildcards.var,
        var_type=snakemake.wildcards.var_type,
        grid=snakemake.wildcards.grid,
        year=snakemake.wildcards.year,
        dir_era5_raw=snakemake.config["paths"]["dir_ERA5Raw"],
        path_output=snakemake.output.nc_file,
    )


if __name__ == "__main__":
    main(snakemake=snakemake)  # noqa: F821
