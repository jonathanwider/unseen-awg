import datetime  # noqa: I001
import os
from typing import Any, Literal
import dask.array
import metpy.calc as mpcalc
import numpy as np
import xarray as xr
import xesmf as xe
import yaml
from icecream import ic
from metpy.units import units
from tqdm.auto import tqdm
from unseen_awg.grids import (
    GRID_ANALOGS,
    GRID_ANALOGS_SMALL,
    GRID_ANALOGS_TINY,
    GRID_IMPACTS_AR6_EU,
    GRID_IMPACTS_AR6_EU_TINY,
)

from unseen_awg.utils import grids_are_identical_subset

from unseen_awg.snakemake_utils import snakemake_handler


def process_save_analogs_single(
    file: str,
    input_dir: str,
    zarr_path: str,
    slice_sel_lead_times: slice,
    indices: xr.DataArray,
    slice_ensemble_member: slice,
    reforecast_vars: dict[str, Any],
    used_vars: list[str],
    target_grid: xr.Dataset,
    second_input_dir: str | None = None,
) -> None:
    """
    Process and save reforecast data of a single file

    Parameters
    ----------
    file : str
        Name of the input file to process.
    input_dir : str
        Directory containing input files.
    zarr_path : str
        Path to the output Zarr store.
    slice_sel_lead_times : slice
        Slice defining the range of lead times to include.
    indices : xr.DataArray
        Indices for init_time, used to assure that output is chronologically ordered.
    slice_ensemble_member : slice
        Slice defining the range of ensemble members to include.
    reforecast_vars : dict[str, Any]
        Dictionary of reforecast variable configurations.
    used_vars : list[str]
        List of variables to process.
    target_grid : xr.Dataset
        Target grid for regridding.
    second_input_dir : str or None, optional
        While some reforecast variables are instantaneous, others are accumulations.
        Downloads were split into parts because for the accumulated variables aren't
        defined at t=0. The secondary input directory contains a file with the same name
        as the one in the primary directory and with data of valid variables at t=0.

    """
    ds = preprocess_analogs_single(
        os.path.join(input_dir, file),
        slice_sel_lead_times=slice_sel_lead_times,
        slice_ensemble_member=slice_ensemble_member,
        reforecast_vars=reforecast_vars,
        used_vars=used_vars,
        target_grid=target_grid,
        second_input_dir=second_input_dir,
    )
    idcs = indices.sel(init_time=ds.init_time).data
    for i, idx in enumerate(idcs):
        ds.isel(init_time=slice(i, i + 1)).drop_vars(ds.coords).compute().to_zarr(
            zarr_path,
            region={
                "init_time": slice(idx, idx + 1),
                "ensemble_member": slice_ensemble_member,
            },
            safe_chunks=False,
        )


def preprocess_analogs(
    input_dir: str,
    zarr_path: str,
    slice_sel_lead_times: slice,
    indices: xr.DataArray,
    slice_ensemble_member: slice,
    reforecast_vars: dict[str, Any],
    used_vars: list[str],
    in_format: str,
    target_grid: xr.Dataset,
    second_input_dir: str | None = None,
) -> None:
    """
    Preprocess reforecast datasets and save to a Zarr store.

    Parameters
    ----------
    input_dir : str
        Directory containing input files.
    zarr_path : str
        Path to the output Zarr store.
    slice_sel_lead_times : slice
        Lead times to include in the final dataset.
    indices : xr.DataArray
        DataArray of indices for init times.
    slice_ensemble_member : slice
        Slice defining the range of ensemble members to save.
    reforecast_vars : dict[str, Any]
        Dictionary of reforecast variable configurations.
    used_vars : list[str]
        List of variables to process.
    in_format : str
        Input file format (e.g., 'grib').
    target_grid : xr.Dataset
        Target grid for regridding.
    second_input_dir : str or None, optional
        Secondary input directory for additional files, by default None.
    """
    files = [file for file in os.listdir(input_dir) if file.endswith(f".{in_format}")]

    for file in tqdm(files, desc=f"Process files in {input_dir}"):
        process_save_analogs_single(
            file,
            input_dir=input_dir,
            zarr_path=zarr_path,
            slice_sel_lead_times=slice_sel_lead_times,
            indices=indices,
            slice_ensemble_member=slice_ensemble_member,
            reforecast_vars=reforecast_vars,
            used_vars=used_vars,
            target_grid=target_grid,
            second_input_dir=second_input_dir,
        )


def resample_daily(ds: xr.Dataset, reductions: dict[str, str]) -> xr.Dataset:
    """
    Resample the input dataset to daily intervals.

    Parameters
    ----------
    ds : xr.Dataset
        Input dataset containing variables to be resampled.
    reductions : dict[str, str]
        Dictionary mapping variable names to their corresponding reduction method.
        Supported reduction methods include 'mean', 'sum', 'min', and 'max'.

    Returns
    -------
    xr.Dataset
        Dataset resampled to daily intervals with specified reduction methods.

    Notes
    -----
    Special handling is implemented for specific variables:
    - 'z' and 't2m' are not modified
    - 'mx2t6', 'mn2t6', and 'tp' are shifted by 6 hours before resampling because
    they are aggregated meteorological variables.
    - 'tp' (total precipitation) is processed to ensure non-negative values and
    setting 'drizzle' values to 0.
    """
    ds_out = xr.Dataset(attrs=ds.attrs.copy())

    for key in ds.data_vars:
        da = ds[key]
        original_attrs = da.attrs.copy()
        original_attrs.pop("cell_methods", None)

        if da.name in ["z", "t2m"]:
            pass
        elif da.name in ["mx2t6", "mn2t6", "tp"]:
            # for these three variables, the values are aggregated over the past 6 hours
            # We therefore shift the step axis by 6 hours before selecting steps.
            step_attrs = da.step.attrs.copy()
            da = da.assign_coords({"step": da.step - np.timedelta64(6, "h")})
            da.step.attrs = step_attrs
        else:
            raise ValueError(f"Time aggregation not implemented for variable {key}")

        ds_out[key] = apply_reduction(da=da, reduction=reductions[key])

        # Preserve original attributes
        ds_out[key].attrs.update(original_attrs)
        # Add information about the reduction method used
        if da.name == "tp":
            ds_out[key] = xr.apply_ufunc(
                np.diff,
                ds_out[key],
                input_core_dims=[["step"]],
                output_core_dims=[["step"]],
                vectorize=True,
                kwargs={"prepend": 0},
            ).compute()
            # Make sure attributes are preserved after the ufunc operation
            ds_out[key].attrs.update(original_attrs)
            ds_out[key] = (
                ds_out[key]
                .where(ds_out[key].metpy.quantify() > 0 * units.millimeter, 0)
                .metpy.dequantify()
            )  # assure non-negative values and remove drizzle
    return ds_out


def apply_reduction(da: xr.DataArray, reduction: str) -> xr.DataArray:
    """
    Apply a specified reduction method to a DataArray.

    Parameters
    ----------
    da : xr.DataArray
        Input DataArray to be reduced.
    reduction : str
        Reduction method to apply. Supported are 'mean', 'sum', 'min', and 'max'.

    Returns
    -------
    xr.DataArray
        DataArray reduced to daily intervals using the specified method.

    Raises
    ------
    ValueError
        If an unsupported reduction method is provided.
    """
    if reduction == "mean":
        res = (
            only_keep_days_with_all_timesteps(da)
            .resample(step="D")
            .mean(keep_attrs=True)
            .assign_attrs({"cell_methods": "lead_time: mean"})
        )
    elif reduction == "sum":
        res = (
            only_keep_days_with_all_timesteps(da)
            .resample(step="D")
            .sum(keep_attrs=True)
            .assign_attrs({"cell_methods": "lead_time: sum"})
        )
    elif reduction == "min":
        res = (
            only_keep_days_with_all_timesteps(da)
            .resample(step="D")
            .min(keep_attrs=True)
            .assign_attrs({"cell_methods": "lead_time: minimum"})
        )
    elif reduction == "max":
        res = (
            only_keep_days_with_all_timesteps(da)
            .resample(step="D")
            .max(keep_attrs=True)
            .assign_attrs({"cell_methods": "lead_time: maximum"})
        )
    else:
        raise ValueError(f"Unsupported reduction method for variable '{reduction}'")
    return res


def only_keep_days_with_all_timesteps(da: xr.DataArray) -> xr.DataArray:
    """
    Filter a DataArray to include only days with all timesteps present.

    Parameters
    ----------
    da : xr.DataArray
        Input DataArray to be filtered.

    Returns
    -------
    xr.DataArray
        DataArray containing only days with all 4 timesteps present.

    Notes
    -----
    This function ensures that only complete days (with 4 timesteps) are retained
    in the DataArray, dropping incomplete days.
    """
    steps = (da.step + da.time[0]).data
    t0_steps = xr.DataArray(steps, coords={"time": steps})
    counts_groups = t0_steps.resample(time="D").count(dim="time")
    counts = counts_groups.sel(
        time=(da.step + da.time[0]).dt.date.astype("datetime64[ns]")
    ).drop_vars("time")
    return da.where(counts == 4, drop=True)


def preprocess_analogs_single(
    file: str,
    slice_sel_lead_times: slice,
    slice_ensemble_member: slice,
    reforecast_vars: dict[str, Any],
    used_vars: list[str],
    target_grid: xr.DataArray,
    second_input_dir: str | None = None,
) -> xr.Dataset:
    """
    Preprocess a single reforecast file.

    Parameters
    ----------
    file : str
        Name of the input file to process.
    slice_sel_lead_times : slice
        Slice defining the range of lead times to include.
    slice_ensemble_member : slice
        Slice defining the range of ensemble members to include.
    reforecast_vars : dict[str, Any]
        Dictionary of reforecast variable configurations.
    used_vars : list[str]
        List of variables to include.
    target_grid : xr.DataArray
        Target grid for regridding.
    second_input_dir : str or None
        Secondary input directory for additional files, or None. Defaults to None.

    Returns
    -------
    xr.Dataset
        Preprocessed dataset with selected lead times.
    """
    secondary_file = (
        None
        if second_input_dir is None
        else os.path.join(second_input_dir, os.path.basename(file))
    )

    used_reforecast_vars = {k: v for k, v in reforecast_vars.items() if k in used_vars}
    reductions = {
        v["reforecasts"]["var_name"]: v["reduction"]
        for v in used_reforecast_vars.values()
    }

    regridding_types = {
        v["reforecasts"]["var_name"]: v["regridding_type"]
        for v in used_reforecast_vars.values()
    }
    ds = xr.open_dataset(file, decode_timedelta=True)

    if secondary_file is not None:
        secondary_ds = xr.open_dataset(secondary_file, decode_timedelta=True)
        ds = xr.concat((secondary_ds, ds), dim="step")
    ds = ds.drop_vars(["surface", "isobaricInhPa"], errors="ignore")

    cf_cell_methods = {
        v["reforecasts"]["var_name"]: v["cf_cell_methods"]
        for v in used_reforecast_vars.values()
        if "cf_cell_methods" in v.keys()
    }

    cf_standard_names = {
        v["reforecasts"]["var_name"]: v["cf_standard_name"]
        for v in used_reforecast_vars.values()
        if "cf_standard_name" in v.keys()
    }

    cf_additional_coordinates = {
        v["reforecasts"]["var_name"]: v["cf_additional_coordinate"]
        for v in used_reforecast_vars.values()
        if "cf_additional_coordinate" in v.keys()
    }

    cf_final_long_names = {
        v["reforecasts"]["var_name"]: v["cf_final_long_name"]
        for v in used_reforecast_vars.values()
        if "cf_final_long_name" in v.keys()
    }

    cf_final_standard_names = {
        v["reforecasts"]["var_name"]: v["cf_final_standard_name"]
        for v in used_reforecast_vars.values()
        if "cf_final_standard_name" in v.keys()
    }

    for var in ds.data_vars:
        if var in cf_cell_methods.keys():
            ds[var].attrs["cell_methods"] = cf_cell_methods[var]
        if var in cf_standard_names.keys():
            ds[var].attrs["standard_name"] = cf_standard_names[var]
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
        ds[var].attrs.pop(
            "long_name", None
        )  # keeping it would require handling names during aggregation

    # drop all attrs that start with GRIB.
    ds.attrs = {k: v for k, v in ds.attrs.items() if not k.startswith("GRIB")}
    for name in list(ds.variables.keys()):
        ds[name].attrs = {
            k: v for k, v in ds[name].attrs.items() if not k.startswith("GRIB")
        }

    if np.array_equal(target_grid.latitude, ds.latitude) and np.array_equal(
        target_grid.longitude, ds.longitude
    ):
        pass
    elif grids_are_identical_subset(
        ds, target_grid
    ):  # if we simply use a crop, make use of this.
        ic("use AR6 grid.")
        ds = ds.sel(latitude=target_grid.latitude, longitude=target_grid.longitude)
    else:
        ds_res = ds[[]]
        for var in ds.data_vars:
            regridder = xe.Regridder(ds[var], target_grid, method=regridding_types[var])
            ds_res[var] = regridder(ds[var], keep_attrs=True)
        ds_res["latitude"].attrs = ds["latitude"].attrs
        ds_res["longitude"].attrs = ds["longitude"].attrs
        ds = ds_res

    ds["latitude"].attrs.pop("stored_direction", None)
    ds["longitude"].attrs.pop("stored_direction", None)

    ds = resample_daily(ds, reductions=reductions)
    for var in ds.data_vars:
        if var in cf_final_long_names.keys():
            ds[var].attrs["long_name"] = cf_final_long_names[var]
    ds = time_dimension_to_init_time_lag(ds)

    ds = rename_convert_vars_and_units(
        ds, reforecast_vars=reforecast_vars, used_vars=used_vars
    )

    renames = {}
    for k, v in used_reforecast_vars.items():
        renames[k] = v["reforecasts"]["var_name"]

    for var in ds.data_vars:
        if renames[var] in cf_final_standard_names.keys():
            ds[var].attrs["standard_name"] = cf_final_standard_names[renames[var]]

    ic("dims of dataset after preprocessing:", ds.dims)

    ic(
        f"{datetime.datetime.now()} | Process file",
        file,
        "| Ensemble_member slice",
        slice_ensemble_member,
    )

    if "ensemble_member" not in ds.dims:
        ds = ds.expand_dims("ensemble_member", axis=0)

    return ds.sel(lead_time=slice_sel_lead_times)


def time_dimension_to_init_time_lag(
    d: xr.Dataset | xr.DataArray,
) -> xr.Dataset | xr.DataArray:
    """
    Rename time-related dimensions to our chosen names.

    Parameters
    ----------
    d : xr.Dataset or xr.DataArray
        Input dataset or dataarray with time-related dimensions.

    Returns
    -------
    xr.Dataset or xr.DataArray
        Dataset or dataarray with renamed dimensions:
        - 'time' -> 'init_time'
        - 'step' -> 'lead_time'
        - 'number' -> 'ensemble_member'
    """
    return d.rename(
        {"time": "init_time", "step": "lead_time", "number": "ensemble_member"}
    )


def drop_non_index_coordinates(ds: xr.Dataset) -> xr.Dataset:
    """
    Remove coordinates that do not have an index dimension.

    Parameters
    ----------
    ds : xr.Dataset
        Input dataset containing coordinates.

    Returns
    -------
    xr.Dataset
        Dataset with non-index coordinates removed.
    """
    return ds.drop_coords(lambda x: [v for v, da in x.coords.items() if not da.ndim])


def rename_convert_vars_and_units(
    ds: xr.Dataset,
    reforecast_vars: dict[str, Any],
    used_vars: list[str],
) -> xr.Dataset:
    """
    Rename, convert, and standardize units for reforecast variables.

    Parameters
    ----------
    ds : xr.Dataset
        Input dataset with original variable names and units.
    reforecast_vars : dict[str, Any]
        Dictionary of variable configurations including renaming and unit conversion.
    used_vars : list[str]
        List of variables to process.

    Returns
    -------
    xr.Dataset
        Dataset with renamed variables, converted units, and applied transformations.

    Notes
    -----
    This function performs the following operations:
    - Filters variables based on used_vars
    - Renames variables
    - Applies unit conversions
    - Executes optional conversion functions
    """
    used_reforecast_vars = {k: v for k, v in reforecast_vars.items() if k in used_vars}
    renames = {}
    for k, v in used_reforecast_vars.items():
        renames[v["reforecasts"]["var_name"]] = k

    conversions = {}
    for k, v in used_reforecast_vars.items():
        if "conversion" in v.keys():
            conversions[k] = getattr(mpcalc, v["conversion"])
    unit_conversions = {}
    for k, v in used_reforecast_vars.items():
        unit_conversions[k] = v["units"]

    # rename:
    ds = ds.rename(renames).metpy.quantify()

    # convert to other quantity:
    for var, conversion_function in conversions.items():
        attrs = ds[var].attrs
        attrs.pop("units", None)
        ds[var] = conversion_function(ds[var])
        ds[var].attrs.update(attrs)

    # convert units:
    for var, new_unit in unit_conversions.items():
        ds[var] = ds[var].metpy.convert_units(new_unit)

    ds = ds.metpy.dequantify()
    for k, v in used_reforecast_vars.items():
        ds[k].attrs["units"] = unit_conversions[k]
    return ds


def do_preprocessing(
    var_type: Literal["circulation", "impact_variables"],
    used_vars: list[str],
    target_grid_name: Literal[
        "analogs",
        "original",
        "analogs_small",
        "analogs_tiny",
        "impacts_AR6_EU",
        "impacts_AR6_EU_tiny",
    ],
    lead_time_days_low: int,
    lead_time_days_high: int,
    in_format: str,
    dir_reforecasts_raw: str,
    variables: dict[str, Any],
    path_output: str,
) -> None:
    """
    Preprocess reforecast data for either atmospheric circulation or impact variables.

    This function handles preprocessing for reforecast data, including reading
    raw files, applying regridding, resampling to daily intervals, renaming variables,
    converting units, and saving the result to a Zarr store.

    Parameters
    ----------
    var_type : Literal["circulation", "impact_variables"]
        Type of variables to process. Either "circulation" or "impact_variables".
    used_vars : list[str]
        List of names of variables to include in the preprocessing.
    target_grid_name : Literal[
        "analogs",
        "original",
        "analogs_small",
        "analogs_tiny",
        "impacts_AR6_EU",
        "impacts_AR6_EU_tiny",
    ]
        Name of the target grid to use for regridding.
    lead_time_days_low : int
        Lower bound of lead times in days to include from the reforecast dataset.
    lead_time_days_high : int
        Upper bound of lead times in days to include from the reforecast dataset.
    in_format : str
        Input file format (e.g., 'grib').
    dir_reforecasts_raw : str
        Directory containing raw reforecast data.
    variables : dict[str, Any]
        Dictionary of variable configurations.
    path_output : str
        Path where the preprocessed data will be saved as a Zarr store.

    Notes
    -----
    This function assumes a particular folder structure in which the reforecasts are
    saved. We assume that the input directory structure follows a specific format
    with 'cf' and 'pf'  subdirectories for control and perturbed forecasts respectively.
    For impact variables, it also handles special files at t=0 that are stored
    separately.
    """
    slice_sel_lead_times = slice(
        lead_time_days_low * np.timedelta64(1, "D").astype("timedelta64[ns]"),
        lead_time_days_high * np.timedelta64(1, "D").astype("timedelta64[ns]"),
    )

    input_dir = os.path.join(
        dir_reforecasts_raw,
        var_type,
    )

    if var_type == "circulation":
        step_zero_dir = None
    elif var_type == "impact_variables":
        step_zero_dir = os.path.join(dir_reforecasts_raw, "impact_variables_step_0")
    else:
        raise NotImplementedError(f"Invalid value for var_type: {var_type}")

    reforecast_vars = variables

    # check that both cf and pf directories contain the same files
    files_cf = [
        file
        for file in os.listdir(os.path.join(input_dir, "cf"))
        if file.endswith(f".{in_format}")
    ]
    files_pf = [
        file
        for file in os.listdir(os.path.join(input_dir, "pf"))
        if file.endswith(f".{in_format}")
    ]
    assert (np.sort(files_cf) == np.sort(files_pf)).all()

    # extract all init_time steps:
    init_times = np.array(
        [
            xr.open_dataset(
                os.path.join(input_dir, "cf", f), decode_timedelta=True
            ).time.data
            for f in files_cf
        ]
    )
    init_times = np.unique(init_times.flatten())

    # create a shared zarr store:
    path_ds_cf_single = os.path.join(input_dir, "cf", files_cf[0])
    ds_0_cf = xr.open_dataset(path_ds_cf_single, decode_timedelta=True).rename(
        {"time": "init_time", "step": "lead_time", "number": "ensemble_member"}
    )
    ds_0_pf = xr.open_dataset(
        os.path.join(input_dir, "pf", files_cf[0]), decode_timedelta=True
    ).rename({"time": "init_time", "step": "lead_time", "number": "ensemble_member"})

    ds_0_cf = rename_convert_vars_and_units(
        ds_0_cf, reforecast_vars=reforecast_vars, used_vars=used_vars
    )
    ds_0_pf = rename_convert_vars_and_units(
        ds_0_pf, reforecast_vars=reforecast_vars, used_vars=used_vars
    )

    assert target_grid_name in [
        "analogs",
        "original",
        "analogs_small",
        "analogs_tiny",
        "impacts_AR6_EU",
        "impacts_AR6_EU_tiny",
    ]
    if target_grid_name == "analogs":
        target_grid = GRID_ANALOGS
    elif target_grid_name == "analogs_small":
        target_grid = GRID_ANALOGS_SMALL
    elif target_grid_name == "analogs_tiny":
        target_grid = GRID_ANALOGS_TINY
    elif target_grid_name == "impacts_AR6_EU":
        target_grid = GRID_IMPACTS_AR6_EU
    elif target_grid_name == "impacts_AR6_EU_tiny":
        target_grid = GRID_IMPACTS_AR6_EU_TINY
    elif target_grid_name == "original":
        target_grid = xr.Dataset(
            {
                "latitude": ds_0_cf.latitude,
                "longitude": ds_0_cf.longitude,
            }
        )
    else:
        raise ValueError(f"Invalid target grid for interpolation: {target_grid_name}")

    ic("dims target grid:", target_grid.dims)

    ic("Setting up zarr store.")

    slices_ensemble_member = {
        "cf": slice(
            ds_0_cf.ensemble_member.min().data.item(),
            ds_0_cf.ensemble_member.max().data.item() + 1,
        ),
        "pf": slice(
            ds_0_pf.ensemble_member.min().data.item(),
            ds_0_pf.ensemble_member.max().data.item() + 1,
        ),
    }

    if step_zero_dir is not None:
        s_dir_cf_single = os.path.join(step_zero_dir, "cf")
    else:
        s_dir_cf_single = None

    # Do this once to be able to set up the lead_time coordinate for the zarr store.
    ds_cf_single = preprocess_analogs_single(
        file=path_ds_cf_single,
        slice_sel_lead_times=slice_sel_lead_times,
        slice_ensemble_member=slices_ensemble_member["cf"],
        reforecast_vars=reforecast_vars,
        used_vars=used_vars,
        target_grid=target_grid,
        second_input_dir=s_dir_cf_single,
    )
    ic(ds_cf_single.dims)

    ds_combined = xr.Dataset(
        coords={
            "latitude": target_grid.latitude,
            "longitude": target_grid.longitude,
            "ensemble_member": xr.concat(
                (ds_0_cf.ensemble_member, ds_0_pf.ensemble_member),
                dim="ensemble_member",
            )
            .drop_vars(["surface", "isobaricInhPa"], errors="ignore")
            .chunk({"ensemble_member": -1}),
            "init_time": xr.DataArray(
                init_times, attrs=ds_0_cf.init_time.attrs, dims="init_time"
            ),
            "lead_time": ds_cf_single.lead_time,
        },
        attrs=ds_cf_single.attrs,
    )

    for v in ds_cf_single.data_vars:
        dims = ds_cf_single[v].dims
        coords = {
            name: ds_combined[name]
            if name in ["latitude", "longitude", "ensemble_member"]
            else ds_cf_single[v].coords[name]
            for name in dims
        }
        shape = {name: len(coords[name]) for name in dims}
        chunks = {name: 10 if name == "init_time" else shape[name] for name in dims}

        ds_combined[v] = xr.DataArray(
            dask.array.full(
                [shape[name] for name in dims],
                np.nan,
                chunks=[chunks[name] for name in dims],
            ),
            dims=dims,
            coords=coords,
            attrs=ds_cf_single[v].attrs,
        )

    ds_combined.attrs = {
        "Conventions": "CF-1.7",
        "Title": f"Daily aggregate extended ensemble forecast hindcast {var_type}",
        "Source": """Contains modified “Extended ensemble forecast hindcast” data from
        the European Centre for Medium-Range Weather Forecasts (ECMWF). Data was
        retrieved from its Operational Archive and is licensed under CC BY 4.0.""",
    }

    ds_combined.to_zarr(path_output, compute=False)

    # used to look up positions in .zarr store.
    indices = xr.DataArray(np.arange(len(init_times)), coords={"init_time": init_times})

    # preprocess files
    for fc_type in ["cf", "pf"]:
        in_dir = os.path.join(input_dir, fc_type)
        if step_zero_dir is not None:
            s_dir = os.path.join(step_zero_dir, fc_type)
        else:
            s_dir = None
        preprocess_analogs(
            input_dir=in_dir,
            zarr_path=path_output,
            slice_sel_lead_times=slice_sel_lead_times,
            indices=indices,
            slice_ensemble_member=slices_ensemble_member[fc_type],
            reforecast_vars=reforecast_vars,
            used_vars=used_vars,
            in_format=in_format,
            target_grid=target_grid,
            second_input_dir=s_dir,
        )


@snakemake_handler
def main(snakemake) -> None:
    # store parameters that influence the output in a human-readable format.
    params = snakemake.params.all_params.copy()
    with open(snakemake.output.params, "w") as f:
        yaml.dump(
            snakemake.params.tracked_params.copy(),
            f,
            default_flow_style=False,
            sort_keys=False,
        )
    do_preprocessing(
        var_type=params[f"{snakemake.rule}.var_type"],
        used_vars=params[f"{snakemake.rule}.used_vars"],
        target_grid_name=params[f"{snakemake.rule}.target_grid"],
        lead_time_days_low=params["lead_time_days_low"],
        lead_time_days_high=params["lead_time_days_high"],
        in_format=params["preprocess_in_format"],
        variables=snakemake.config["variables"],
        dir_reforecasts_raw=snakemake.config["paths"]["dir_ReforecastsRaw"],
        path_output=snakemake.output.zarr,
    )


if __name__ == "__main__":
    main(snakemake)  # noqa: F821
