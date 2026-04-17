"""Utility functions for unseen-awg."""

import numpy as np
import xarray as xr
from numpy.typing import NDArray

from unseen_awg.timestep_utils import dayofyear_year_to_datetime64


def get_map_valid_n_day_transitions(da: xr.Dataset, n: int) -> xr.Dataset:
    """Map valid n-day transitions in a dataset.

    This function is used when sampling trajectories. In particular, it allows
    identifying the "next state" for each base state - and it allows identifying
    which states are actually valid samples to be included in the sampled time series
    (both the state and the next state are in data set).

    Parameters
    ----------
    da : xr.Dataset
        Input dataset containing valid_time and sample dimensions.
    n : int
        Number of days to look ahead.

    Returns
    -------
    xr.Dataset
        Dataset with next sample, year, and dayofyear information.
    """
    next_year = (da.astype(float) * np.nan).rename("next_year")
    next_dayofyear = (da.astype(float) * np.nan).rename("next_dayofyear")
    next_sample = (da.astype(float) * np.nan).rename("next_sample")

    valid_time = xr.apply_ufunc(
        np.vectorize(dayofyear_year_to_datetime64),
        da.dayofyear,
        da.year,
    ).expand_dims({"sample": da.sample.data}, axis=-1)

    # mask for not missing init times:
    m_init_time_not_nat = ~np.isnat(da)

    sel_init_time = da.data[m_init_time_not_nat.data]
    sel_valid_time = valid_time.data[m_init_time_not_nat.data]
    sel_sample = da.sample.expand_dims(
        {"dayofyear": da.dayofyear, "year": da.year}, axis=(0, 1)
    ).data[m_init_time_not_nat.data]

    next_valid_time = sel_valid_time + np.timedelta64(n, "D")

    # mask for next valid time.
    m_next_valid_time_exists = xr.DataArray(next_valid_time).isin(sel_valid_time).data
    next_valid_time = xr.DataArray(
        next_valid_time[m_next_valid_time_exists], dims="datapoint"
    )
    next_init_time = xr.DataArray(
        sel_init_time[m_next_valid_time_exists], dims="datapoint"
    )

    da_subset = da.sel(
        dayofyear=next_valid_time.dt.dayofyear, year=next_valid_time.dt.year
    )
    idcs_datapoint, idcs_sample = np.where(da_subset == next_init_time)

    nx_doy = da_subset.isel(
        datapoint=xr.DataArray(idcs_datapoint, dims="dim_0"),
        sample=xr.DataArray(idcs_sample, dims="dim_0"),
    ).dayofyear
    nx_yr = da_subset.isel(
        datapoint=xr.DataArray(idcs_datapoint, dims="dim_0"),
        sample=xr.DataArray(idcs_sample, dims="dim_0"),
    ).year
    nx_sample = da_subset.isel(
        datapoint=xr.DataArray(idcs_datapoint, dims="dim_0"),
        sample=xr.DataArray(idcs_sample, dims="dim_0"),
    ).sample

    current_doy = xr.DataArray(
        sel_valid_time[m_next_valid_time_exists][idcs_datapoint], dims="dim_0"
    ).dt.dayofyear
    current_yr = xr.DataArray(
        sel_valid_time[m_next_valid_time_exists][idcs_datapoint], dims="dim_0"
    ).dt.year
    current_sample = xr.DataArray(
        sel_sample[m_next_valid_time_exists][idcs_datapoint], dims="dim_0"
    )

    next_year.loc[
        {"year": current_yr, "dayofyear": current_doy, "sample": current_sample}
    ] = nx_yr
    next_sample.loc[
        {"year": current_yr, "dayofyear": current_doy, "sample": current_sample}
    ] = nx_sample
    next_dayofyear.loc[
        {"year": current_yr, "dayofyear": current_doy, "sample": current_sample}
    ] = nx_doy

    return xr.Dataset(
        {
            "next_sample": next_sample,
            "next_year": next_year,
            "next_dayofyear": next_dayofyear,
        }
    )


def apply_similarity_metric(
    ds_reference: xr.Dataset,
    ds_candidate: xr.Dataset,
    similarity_func,  # The actual function object, e.g., metrics.mse_similarity
    variable_name: str = "geopotential_height",
    ref_core_dims: list | None = None,
    cand_core_dims: list | None = None,
    output_core_dims: list | None = None,
    reduction_axes_for_numpy: tuple[int, ...] = (-3, -2, -1),
    dask_options: dict | None = None,
    **similarity_func_kwargs,  # Extra kwargs for the similarity_func
) -> xr.DataArray:
    """Apply a similarity metric between reference point and candidates.

    Parameters
    ----------
    ds_reference : xr.Dataset
        Reference dataset, expanded to match dimensions of candidates.
    ds_candidate : xr.Dataset
        Candidate for the candidate states.
    similarity_func
        The similarity function to apply.
    variable_name : str, optional
        Name of the variable to compare, by default "geopotential_height".
    ref_core_dims : list of str, optional
        Core dimensions (passed to xr.apply_ufunc) of the reference dataset, by default
        ["latitude", "longitude", "lag"].
    cand_core_dims : list of str, optional
        Core dimensions (passed to xr.apply_ufunc) of the candidate dataset, by default
        ["c_year", "c_sample", "c_ensemble_member", "latitude", "longitude", "lag"].
    output_core_dims : list of str, optional
        Output core dimensions, by default ["c_year", "c_sample", "c_ensemble_member"].
    reduction_axes_for_numpy : tuple of int, optional
        Axes to reduce when applying numpy function, by default (-3, -2, -1).
    dask_options : dict, optional
        Dask options for apply_ufunc, by default None.
    **similarity_func_kwargs
        Additional keyword arguments for the similarity function.

    Returns
    -------
    xr.DataArray
        Array of computed similarities.
    """
    if ref_core_dims is None:
        ref_core_dims = ["latitude", "longitude", "lag"]
    if cand_core_dims is None:
        cand_core_dims = [
            "c_year",
            "c_sample",
            "c_ensemble_member",
            "latitude",
            "longitude",
            "lag",
        ]
    if output_core_dims is None:
        output_core_dims = [
            ["c_year", "c_sample", "c_ensemble_member"]
        ]  # apply_ufunc expects list of lists
    else:
        output_core_dims = [output_core_dims]  # Ensure it's a list of lists

    if dask_options is None:
        dask_options = {"dask": "parallelized", "output_dtypes": [float]}

    # Determine dimensions to exclude from broadcasting/alignment checks
    # These are the dimensions that are *reduced* by the numpy function
    similarity_dims = set(ref_core_dims)  # Assuming these are the ones reduced

    da_reference = ds_reference[variable_name]
    da_candidate = ds_candidate[variable_name]

    # Prepare kwargs for the similarity function
    # The numpy function will receive these through apply_ufunc's kwargs argument
    numpy_func_kwargs = {"reduction_axes": reduction_axes_for_numpy}
    numpy_func_kwargs.update(similarity_func_kwargs)

    da_similarities = xr.apply_ufunc(
        similarity_func,
        da_reference,
        da_candidate,
        input_core_dims=[ref_core_dims, cand_core_dims],
        output_core_dims=output_core_dims,
        exclude_dims=similarity_dims,
        vectorize=True,  # Important if non-core dims don't match perfectly
        kwargs=numpy_func_kwargs,  # Pass reduction_axes and other kwargs here
        **dask_options,
    )
    return da_similarities


def get_k_smallest_indices(arr: NDArray, mask: NDArray, k: int) -> NDArray:
    """Get indices of k smallest values in an array after applying masking.

    Parameters
    ----------
    arr : NDArray
        Array of values.
    mask : NDArray
        Boolean mask indicating valid elements.
    k : int
        Number of smallest indices to return.

    Returns
    -------
    NDArray
        Indices of k smallest values.
    """
    masked_arr = np.where(np.logical_or(np.isnan(arr), ~mask), np.inf, arr)
    partitioned_indices = np.argpartition(masked_arr, k + 1)
    smallest_indices = partitioned_indices[: k + 1]
    return smallest_indices[np.argsort(masked_arr[smallest_indices])[1:]]


def get_k_random_indices(
    arr: NDArray, mask: NDArray, k: int, rng: np.random.Generator
) -> NDArray:
    """Get k random indices in an array after applying masking.

    Parameters
    ----------
    arr : NDArray
        Array of values.
    mask : NDArray
        Boolean mask indicating valid elements.
    k : int
        Number of random indices to return.
    rng : np.random.Generator
        Random number generator.

    Returns
    -------
    NDArray
        Array of k random indices.
    """
    indices_valid = np.arange(len(arr))[np.logical_and(~np.isnan(arr), mask)]
    return rng.choice(indices_valid, size=k, replace=False)


def is_no_jump(traj: xr.Dataset) -> NDArray:
    """Check which steps in a trajectory are "jumps".

    For each pair of consecutive states in the trajectory, test whether the trajectory
    has a "jump", i.e. the samples aren't actually consecutive in the original dataset.
    For a reforecast dataset, not having a jump means having the same init_time and
    ensemble_member, while the second state in each pair has a lead_time 1 day larger
    than the first state.

    Parameters
    ----------
    traj : xr.Dataset
        Sampled trajectory data.

    Returns
    -------
    NDArray
        Boolean DataArray indicating where there are no jumps.
    """
    return np.logical_and(
        np.logical_and(
            traj.isel(out_time=slice(1, None)).ensemble_member.data
            == traj.isel(out_time=slice(None, -1)).ensemble_member.data,
            traj.isel(out_time=slice(1, None)).init_time.data
            == traj.isel(out_time=slice(None, -1)).init_time.data,
        ),
        traj.isel(out_time=slice(1, None)).lead_time.data
        == traj.isel(out_time=slice(None, -1)).lead_time.data + np.timedelta64(1, "D"),
    )


def grids_are_identical_subset(
    source_ds: xr.Dataset | xr.DataArray,
    target_ds: xr.Dataset | xr.DataArray,
    coord_names=["latitude", "longitude"],
):
    """Check if target is a simple subset with IDENTICAL spacing"""
    for coord in coord_names:
        source_vals = source_ds[coord].values
        target_vals = target_ds[coord].values

        # Check all target values exist in source
        if not all(np.isin(target_vals, source_vals)):
            return False

        # Check spacing is identical
        source_spacing = np.diff(source_vals)
        target_spacing = np.diff(target_vals)

        # Compare typical spacing (accounting for floating point)
        source_typical = np.median(source_spacing)
        target_typical = np.median(target_spacing)

        if not np.isclose(source_typical, target_typical, rtol=1e-5):
            return False

    return True
