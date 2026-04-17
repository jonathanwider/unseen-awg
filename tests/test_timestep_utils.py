import cftime
import numpy as np
import xarray as xr

from unseen_awg.timestep_utils import (
    is_in_window_from_time,
    time_to_year_fraction_cftime,
    time_to_year_fraction_np_datetime64,
)


def test_time_to_year_fraction_single_cftime():
    time = cftime.datetime(2022, 1, 2, calendar="gregorian")
    ref_time = cftime.datetime(2022, 1, 1, calendar="gregorian")
    assert np.isclose(time_to_year_fraction_cftime(time, ref_time), 1 / 365.2422)


def test_time_to_year_fraction_single_numpy():
    time = np.datetime64("2022-01-02")
    ref_time = np.datetime64("2022-01-01")
    assert np.isclose(time_to_year_fraction_np_datetime64(time, ref_time), 1 / 365.2422)


def test_time_to_year_fraction_array_cftime():
    times = np.array(
        [
            cftime.datetime(2022, 1, 2, calendar="gregorian"),
            cftime.datetime(2022, 1, 3, calendar="gregorian"),
        ]
    )
    ref_time = cftime.datetime(2022, 1, 1, calendar="gregorian")
    expected = np.array([1 / 365.2422, 2 / 365.2422])
    np.testing.assert_allclose(
        time_to_year_fraction_cftime(times, ref_time).astype(float), expected
    )


def test_time_to_year_fraction_array_numpy():
    times = np.array([np.datetime64("2022-01-02"), np.datetime64("2022-01-03")])
    ref_time = np.datetime64("2022-01-01")
    expected = np.array([1 / 365.2422, 2 / 365.2422])
    np.testing.assert_allclose(
        time_to_year_fraction_np_datetime64(times, ref_time).astype(float), expected
    )


def test_time_to_year_fraction_array_xarray():
    times = xr.DataArray(
        np.array([np.datetime64("2022-01-02"), np.datetime64("2022-01-03")]),
    )
    ref_time = np.datetime64("2022-01-01")
    expected = np.array([1 / 365.2422, 2 / 365.2422])
    np.testing.assert_allclose(
        time_to_year_fraction_np_datetime64(times, ref_time).astype(float), expected
    )


def test_is_in_window_from_time():
    ref_dates = xr.DataArray(
        np.array([np.datetime64("2022-01-01"), np.datetime64("2022-01-15")]),
        dims=["time"],
    )
    other_dates = xr.DataArray(
        np.array([np.datetime64("2022-01-01"), np.datetime64("2022-01-20")]),
        dims=["time"],
    )
    ref_time = np.datetime64("2000-01-01")
    window_size = 10
    expected = np.array([True, True])
    np.testing.assert_array_equal(
        is_in_window_from_time(ref_dates, other_dates, window_size, ref_time),
        expected,
    )


def test_is_in_window_from_time_outside_window():
    ref_dates = xr.DataArray(
        np.array([np.datetime64("2022-01-01"), np.datetime64("2022-01-15")]),
        dims=["time"],
    )
    other_dates = xr.DataArray(
        np.array([np.datetime64("2022-01-01"), np.datetime64("2022-02-01")]),
        dims=["time"],
    )
    ref_time = np.datetime64("2022-01-01")
    window_size = 10
    expected = np.array([True, False])
    np.testing.assert_array_equal(
        is_in_window_from_time(ref_dates, other_dates, window_size, ref_time),
        expected,
    )


def test_is_in_window_from_time_edge_cases():
    ref_dates = xr.DataArray(
        np.array([np.datetime64("2022-01-01"), np.datetime64("2021-12-31")]),
        dims=["time"],
    )
    other_dates = xr.DataArray(
        np.array([np.datetime64("2022-01-03"), np.datetime64("2022-01-01")]),
        dims=["time"],
    )
    ref_time = np.datetime64("2022-01-01")
    window_size = 2
    expected = np.array([True, True])
    np.testing.assert_array_equal(
        is_in_window_from_time(ref_dates, other_dates, window_size, ref_time),
        expected,
    )


def test_is_in_window_from_time_edge_cases_outside():
    ref_dates = xr.DataArray(
        np.array([np.datetime64("2022-01-01"), np.datetime64("2021-12-31")]),
        dims=["time"],
    )
    other_dates = xr.DataArray(
        np.array([np.datetime64("2022-01-04"), np.datetime64("2022-01-03")]),
        dims=["time"],
    )
    ref_time = np.datetime64("2022-01-01")
    window_size = 2
    expected = np.array([False, False])
    np.testing.assert_array_equal(
        is_in_window_from_time(ref_dates, other_dates, window_size, ref_time),
        expected,
    )


def test_is_in_window_from_time_different_years():
    ref_dates = xr.DataArray(
        np.array([np.datetime64("1950-01-01"), np.datetime64("1950-12-31")]),
        dims=["time"],
    )
    other_dates = xr.DataArray(
        np.array([np.datetime64("2022-01-02"), np.datetime64("2022-01-01")]),
        dims=["time"],
    )
    ref_time = np.datetime64("2022-01-01")
    window_size = 2
    expected = np.array([True, True])
    np.testing.assert_array_equal(
        is_in_window_from_time(ref_dates, other_dates, window_size, ref_time),
        expected,
    )
