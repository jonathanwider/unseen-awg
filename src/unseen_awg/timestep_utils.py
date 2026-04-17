import datetime

import cftime
import numpy as np
import xarray as xr


def time_to_year_fraction_np_datetime64(
    time: xr.DataArray | xr.Dataset | np.datetime64,
    ref_time: xr.DataArray | xr.Dataset | np.datetime64,
    tropical_year: float = 365.2422,
) -> float:
    """
    Convert datetime(s) to fraction of years relative to a reference time.

    Parameters
    ----------
    time : xr.DataArray | xr.Dataset | np.datetime64
        The time to convert to fractional year. Data should be of numpy.datetime64 type.
    ref_time : xr.DataArray | xr.Dataset | np.datetime64
        The reference time for the conversion. Data should be of numpy.datetime64 type.
    tropical_year : float, optional
        The length of a tropical year in days, by default 365.2422.

    Returns
    -------
    float
        The fractional year representation of the input time.
    """
    return ((time - ref_time) / np.timedelta64(1, "D")) / tropical_year


def time_to_year_fraction_cftime(
    time: xr.DataArray | xr.Dataset | cftime.datetime,
    ref_time: xr.DataArray | xr.Dataset | cftime.datetime,
    tropical_year: float = 365.2422,
) -> float:
    """
    Convert cftime datetimes to fraction of years relative to a reference time.

    Parameters
    ----------
    time : xr.DataArray | xr.Dataset | cftime.datetime
        The time to convert to fractional year. Data should be of cftime.datetime type.
    ref_time : xr.DataArray | xr.Dataset | cftime.datetime
        The reference time for the conversion. Data should be of cftime.datetime type.
    tropical_year : float, optional
        The length of a tropical year in days, by default 365.2422.

    Returns
    -------
    float
        The fractional year representation of the input time.
    """
    return ((time - ref_time) / datetime.timedelta(days=1)) / tropical_year


def convert_to_cftime_gregorian(
    data: xr.Dataset | xr.DataArray,
) -> xr.Dataset | xr.DataArray:
    """
    Convert data to use cftime with Gregorian calendar.

    Parameters
    ----------
    data : xr.Dataset | xr.DataArray
        Input data to convert.

    Returns
    -------
    xr.Dataset | xr.DataArray
        Data with cftime datetimes and Gregorian calendar.
    """
    encoded = xr.coding.times.encode_cf_datetime(data)
    return xr.coding.times.decode_cf_datetime(
        *encoded, use_cftime=True, calendar="gregorian"
    )


def dayofyear_year_to_datetime64(dayofyear: int, year: int) -> np.datetime64:
    """
    Create numpy datetime64 from year and dayofyear.

    Parameters
    ----------
    dayofyear : int
        Day of the year (1-366).
    year : int
        Year value.

    Returns
    -------
    np.datetime64
        Corresponding datetime64 object.
    """
    return np.datetime64(f"{year}-01-01", "ns") + np.timedelta64(1, "D") * (
        dayofyear - 1
    )


def dayofyear_year_to_datetime64_naive(dayofyear: int, year: int) -> np.datetime64:
    """
    Create numpy datetime64 from year and dayofyear,
    setting NaTs when dayofyear is out of range.

    Parameters
    ----------
    dayofyear : int
        Day of the year (1-366).
    year : int
        Year.

    Returns
    -------
    np.datetime64
        Corresponding datetime64 object, or NaT if invalid.
    """
    max_dayofyear = xr.DataArray(
        np.datetime64(f"{year + 1}-01-01", "ns") - np.timedelta64(1, "D")
    ).dt.dayofyear
    if dayofyear > max_dayofyear or dayofyear < 1:
        return np.datetime64("NaT")
    else:
        return np.datetime64(f"{year}-01-01", "ns") + np.timedelta64(1, "D") * (
            dayofyear - 1
        )


def is_in_window_from_time(
    base_dates: xr.DataArray,
    other_dates: xr.DataArray,
    window_size: int,
    ref_time: np.datetime64,
    tropical_year: float = 365.2422,
) -> xr.DataArray:
    """
    Check if dates are within a temporal window from base dates.

    Parameters
    ----------
    base_dates : xr.DataArray
        Base dates for comparison.
    other_dates : xr.DataArray
        Dates to check.
    window_size : int
        Size of the window in days.
    ref_time : np.datetime64
        Reference time for year fraction calculation.
    tropical_year : float, optional
        Length of tropical year in days, by default 365.2422.

    Returns
    -------
    xr.DataArray
        Boolean array indicating which dates fall within the window.
    """
    # assume numpy datetime64 dates!
    year_fraction = time_to_year_fraction_np_datetime64(base_dates, ref_time=ref_time)
    return is_in_window_from_year_fraction(
        year_fraction,
        other_dates=other_dates,
        window_size=window_size,
        ref_time=ref_time,
        tropical_year=tropical_year,
    )


def is_in_window_from_year_fraction(
    base_year_fractions: xr.DataArray | float,
    other_dates: xr.DataArray,
    window_size: int,
    ref_time: np.datetime64,
    tropical_year: float = 365.2422,
) -> xr.DataArray:
    """
    Check if dates are within a temporal window from base "fraction of years".

    Parameters
    ----------
    base_year_fractions : xr.DataArray | float
        Base year fractions for comparison.
    other_dates : xr.DataArray
        Dates to check against the window.
    window_size : int
        Size of the window in days.
    ref_time : np.datetime64
        Reference time for year fraction calculation.
    tropical_year : float, optional
        Length of tropical year in days, by default 365.2422.

    Returns
    -------
    xr.DataArray
        Boolean DataArray indicating which dates fall within the window.
    """
    # assume numpy datetime64!
    other_year_fractions = time_to_year_fraction_np_datetime64(
        other_dates, ref_time=ref_time
    )
    delta = (window_size + 0.5) / tropical_year

    d_year_fractions = np.mod(base_year_fractions - other_year_fractions, 1)
    return np.logical_or(d_year_fractions < delta, 1 - d_year_fractions < delta)
