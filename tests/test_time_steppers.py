import cftime
import numpy as np

from unseen_awg.time_steppers import (
    FractionalYearStepper,
    NoLeapYearStepper,
    StandardStepper,
)


def test_fractional_year_stepper():
    stepper = FractionalYearStepper(
        blocksize=1,
        initial_time=cftime.DatetimeGregorian(2000, 1, 1),
        reference_time=cftime.DatetimeGregorian(2000, 1, 1),
    )
    assert np.isclose(next(stepper), 0.0)
    assert np.isclose(next(stepper), 1 / 365.2422)
    assert isinstance(next(stepper), float)


def test_standard_stepper():
    stepper = StandardStepper(init_year=2000, init_month=1, init_day=1, blocksize=1)
    current_time, current_year_fraction = next(stepper)
    assert current_time == cftime.DatetimeGregorian(2000, 1, 1)
    yr_fraction = np.mod(current_year_fraction, 1)
    assert np.logical_or(
        np.isclose(yr_fraction, 0.0),
        np.isclose(yr_fraction, 1.0),
    )
    current_time, current_year_fraction = next(stepper)
    assert current_time == cftime.DatetimeGregorian(2000, 1, 2)
    assert np.isclose(current_year_fraction, 1 / 365.2422)
    assert isinstance(current_time, cftime.DatetimeGregorian)
    assert isinstance(current_year_fraction, float)


def test_no_leap_year_stepper():
    stepper = NoLeapYearStepper(init_year=2000, init_month=1, init_day=1, blocksize=1)
    current_time, current_year_fraction = next(stepper)
    assert current_time == cftime.DatetimeNoLeap(2000, 1, 1)
    assert np.isclose(current_year_fraction, 0.0)
    current_time, current_year_fraction = next(stepper)
    assert current_time == cftime.DatetimeNoLeap(2000, 1, 2)
    assert np.isclose(current_year_fraction, 1 / 365.2422)

    for i in range(364):
        current_time, current_year_fraction = next(stepper)
    assert np.isclose(current_year_fraction, 0)
    assert isinstance(current_time, cftime.DatetimeNoLeap)
    assert isinstance(current_year_fraction, float)
