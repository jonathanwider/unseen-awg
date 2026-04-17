"""Time stepping strategies for weather generation simulations.

This module provides abstract and concrete implementations of time steppers
that control how time progresses during weather generation. Different steppers
handle various calendar systems and time representations.
"""

import datetime
from abc import ABC, abstractmethod
from typing import Any, Iterator

import cftime

from unseen_awg.timestep_utils import time_to_year_fraction_cftime


class TimeStepper(ABC):
    """Abstract base class for time stepping strategies in weather generation.

    This class defines a template for different time stepping methods,
    allowing flexible time progression for weather simulation.

    Parameters
    ----------
    blocksize : int
        Number of days to advance in each time step.

    Attributes
    ----------
    blocksize : int
        Number of days to advance in each time step.
    """

    def __init__(self, blocksize: int) -> None:
        """Initialize the time stepper with a specified block size.

        Parameters
        ----------
        blocksize : int
            Number of days to advance in each time step.
        """
        self.blocksize = blocksize

    def __iter__(self) -> Iterator[Any]:
        """Make the time stepper iterable.

        Returns
        -------
        Iterator[Any]
            The time stepper itself.
        """
        return self

    @abstractmethod
    def __next__(self) -> Any:
        """Abstract method to advance to the next time step.

        Returns
        -------
        Any
            The next time step value(s).

        Raises
        ------
        StopIteration
            When no more time steps are available.
        """
        pass


class FractionalYearStepper(TimeStepper):
    """Time stepper that works with fractional year representations.

    This stepper advances time using fractional year values.

    Parameters
    ----------
    blocksize : int
        Number of days to advance in each time step.
    initial_time : cftime.DatetimeGregorian
        Starting time for the simulation.
    reference_time : cftime.DatetimeGregorian
        Reference time used for fractional year calculations.
    tropical_year : float, optional
        Length of tropical year in days, by default 365.2422.

    Attributes
    ----------
    blocksize : int
        Number of days to advance in each time step.
    initial_time : cftime.DatetimeGregorian
        Starting time for the simulation.
    reference_time : cftime.DatetimeGregorian
        Reference time used for fractional year calculations.
    tropical_year : float
        Length of tropical year in days.
    daily_increment : float
        Fractional year increment per day.
    current_year_fraction : float
        Current position as fractional year.
    """

    def __init__(
        self,
        blocksize: int,
        initial_time: cftime.DatetimeGregorian,
        reference_time: cftime.DatetimeGregorian,
        tropical_year: float = 365.2422,
    ) -> None:
        """Initialize the fractional year stepper.

        Parameters
        ----------
        blocksize : int
            Number of days to advance in each time step.
        initial_time : cftime.DatetimeGregorian
            Starting time for the simulation.
        reference_time : cftime.DatetimeGregorian
            Reference time used for fractional year calculations.
        tropical_year : float, optional
            Length of tropical year in days, by default 365.2422.
        """
        self.blocksize = blocksize
        self.initial_time = initial_time
        self.reference_time = reference_time
        self.tropical_year = tropical_year
        self.daily_increment = 1 / tropical_year
        self.current_year_fraction = time_to_year_fraction_cftime(
            time=self.initial_time,
            ref_time=self.reference_time,
            tropical_year=self.tropical_year,
        )

    def __next__(self) -> float:
        """Advance to the next time step.

        Returns
        -------
        float
            Current fractional year before advancing.
        """
        current_fraction = self.current_year_fraction
        self.current_year_fraction += self.daily_increment * self.blocksize
        return current_fraction


class StandardStepper(TimeStepper):
    """Standard time stepper using Gregorian calendar with leap years.

    This stepper advances time using standard datetime objects and provides
    both datetime and fractional year representations.

    Parameters
    ----------
    init_year : int
        Initial year for the simulation.
    init_month : int
        Initial month for the simulation.
    init_day : int
        Initial day for the simulation.
    blocksize : int
        Number of days to advance in each time step.
    tropical_year : float, optional
        Length of tropical year in days, by default 365.2422.

    Attributes
    ----------
    blocksize : int
        Number of days to advance in each time step.
    initial_time : cftime.DatetimeGregorian
        Starting time for the simulation.
    ref_time : cftime.DatetimeGregorian
        Reference time (2000-01-01) for fractional year calculations.
    tropical_year : float
        Length of tropical year in days.
    daily_increment : float
        Fractional year increment per day.
    initial_year_fraction : float
        Initial position as fractional year.
    current_year_fraction : float
        Current position as fractional year.
    current_time : cftime.DatetimeGregorian
        Current datetime.

    Notes
    -----
    Uses cftime.DatetimeGregorian instead of standard datetime to avoid
    issues with time delta calculations in time conversion operations.
    """

    def __init__(
        self,
        init_year: int,
        init_month: int,
        init_day: int,
        blocksize: int,
        tropical_year: float = 365.2422,
    ) -> None:
        """Initialize the standard stepper.

        Parameters
        ----------
        init_year : int
            Initial year for the simulation.
        init_month : int
            Initial month for the simulation.
        init_day : int
            Initial day for the simulation.
        blocksize : int
            Number of days to advance in each time step.
        tropical_year : float, optional
            Length of tropical year in days, by default 365.2422.
        """
        self.blocksize = blocksize
        # Use cftime.DatetimeGregorian to avoid time conversion issues
        # when time deltas are added
        self.initial_time = cftime.DatetimeGregorian(
            year=init_year, month=init_month, day=init_day
        )
        self.ref_time = cftime.DatetimeGregorian(year=2000, month=1, day=1)
        self.tropical_year = tropical_year
        self.daily_increment = 1 / tropical_year

        self.initial_year_fraction = (
            (self.initial_time - self.ref_time)
            / datetime.timedelta(days=1)
            / self.tropical_year
        )
        self.current_year_fraction = self.initial_year_fraction
        self.current_time = self.initial_time

    def __next__(self) -> tuple[cftime.DatetimeGregorian, float]:
        """Advance to the next time step.

        Returns
        -------
        tuple[cftime.DatetimeGregorian, float]
            Tuple containing current datetime and current fractional year
            before advancing.
        """
        current_datetime = self.current_time
        current_fraction = self.current_year_fraction

        self.current_time += datetime.timedelta(days=self.blocksize)
        self.current_year_fraction = (
            (self.current_time - self.ref_time) / datetime.timedelta(days=1)
        ) * self.daily_increment

        return current_datetime, current_fraction


class NoLeapYearStepper(TimeStepper):
    """Time stepper using a no-leap-year calendar system.

    This stepper advances time using a calendar without leap years,
    ensuring consistent 365-day years. The fractional year calculation
    wraps around at 365 days to maintain annual periodicity.

    Parameters
    ----------
    init_year : int
        Initial year for the simulation.
    init_month : int
        Initial month for the simulation.
    init_day : int
        Initial day for the simulation.
    blocksize : int
        Number of days to advance in each time step.
    tropical_year : float, optional
        Length of tropical year in days, by default 365.2422.

    Attributes
    ----------
    blocksize : int
        Number of days to advance in each time step.
    initial_time : cftime.DatetimeNoLeap
        Starting time for the simulation.
    ref_time : cftime.DatetimeNoLeap
        Reference time (2000-01-01) for fractional year calculations.
    tropical_year : float
        Length of tropical year in days.
    daily_increment : float
        Fractional year increment per day.
    initial_year_fraction : float
        Initial position as fractional year.
    current_year_fraction : float
        Current position as fractional year.
    current_time : cftime.DatetimeNoLeap
        Current datetime.

    Notes
    -----
    The fractional year calculation uses modulo 365 to ensure proper
    wrapping for the no-leap-year calendar system.
    """

    def __init__(
        self,
        init_year: int,
        init_month: int,
        init_day: int,
        blocksize: int,
        tropical_year: float = 365.2422,
    ) -> None:
        """Initialize the no-leap-year stepper.

        Parameters
        ----------
        init_year : int
            Initial year for the simulation.
        init_month : int
            Initial month for the simulation.
        init_day : int
            Initial day for the simulation.
        blocksize : int
            Number of days to advance in each time step.
        tropical_year : float, optional
            Length of tropical year in days, by default 365.2422.
        """
        self.blocksize = blocksize
        self.initial_time = cftime.DatetimeNoLeap(
            year=init_year, month=init_month, day=init_day
        )
        self.ref_time = cftime.DatetimeNoLeap(year=2000, month=1, day=1)
        self.tropical_year = tropical_year
        self.daily_increment = 1 / tropical_year

        self.initial_year_fraction = (
            (self.initial_time - self.ref_time)
            / datetime.timedelta(days=1)
            / self.tropical_year
        )
        self.current_year_fraction = self.initial_year_fraction
        self.current_time = self.initial_time

    def __next__(self) -> tuple[cftime.DatetimeNoLeap, float]:
        """Advance to the next time step.

        Returns
        -------
        tuple[cftime.DatetimeNoLeap, float]
            Tuple containing current datetime and current fractional year
            before advancing.

        Notes
        -----
        The fractional year is calculated with modulo 365 to ensure
        proper wrapping for the no-leap-year calendar.
        """
        current_datetime = self.current_time
        current_fraction = self.current_year_fraction

        self.current_time += datetime.timedelta(days=self.blocksize)
        self.current_year_fraction = (
            ((self.current_time - self.ref_time) / datetime.timedelta(days=1)) % 365
        ) * self.daily_increment

        return current_datetime, current_fraction
