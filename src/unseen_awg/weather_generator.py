"""Weather generator module for analog-based weather simulation.

This module provides the core WeatherGenerator class that implements analog-based
weather generation using similarity measures and probability models to sample
realistic weather trajectories from historical data.
"""

import datetime
import os
import shutil
import uuid
from typing import Any

import cartopy.crs as ccrs
import dask
import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import yaml
from icecream import ic
from matplotlib.gridspec import GridSpec
from tqdm.auto import tqdm

import unseen_awg.similarity_measures
from unseen_awg.data_classes import InitTimeLeadTimeMemberState
from unseen_awg.plotting_utils import add_contours, map_plot_without_frame_with_bounds
from unseen_awg.probability_models import ProbabilityModel
from unseen_awg.snakemake_utils import snakemake_handler
from unseen_awg.time_steppers import TimeStepper
from unseen_awg.timestep_utils import (
    dayofyear_year_to_datetime64_naive,
    is_in_window_from_time,
    is_in_window_from_year_fraction,
)
from unseen_awg.utils import (
    apply_similarity_metric,
    get_k_random_indices,
    get_k_smallest_indices,
    get_map_valid_n_day_transitions,
)


def setup_lazy_similarity_dataset(
    ds_year_dayofyear_format: xr.Dataset,
    window_size: int,
    ref_time: np.datetime64 = np.datetime64("2000-01-01", "ns"),
) -> xr.Dataset:
    """Set up a lazy dataset to store similarities computed in weather generator in.

    Creates a dataset structure for computing similarities between weather states
    within a specified time window. The dataset includes coordinates for reference
    states and candidate states with time shifts.

    The dataset has dimensions that identify the base sample
    (dayofyear, year, sample, ensemble member) and additional dimensions that identify
    the candidate (d_shift, c_year, c_sample, c_ensemble_member). The valid_time of
    the candidate state can be computed from c_year, dayofyear, and d_shift.

    Parameters
    ----------
    ds_year_dayofyear_format : xr.Dataset
        Input dataset in year-dayofyear format containing weather data.
    window_size : int
        Size of the time window (in days) for similarity computations.
    ref_time : np.datetime64, optional
        Reference time for temporal calculations, by default
        np.datetime64("2000-01-01", "ns").

    Returns
    -------
    xr.Dataset
        Lazy dataset with similarity computation structure including coordinates
        for reference and candidate states.
    """
    ds_similarities = xr.Dataset(
        coords={
            "dayofyear": ds_year_dayofyear_format.dayofyear.data,
            "year": ds_year_dayofyear_format.year.data,
            "sample": ds_year_dayofyear_format.sample.data,
            "ensemble_member": ds_year_dayofyear_format.ensemble_member.data,
            "d_shift": np.arange(-(window_size + 1), (window_size + 1) + 1),
            "c_year": ds_year_dayofyear_format.year.data,
            "c_sample": ds_year_dayofyear_format.sample.data,
            "c_ensemble_member": ds_year_dayofyear_format.ensemble_member.data,
        }
    )

    l_dims = [len(ds_similarities[d]) for d in ds_similarities.dims]
    l_chunks = []
    for d in ds_similarities.dims:
        if d in ["dayofyear", "year"]:
            l_chunks.append(1)
        else:
            l_chunks.append(len(ds_similarities[d]))
    ds_similarities["similarities"] = (
        ds_similarities.dims,
        dask.array.full(l_dims, fill_value=np.nan, chunks=l_chunks),
    )
    ds_similarities = ds_similarities.assign_coords(
        {
            "init_time": ds_year_dayofyear_format.init_time.load(),
            "c_init_time": ds_year_dayofyear_format.init_time.load()
            .rename({"sample": "c_sample", "year": "c_year"})
            .sel(
                dayofyear=(
                    ((ds_similarities.dayofyear + ds_similarities.d_shift) - 1) % 366
                )
                + 1
            ),
            "c_dayofyear": (
                ((ds_similarities.dayofyear + ds_similarities.d_shift) - 1) % 366
            )
            + 1,
        }
    )
    valid_time_reference = xr.apply_ufunc(
        np.vectorize(dayofyear_year_to_datetime64_naive),
        ds_similarities.dayofyear,
        ds_similarities.year,
    )
    valid_time_candidates = xr.apply_ufunc(
        np.vectorize(dayofyear_year_to_datetime64_naive),
        (((ds_similarities.dayofyear + ds_similarities.d_shift) - 1) % 366) + 1,
        ds_similarities.c_year,
    )

    ds_similarities = ds_similarities.assign_coords(
        m_is_near=is_in_window_from_time(
            valid_time_reference,
            valid_time_candidates,
            window_size=window_size,
            ref_time=ref_time,
        )
    )
    ds_similarities = ds_similarities.assign_coords(
        {
            "valid_time": xr.apply_ufunc(
                np.vectorize(dayofyear_year_to_datetime64_naive),
                ds_similarities.dayofyear,
                ds_similarities.year,
            ),
            "c_valid_time": xr.apply_ufunc(
                np.vectorize(dayofyear_year_to_datetime64_naive),
                (((ds_similarities.dayofyear + ds_similarities.d_shift) - 1) % 366) + 1,
                ds_similarities.c_year,
            ),
        }
    )
    ds_similarities = ds_similarities.assign_coords(
        {
            "lead_time": ds_similarities.valid_time - ds_similarities.init_time,
            "c_lead_time": ds_similarities.c_valid_time - ds_similarities.c_init_time,
        }
    )

    # cf-conventions:
    ds_similarities.attrs["Conventions"] = "CF-1.7"
    ds_similarities.attrs["Title"] = (
        "Similarities between pairs of states in the underlying dataset"
    )
    ds_similarities.attrs["Source"] = (
        "Computed according to the selected similarity measure from underlying dataset."
    )
    ds_similarities["dayofyear"].attrs.update(
        long_name="day of year",
        units="1",  # dimensionless integer index
    )
    ds_similarities["year"].attrs.update(
        long_name="year",
        units="1",
    )
    ds_similarities["sample"].attrs.update(
        long_name="index for data on same valid_time.",
        units="1",
    )
    ds_similarities["ensemble_member"].attrs.update(
        long_name="reforecast ensemble member",
        units="1",
    )
    ds_similarities["d_shift"].attrs.update(
        long_name="shift between day of year of base sample "
        "and day of year of candidate",
        units="1",
    )
    ds_similarities["c_year"].attrs.update(
        long_name="year of candidate",
        units="1",
    )
    ds_similarities["c_sample"].attrs.update(
        long_name="index for data on same valid_time of candidate",
        units="1",
    )
    ds_similarities["c_ensemble_member"].attrs.update(
        long_name="reforecas ensemble member of candidate",
        units="1",
    )
    ds_similarities["valid_time"].attrs.update(
        long_name="valid time of forecast",
        standard_name="time",
    )
    ds_similarities["lead_time"].attrs.update(
        standard_name="forecast_period",
        long_name="reforecast lead time",
    )
    ds_similarities["init_time"].attrs.update(
        long_name="reforecast initialisation time",
        standard_name="forecast_reference_time",
    )
    ds_similarities["c_valid_time"].attrs.update(
        long_name="valid time of candidate",
        standard_name="time",
    )
    ds_similarities["c_init_time"].attrs.update(
        long_name="reforecast initialisation time of candidate",
        standard_name="forecast_reference_time",
    )
    ds_similarities["c_lead_time"].attrs.update(
        long_name="reforecast lead time of candidate",
    )
    ds_similarities["c_dayofyear"].attrs.update(
        long_name="day of year of candidate",
        units="1",  # dimensionless integer index
    )
    ds_similarities["m_is_near"].attrs.update(
        long_name="mask: candidate date is near target date",
        units="1",
    )
    ds_similarities["similarities"].attrs.update(
        long_name="similarity score between forecast and candidate analog",
        units="1",
    )
    for var in ["init_time", "c_init_time", "valid_time", "c_valid_time"]:
        ds_similarities[var].encoding.update(
            units="seconds since 1970-01-01",
            dtype="int64",
        )

    # timedelta variables: encode as seconds (a plain number + units)
    for var in ["lead_time", "c_lead_time"]:
        ds_similarities[var].encoding.update(
            units="seconds",
            dtype="int64",
        )
    return ds_similarities


class WeatherGenerator:
    """Analog-based weather generator for creating synthetic weather trajectories.

    This class implements an analog-based approach to weather generation, where
    weather states are sampled from historical data while assuring that successive
    states either follow each other in the historical dataset or are analogs of the true
    successor. The generator uses configurable similarity measures and probability
    models to create realistic weather sequences, additional parameters can be specified
    when sampling time series.

    Parameters
    ----------
    params : dict[str, Any]
        Configuration parameters containing:

        - weather_generator.window_size : int
              Half-window size within which states are considered as potential analogs.
        - weather_generator.var : str
              Name of variable to use for similarity calculations.
        - weather_generator.similarity : str
              Name of similarity function to use.
        - weather_generator.use_precomputed_similarities : bool
              Whether to use precomputed similarities or not. If True, similarities
              are precomputed when a WeatherGenerator instance is initialized.
              Otherwise, a lazy array is set up and similarities are computed on-the-fly
              during the sampling process. This slows down the sampling process.
        - weather_generator.n_samples : int
              Number of samples to use from dataset. By "sample" we denote
              datapoints that possess the same valid_time (but a different init_time).
              Providing a low n_samples allows restricting the number of included
              states.
        - dir_wg : str
              Directory path for weather generator outputs.
        - zarr_year_dayofyear : str
              Path to zarr store containing the preprocessed input dataset.

    Attributes
    ----------
    window_size : int
        Half-window size within which states are considered as potential analogs.
    var : str
        Name of variable to use for similarity calculations.
    similarity_function : callable
        Function used to compute similarities between states.
    path_wg : str
        Path to weather generator working directory.
    path_dataset : str
        Path to input dataset.
    use_precomputed_similarities : bool
        Flag indicating whether to use precomputed similarities.
    ds_similarities : xr.Dataset
        Dataset containing results of similarity computations. Is initialized as
        lazy dataset and computed during initialization if
        use_precomputed_similarities is True.
    """

    def __init__(self, params: dict[str, Any]) -> None:
        self.window_size = params["weather_generator.window_size"]
        self.var = params["weather_generator.var"]
        self.similarity_function = getattr(
            unseen_awg.similarity_measures, params["weather_generator.similarity"]
        )
        self.path_wg = os.path.join(params["dir_wg"])

        self.path_dataset = params["zarr_year_dayofyear"]
        self.use_precomputed_similarities = params[
            "weather_generator.use_precomputed_similarities"
        ]

        # load the reshaped year dayofyear dataset (chunks along dayofyear).
        ic("Load circulation dataset")
        ds = xr.open_zarr(self.path_dataset, decode_timedelta=True).isel(
            sample=slice(0, params["weather_generator.n_samples"])
        )

        # set up store for similarities
        path_similarities = os.path.join(self.path_wg, "similarities.zarr")

        ic("Set up array for similarities")
        self.ds_similarities = setup_lazy_similarity_dataset(
            ds_year_dayofyear_format=ds,
            window_size=self.window_size,
        )
        d_shifts = xr.DataArray(
            np.arange(-(self.window_size + 1), (self.window_size + 1) + 1),
            dims="d_shift",
        )

        ds = ds.drop_vars("init_time")
        ds_candidate = (
            ds.rename(
                {
                    "year": "c_year",
                    "ensemble_member": "c_ensemble_member",
                    "sample": "c_sample",
                }
            )
            .sel(dayofyear=((((ds.dayofyear + d_shifts) - 1) % 366) + 1))
            .assign_coords({"d_shifts": d_shifts})
        )

        ic("Set up similarity computation lazily")
        similarities = apply_similarity_metric(
            ds_reference=ds,
            ds_candidate=ds_candidate,
            similarity_func=self.similarity_function,
            variable_name=self.var,
        ).rename("similarities")

        if self.use_precomputed_similarities:
            self.ds_similarities.to_zarr(
                path_similarities,
                compute=False,
            )
            for i_doy, _ in enumerate(
                tqdm(
                    self.ds_similarities.similarities.dayofyear,
                    desc="Compute similarities",
                )
            ):
                similarities.isel(
                    dayofyear=slice(i_doy, i_doy + 1),
                ).drop_vars(
                    [
                        "ensemble_member",
                        "c_sample",
                        "c_ensemble_member",
                        "c_year",
                        "d_shifts",
                        "sample",
                        "year",
                    ]
                ).load().to_zarr(
                    path_similarities,
                    region={
                        "dayofyear": slice(i_doy, i_doy + 1),
                    },
                )
        else:
            os.makedirs(self.path_wg, exist_ok=False)
            self.ds_similarities["similarities"] = similarities

    def sample_trajectory(
        self,
        blocksize: int,
        probability_model: ProbabilityModel,
        stepper_class: type[TimeStepper],
        n_steps: int,
        rng: np.random.Generator,
        initialization: InitTimeLeadTimeMemberState | None = None,
        start_by_taking_analog: bool = False,
        show_progressbar: bool = False,
    ) -> xr.Dataset:
        """Sample a synthetic weather trajectory using the analog method.

        Generates a weather trajectory by iteratively sampling analog states
        from historical data. The sampling alternates between following a historical
        trajectory and sampling analogs of the true successor states - so that in effect
        blocks of size blocksize are sampled while for the transition between blocks
        close analogs of the "true" state that would follow each block are chosen.

        Parameters
        ----------
        blocksize : int
            Number of days to sample contiguously from the same historical trajectory.
        probability_model : ProbabilityModel
            Model defining sampling probabilities given similarities between base states
            and candidate states.
        stepper_class : type[TimeStepper]
            Class for managing the output time assigned to each sample in the resulting
            trajectory. This is used as a means for supporting different calendars in
            sampled datasets.
        n_steps : int
            Number of sampling steps to perform, not necessarily equal to the length of
            the sampled series in days.
        rng : np.random.Generator
            Random number generator for sampling.
        initialization : InitTimeLeadTimeMemberState | None, optional
            Initial state specification, by default None (random initialization).
        start_by_taking_analog : bool, optional
            Whether to start by taking an analog of the initial state, by default False.
        show_progressbar : bool, optional
            Whether to display progress bar, by default False.

        Returns
        -------
        xr.Dataset
            Generated weather trajectory with time series of sampled states.
        """
        # set up a "map" between states and their successors states
        # for blocksize-long transitions:
        path_map = os.path.join(self.path_wg, f"map_{blocksize}_steps_transition.nc")
        map_n_step_transition = self._load_or_create_map_file(path_map, blocksize)

        current_block_start_state = self.get_initial_state(
            initialization=initialization,
            map_n_step_transition=map_n_step_transition,
            blocksize=blocksize,
            rng=rng,
        )
        # initialize stepper with starting condition.
        # Stepper is iterator that on each call returns tuple of (time, year_fraction)
        stepper = stepper_class(
            init_year=current_block_start_state.valid_time.dt.year,
            init_month=current_block_start_state.valid_time.dt.month,
            init_day=current_block_start_state.valid_time.dt.day,
            blocksize=blocksize,
        )
        current_out_time, current_year_fraction = next(stepper)
        current_block_start_state["out_time"] = current_out_time
        if start_by_taking_analog:
            current_block_start_state = self.sampling_step(
                next_state=current_block_start_state,
                next_year_fraction=current_year_fraction,
                map_n_step_transition=map_n_step_transition,
                probability_model=probability_model,
                rng=rng,
            )

        # initialize empty trajectory
        trajectory: list[xr.Dataset] = []

        for _ in tqdm(
            range(n_steps), disable=(not show_progressbar), desc="Sampling trajectory"
        ):
            # alternate between following what actually happend and analog sampling
            next_state, next_year_fraction = self.time_evolution_step(
                trajectory=trajectory,
                current_block_start_state=current_block_start_state,
                map_n_step_transition=map_n_step_transition,
                stepper=stepper,
                blocksize=blocksize,
            )
            next_state = self.sampling_step(
                next_state=next_state,
                next_year_fraction=next_year_fraction,
                map_n_step_transition=map_n_step_transition,
                probability_model=probability_model,
                rng=rng,
            )
            current_block_start_state = next_state
        return xr.concat(trajectory, dim="out_time")

    @classmethod
    def load(cls, wg_path: str) -> "WeatherGenerator":
        """Load a WeatherGenerator instance from saved configuration.

        Parameters
        ----------
        wg_path : str
            Path to directory containing saved weather generator configuration.

        Returns
        -------
        WeatherGenerator
            Loaded weather generator instance.
        """
        with open(os.path.join(wg_path, "params.yaml"), "r") as file:
            params = yaml.safe_load(file)
        instance = super().__new__(cls)
        instance.window_size = params["weather_generator.window_size"]
        instance.var = params["weather_generator.var"]
        instance.similarity_function = getattr(
            unseen_awg.similarity_measures, params["weather_generator.similarity"]
        )

        instance.path_wg = params["dir_wg"]
        instance.path_dataset = params["zarr_year_dayofyear"]
        instance.use_precomputed_similarities = params[
            "weather_generator.use_precomputed_similarities"
        ]
        if instance.use_precomputed_similarities:
            instance.ds_similarities = xr.open_zarr(
                os.path.join(wg_path, "similarities.zarr"), decode_timedelta=True
            )
        else:
            ds = xr.open_zarr(instance.path_dataset, decode_timedelta=True).isel(
                sample=slice(0, params["weather_generator.n_samples"])
            )

            instance.ds_similarities = setup_lazy_similarity_dataset(
                ds_year_dayofyear_format=ds,
                window_size=instance.window_size,
            )
            d_shifts = xr.DataArray(
                np.arange(-(instance.window_size + 1), (instance.window_size + 1) + 1),
                dims="d_shift",
            )

            ds = ds.drop_vars("init_time")
            ds_candidate = (
                ds.rename(
                    {
                        "year": "c_year",
                        "ensemble_member": "c_ensemble_member",
                        "sample": "c_sample",
                    }
                )
                .sel(dayofyear=((((ds.dayofyear + d_shifts) - 1) % 366) + 1))
                .assign_coords({"d_shifts": d_shifts})
            )
            similarities = apply_similarity_metric(
                ds_reference=ds,
                ds_candidate=ds_candidate,
                similarity_func=instance.similarity_function,
                variable_name=instance.var,
            ).rename("similarities")
            instance.ds_similarities["similarities"] = similarities

        return instance

    def time_evolution_step(
        self,
        trajectory: list[xr.Dataset],
        current_block_start_state: xr.Dataset,
        map_n_step_transition: xr.Dataset,
        stepper: TimeStepper,
        blocksize: int,
    ) -> tuple[xr.Dataset, float]:
        """Perform one time evolution step in trajectory generation.

        Advances the trajectory by one block of time steps, following the
        evolution in the underlying historical data set starting from current_state.

        Parameters
        ----------
        trajectory : list[xr.Dataset]
            List of trajectory states to append new states to.
        current_block_start_state : xr.Dataset
            State to start current state from.
        map_n_step_transition : xr.Dataset
            Mapping of allowed n-day transitions between states.
        stepper : TimeStepper
            Time stepper instance for managing temporal progression.
        blocksize : int
            Number of days in each time block.

        Returns
        -------
        tuple[xr.Dataset, float]
            Next state and corresponding year fraction.
        """
        for i in range(0, blocksize):
            trajectory.append(
                xr.Dataset(
                    {
                        "lead_time": current_block_start_state.lead_time
                        + i * np.timedelta64(1, "D"),
                        "init_time": current_block_start_state.init_time,
                        "ensemble_member": current_block_start_state.ensemble_member,
                    },
                    coords={
                        "out_time": current_block_start_state.out_time
                        + datetime.timedelta(days=i)
                    },
                )
            )
        next_coords = map_n_step_transition.sel(
            dayofyear=current_block_start_state.dayofyear,
            year=current_block_start_state.year,
            sample=current_block_start_state.sample,
        )
        next_state = xr.Dataset(
            self.ds_similarities.init_time.sel(
                dayofyear=next_coords.next_dayofyear,
                year=next_coords.next_year,
                sample=next_coords.next_sample,
            ).coords
        ).reset_coords(drop=False)
        next_out_time, next_year_fraction = next(stepper)
        next_state["out_time"] = next_out_time
        next_state["ensemble_member"] = current_block_start_state.ensemble_member
        return next_state, next_year_fraction

    def sampling_step(
        self,
        next_state: xr.Dataset,
        next_year_fraction: float,
        map_n_step_transition: xr.Dataset,
        probability_model: ProbabilityModel,
        rng: np.random.Generator,
    ) -> xr.Dataset:
        """Perform analog sampling step to select next weather state.

        Samples an analog state from historical data based on similarity
        to the true next state and according to distribution and constraints
        defined by the probability_model.

        Parameters
        ----------
        next_state : xr.Dataset
            True next state in underlying historic dataset.
        next_year_fraction : float
            Year fraction of next sample. Used to define temporal similarity
            rather than an actual calender date to simplify calendar handling.
        map_n_step_transition : xr.Dataset
            Mapping of allowed n-day transitions between states.
        probability_model : ProbabilityModel
            Model defining sampling probabilities given similarities.
        rng : np.random.Generator
            Random number generator for sampling.

        Returns
        -------
        xr.Dataset
            Sampled analog state for the next time step.
        """
        s_sims = self.ds_similarities.sel(
            year=next_state.year,
            dayofyear=next_state.dayofyear,
            sample=next_state.sample,
            ensemble_member=next_state.ensemble_member,
        ).load()

        for var in [
            "c_year",
            "c_valid_time",
            "c_sample",
            "c_init_time",
            "c_dayofyear",
            "c_lead_time",
            "c_ensemble_member",
        ]:
            s_sims[f"sampled_{var}"] = s_sims[var].broadcast_like(s_sims.similarities)

        # Mask that indicates whether a sample is a valid sample
        # (i.e. the corresponding date is actually contained in the data set).
        m_is_valid = ~np.isnan(
            map_n_step_transition.sel(
                sample=s_sims.c_sample,
                year=s_sims.c_year,
                dayofyear=((s_sims.dayofyear + s_sims.d_shift) - 1) % 366 + 1,
            ).next_year
        )

        # Mask that is true if the states are close to the next assigned output date.
        m_is_near_to_year_fraction = is_in_window_from_year_fraction(
            base_year_fractions=next_year_fraction,
            other_dates=s_sims.c_valid_time.load(),
            window_size=self.window_size,
            ref_time=np.datetime64("2000-01-01", "ns"),
        )

        # Combine masks with additional mask that is true if the states
        # are close to valid_date of the next sample.
        m = (
            m_is_valid
            & m_is_near_to_year_fraction
            & s_sims.m_is_near.expand_dims({"c_sample": s_sims.c_sample})
            .expand_dims(
                {"ensemble_member": self.ds_similarities.ensemble_member}, axis=-1
            )
            .load()
        )

        # take the mask subset:
        similarities = s_sims.similarities.data[m]
        coords = xr.Dataset(
            {
                c: ("datapoint", v.data[m])
                for c, v in s_sims.data_vars.items()
                if c
                in [
                    "sampled_c_valid_time",
                    "sampled_c_init_time",
                    "sampled_c_ensemble_member",
                ]
            }
        ).rename(
            {
                "sampled_c_valid_time": "valid_time",
                "sampled_c_init_time": "init_time",
                "sampled_c_ensemble_member": "ensemble_member",
            }
        )

        i = probability_model.sample(
            similarities=similarities,
            coords_s_next=next_state[["init_time", "valid_time", "ensemble_member"]],
            coords_candidates=coords,
            rng=rng,
            size=1,
        )

        res = xr.Dataset(
            {
                var: s_sims[f"sampled_c_{var}"].data[m][i][0]
                for var in [
                    "year",
                    "valid_time",
                    "sample",
                    "init_time",
                    "dayofyear",
                    "lead_time",
                    "ensemble_member",
                ]
            }
        )
        res["out_time"] = next_state["out_time"]
        return res

    def _load_or_create_map_file(self, path_map: str, blocksize: int) -> xr.Dataset:
        """Load existing transition map or create new one if not found.

        Attempts to load a precomputed transition map file. If the file doesn't
        exist or cannot be opened, creates a new transition map and saves it
        atomically to prevent race conditions in parallel execution.
        The map file is used to identify valid samples and provides a mapping between
        coordinates of each state and the coordinates of its corresponding true
        successor state.

        Parameters
        ----------
        path_map : str
            Path to the transition map file.
        blocksize : int
            Number of days for each block of states.

        Returns
        -------
        xr.Dataset
            Transition map dataset containing valid n-day transitions.

        Raises
        ------
        ValueError
            If no valid transitions exist for the chosen blocksize.
        """
        # Try to load the existing file first
        try:
            return xr.open_dataset(
                path_map, decode_timedelta=True, lock=False, mode="r"
            )
        except (FileNotFoundError, OSError):
            # File doesn't exist or can't be opened - we'll need to create it
            # Create a unique temporary filename
            temp_path = os.path.join(
                os.path.dirname(path_map),
                f"temp_map_{blocksize}_{uuid.uuid4().hex}.nc",
            )

            try:
                # Create the map
                map_n_step_transition = get_map_valid_n_day_transitions(
                    self.ds_similarities.init_time.load(), n=blocksize
                )

                if np.isnan(map_n_step_transition["next_sample"]).all():
                    raise ValueError(
                        f"No valid transitions for chosen blocksize: {blocksize}"
                    )
                # Save to temporary file first
                map_n_step_transition.to_netcdf(temp_path)

                # Try to atomically move the temp file to the final location
                # This will fail if another process has created the file in the meantime
                try:
                    # Make sure directory exists
                    os.makedirs(os.path.dirname(path_map), exist_ok=True)

                    # Try to move the file (atomic on same filesystem)
                    shutil.move(temp_path, path_map)
                    return map_n_step_transition
                except (OSError, shutil.Error):
                    # If move fails, another process likely created the file first
                    # Try to load the existing file
                    if os.path.exists(path_map):
                        return xr.open_dataset(
                            path_map, decode_timedelta=True, lock=False, mode="r"
                        )
                    else:
                        # If the file still doesn't exist, return our computed map
                        return map_n_step_transition
            finally:
                # Clean up the temp file if it still exists
                if os.path.exists(temp_path):
                    try:
                        os.remove(temp_path)
                    except OSError:
                        pass

    def get_initial_state(
        self,
        initialization: InitTimeLeadTimeMemberState | None,
        map_n_step_transition: xr.Dataset,
        blocksize: int,
        rng: np.random.Generator,
    ) -> xr.Dataset:
        """Get initial state to start sampling a trajectory from.

        Determines the starting state for weather generation, either from
        a specified initialization or by random selection from set of valid states.

        Parameters
        ----------
        initialization : InitTimeLeadTimeMemberState | None
            Specific initialization state, or None for random selection.
        map_n_step_transition : xr.Dataset
            Mapping of allowed n-day transitions between states.
        blocksize : int
            Number of days in each time block.
        rng : np.random.Generator
            Random number generator for random initialization.

        Returns
        -------
        xr.Dataset
            Initial state for trajectory generation.

        Raises
        ------
        ValueError
            If the specified initialization is invalid.
        AssertionError
            If the initialization state is not found in valid transitions.
        """
        if initialization is None:
            # if no initialization provided select randomly from possible samples.
            stacked_isnotnan_ds = ~np.isnan(map_n_step_transition.next_sample).stack(
                datapoint=("dayofyear", "year", "sample")
            )
            initial_state = xr.Dataset(
                stacked_isnotnan_ds[stacked_isnotnan_ds]
                .isel(
                    datapoint=rng.integers(
                        len(stacked_isnotnan_ds[stacked_isnotnan_ds])
                    )
                )
                .drop("datapoint")
                .coords
            )
            return initial_state.assign_coords(
                ensemble_member=rng.choice(self.ds_similarities.ensemble_member)
            ).reset_coords(drop=False)
        elif isinstance(initialization, InitTimeLeadTimeMemberState):
            stacked_isnotnan_ds = ~np.isnan(
                get_map_valid_n_day_transitions(
                    self.ds_similarities.init_time.load(), n=blocksize
                ).next_sample
            ).stack(datapoint=("dayofyear", "year", "sample"))
            vsa = stacked_isnotnan_ds[stacked_isnotnan_ds]
            assert len(
                vsa.where(
                    (vsa.init_time == initialization.init_time)
                    & (vsa.lead_time == initialization.lead_time),
                    drop=True,
                ).datapoint
                == 1
            ), f"{initialization} seems to be an invalid starting point."
            initial_state = xr.Dataset(
                vsa.where(
                    (vsa.init_time == initialization.init_time)
                    & (vsa.lead_time == initialization.lead_time),
                    drop=True,
                )
                .squeeze()
                .coords
            ).drop_vars("datapoint")
            return initial_state.assign_coords(
                ensemble_member=initialization.ensemble_member
            ).reset_coords(drop=False)
        else:
            raise ValueError(f"Invalid initial condition {initialization}")

    def get_similarities_k_closest_neighbors(
        self,
        states: xr.DataArray,
        k: int,
        minimum_timedelta_days: int | None = None,
        dim_states: str | None = None,
    ) -> xr.Dataset:
        """Get the k closest neighbors based on similarity measures.

        Finds the k most similar historical states to the given query states
        based on (precomputed) similarity measures.

        Parameters
        ----------
        states : xr.DataArray
            Query states to find neighbors for.
        k : int
            Number of closest neighbors to return.
        minimum_timedelta_days : int | None, optional
            Minimum time separation in days between query and candidate states, that
            allows excluding analogs that are temporally close to the base state if
            this is undesired. By default None, i.e. no restriction.
        dim_states : str | None, optional
            Dimension name for states, by default None.

        Returns
        -------
        xr.Dataset
            Dataset containing the k closest neighbor states and their similarities.
        """
        sims = self.ds_similarities.sel(
            dayofyear=states.dayofyear,
            year=states.year,
            sample=states.sample,
            ensemble_member=states.ensemble_member,
        ).load()
        sims_flattened = -sims.similarities.stack(
            flat_dim=[d for d in sims.dims if d != dim_states]
        )  # assume that similarity increases the more similar the points are

        if minimum_timedelta_days is not None:
            keeps_minimum_distance = (
                abs(
                    (sims_flattened.valid_time - sims_flattened.c_valid_time)
                    / np.timedelta64(1, "D")
                )
                >= minimum_timedelta_days
            )
        else:
            keeps_minimum_distance = xr.ones_like(sims_flattened, dtype=bool)

        return -sims_flattened.isel(
            flat_dim=xr.apply_ufunc(
                get_k_smallest_indices,
                sims_flattened,
                keeps_minimum_distance,
                k,
                input_core_dims=[["flat_dim"], ["flat_dim"], []],
                output_core_dims=[["neighbor"]],
                vectorize=True,
            )
        )

    def get_similarities_k_random_neighbors(
        self,
        states: xr.DataArray,
        k: int,
        rng: np.random.Generator,
        minimum_timedelta_days: int | None = None,
        dim_states: str | None = None,
    ) -> xr.Dataset:
        """Get k randomly selected neighbors from valid candidates.

        Randomly selects k historical states from valid candidates that meet
        the specified temporal constraints.

        Parameters
        ----------
        states : xr.DataArray
            Query states to find neighbors for.
        k : int
            Number of random neighbors to return.
        rng : np.random.Generator
            Random number generator for sampling.
        minimum_timedelta_days : int | None, optional
            Minimum time separation in days between query and candidate states,
            by default None.
        dim_states : str | None, optional
            Dimension name for states, by default None.

        Returns
        -------
        xr.Dataset
            Dataset containing k randomly selected neighbor states.
        """
        sims = self.ds_similarities.sel(
            dayofyear=states.dayofyear,
            year=states.year,
            sample=states.sample,
            ensemble_member=states.ensemble_member,
        ).load()
        sims_flattened = -sims.similarities.stack(
            flat_dim=[d for d in sims.dims if d != dim_states]
        )  # assume that similarity increases the more similar the points are

        if minimum_timedelta_days is not None:
            keeps_minimum_distance = (
                abs(
                    (sims_flattened.valid_time - sims_flattened.c_valid_time)
                    / np.timedelta64(1, "D")
                )
                >= minimum_timedelta_days
            )
        else:
            keeps_minimum_distance = xr.ones_like(sims_flattened, dtype=bool)

        return -sims_flattened.isel(
            flat_dim=xr.apply_ufunc(
                get_k_random_indices,
                sims_flattened,
                keeps_minimum_distance,
                k,
                rng,
                input_core_dims=[["flat_dim"], ["flat_dim"], [], []],
                output_core_dims=[["neighbor"]],
                vectorize=True,
            )
        )

    def get_analog_data(
        self, queries: xr.DataArray, use_candidate_coords: bool = False
    ) -> xr.Dataset:
        """Retrieve analog weather data for specified query coordinates.

        Extracts weather data from the dataset at the coordinates specified
        in the query array, either using the query coordinates directly or
        the candidate coordinates.

        Parameters
        ----------
        queries : xr.DataArray
            Query array containing coordinate information.
        use_candidate_coords : bool, optional
            Whether to pick the sample according to provided coordinates of a candidate
            state or of a base state.
            by default False.

        Returns
        -------
        xr.Dataset
            Weather data at the specified coordinates.
        """
        da_wg = xr.open_zarr(self.path_dataset, decode_timedelta=True)[self.var]
        if use_candidate_coords:
            return da_wg.sel(
                dayofyear=queries.c_dayofyear,
                year=queries.c_year,
                sample=queries.c_sample,
                ensemble_member=queries.c_ensemble_member,
            )
        else:
            return da_wg.sel(
                dayofyear=queries.dayofyear,
                year=queries.year,
                sample=queries.sample,
                ensemble_member=queries.ensemble_member,
            )

    def plot_k_nearest_and_random_neighbors(
        self,
        state: xr.Dataset,
        k: int,
        rng: np.random.Generator,
        minimum_timedelta_days: int | None = None,
        vmin: float = 450,
        vmax: float = 600,
        minor_spacing_contours: float = 10,
        major_spacing_contours: float = 30,
    ) -> matplotlib.figure.Figure:
        """Create comparison plot of nearest neighbors vs random neighbors.

        Generates a visualization comparing the k nearest neighbors and k random
        neighbors for a given weather state, showing the base state, and the random and
        nearest neighbors (analogs) among the candidates and side by side.

        Parameters
        ----------
        state : xr.Dataset
            Reference weather state to find neighbors for.
        k : int
            Number of neighbors to display.
        rng : np.random.Generator
            Random number generator for random neighbor selection.
        minimum_timedelta_days : int | None, optional
            Minimum time separation constraint, by default None.
        vmin : float, optional
            Minimum value for color scale, by default 450.
        vmax : float, optional
            Maximum value for color scale, by default 600.
        minor_spacing_contours : float, optional
            Spacing for minor contour lines, by default 10.
        major_spacing_contours : float, optional
            Spacing for major contour lines, by default 30.

        Returns
        -------
        matplotlib.figure.Figure
            Figure containing the comparison plots.
        """
        similarities_nbs = self.get_similarities_k_closest_neighbors(
            states=state,
            k=k,
            minimum_timedelta_days=minimum_timedelta_days,
        )
        similarities_rands = self.get_similarities_k_random_neighbors(
            states=state,
            k=k,
            rng=rng,
            minimum_timedelta_days=minimum_timedelta_days,
        )

        da_nbs = self.get_analog_data(similarities_nbs, use_candidate_coords=True)
        da_rands = self.get_analog_data(similarities_rands, use_candidate_coords=True)
        da_base = self.get_analog_data(
            similarities_nbs, use_candidate_coords=False
        ).metpy.quantify()

        # preps for contour plots
        major_levels = np.arange(vmin, vmax, major_spacing_contours)
        minor_levels = np.arange(vmin, vmax, minor_spacing_contours)
        minor_levels = minor_levels[~np.isin(minor_levels, major_levels)]

        fig = plt.figure(figsize=(8, 14))
        gs = GridSpec(
            k + 3,
            2,
            figure=fig,
            height_ratios=[2, 0.2]
            + [1] * k
            + [
                0.1,
            ],
        )

        ax_cbar = fig.add_subplot(gs[-1, :])

        title_ax1 = fig.add_subplot(gs[1, 0])
        title_ax2 = fig.add_subplot(gs[1, 1])
        title_ax1.text(
            x=0.5,
            y=0.5,
            s="Nearest neighbors\namong candidates",
            horizontalalignment="center",
            fontsize="x-large",
        )
        title_ax2.text(
            x=0.5,
            y=0.5,
            s="Random samples\namong candidates",
            horizontalalignment="center",
            fontsize="x-large",
        )
        title_ax1.axis("off")
        title_ax2.axis("off")

        # actual plotting
        # base state:
        ax = fig.add_subplot(gs[0, :], projection=ccrs.Robinson())
        map_plot_without_frame_with_bounds(
            ax=ax,
            da=da_base,
            vmin=vmin,
            vmax=vmax,
            cbar_ax=ax_cbar,
            cbar_kwargs={
                "orientation": "horizontal",
            },
        )
        add_contours(
            ax=ax,
            da=da_base,
            major_levels=major_levels,
            minor_levels=minor_levels,
            add_labels=True,
        )

        if len(self.ds_similarities.c_sample) > 1:
            ax.set_title(
                r"$t_{init}$"
                + f": {np.datetime_as_string((da_base.init_time).squeeze(), unit='D')} "
                + r"$t_{lead}$"
                + f": {int((da_base.lead_time / np.timedelta64(1, 'D')).data)}d "
                + "$m$"
                + f": {da_base.ensemble_member.data} "
            )
        else:
            vt = da_base.init_time + da_base.lead_time
            ax.set_title(
                r"$t_{valid}$: "
                + f"{np.datetime_as_string((vt).squeeze(), unit='D')} "
                + "$m$"
                + f": {da_base.ensemble_member.data} "
            )

        for i in range(k):
            # nearest neigbors:
            ax_nb = fig.add_subplot(gs[2 + i, 0], projection=ccrs.Robinson())
            da_nb = da_nbs.isel(neighbor=i)
            map_plot_without_frame_with_bounds(
                ax=ax_nb, da=da_nb, add_colorbar=False, vmin=vmin, vmax=vmax
            )
            add_contours(
                ax=ax_nb, da=da_nb, major_levels=major_levels, minor_levels=minor_levels
            )
            if len(self.ds_similarities.c_sample) > 1:
                t_init_out = (da_nb.c_init_time).squeeze()
                ax_nb.set_title(
                    r"$t_{init}$: "
                    + f"{np.datetime_as_string(t_init_out, unit='D')}"
                    + r" $t_{lead}$"
                    + f": {int((da_nb.c_lead_time / np.timedelta64(1, 'D')).data)}d "
                    + "$m$"
                    + f": {da_nb.c_ensemble_member.data} "
                )
            else:
                vt = da_nb.c_init_time + da_nb.c_lead_time
                ax_nb.set_title(
                    r"$t_{valid}$"
                    + f": {np.datetime_as_string((vt).squeeze(), unit='D')} "
                    + "$m$"
                    + f": {da_nb.c_ensemble_member.data} "
                )

            # random neighbors:
            ax_rand = fig.add_subplot(gs[2 + i, 1], projection=ccrs.Robinson())
            da_rand = da_rands.isel(neighbor=i)
            map_plot_without_frame_with_bounds(
                ax=ax_rand, da=da_rand, add_colorbar=False, vmin=vmin, vmax=vmax
            )
            add_contours(
                ax=ax_rand,
                da=da_rand,
                major_levels=major_levels,
                minor_levels=minor_levels,
            )
            if len(self.ds_similarities.c_sample) > 1:
                t_init_out = (da_rand.c_init_time).squeeze()
                ax_rand.set_title(
                    r"$t_{init}$"
                    + f": {np.datetime_as_string(t_init_out, unit='D')} "
                    + r"$t_{lead}$"
                    + f": {int((da_rand.c_lead_time / np.timedelta64(1, 'D')).data)}d "
                    + "$m$"
                    + f": {da_rand.c_sample.data} "
                )
            else:
                vt = da_rand.c_init_time + da_rand.c_lead_time
                ax_rand.set_title(
                    r"$t_{valid}$"
                    + f": {np.datetime_as_string((vt).squeeze(), unit='D')} "
                    + "$m$"
                    + f": {da_rand.c_sample.data} "
                )
        return fig


@snakemake_handler
def main(snakemake: Any) -> None:
    """Main function for weather generator execution in Snakemake workflow.

    Initializes and runs the weather generator with parameters from Snakemake,
    handling logging and parameter management for the workflow execution.

    Parameters
    ----------
    snakemake : Any
        Snakemake object containing input/output paths, parameters, and logging
        configuration.
    """
    all_params = snakemake.params.all_params.copy()
    tracked_params = snakemake.params.tracked_params.copy()

    all_params["zarr_year_dayofyear"] = snakemake.input["zarr_year_dayofyear"]
    tracked_params["zarr_year_dayofyear"] = snakemake.input["zarr_year_dayofyear"]
    all_params["dir_wg"] = snakemake.output["dir_wg"]
    tracked_params["dir_wg"] = snakemake.output["dir_wg"]
    os.makedirs(snakemake.output["dir_wg"], exist_ok=True)

    with open(os.path.join(snakemake.output["dir_wg"], "params.yaml"), "w") as f:
        yaml.dump(tracked_params, f, default_flow_style=False, sort_keys=False)

    WeatherGenerator(params=all_params)


if __name__ == "__main__":
    main(snakemake=snakemake)  # noqa: F821
