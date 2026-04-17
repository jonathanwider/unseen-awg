"""
Trajectory simulation module for weather generator within Snakemake workflows.

This module provides functionality to simulate weather trajectories using analog
sampling methods with configurable probability models and time stepping approaches.
"""

from typing import Any

import numpy as np
import yaml

from unseen_awg.probability_models import (
    NormalProbabilityKeepMinimalNDays,
    NormalProbabilityModel,
)
from unseen_awg.snakemake_utils import snakemake_handler
from unseen_awg.time_steppers import StandardStepper
from unseen_awg.weather_generator import WeatherGenerator


def simulate_trajectory(
    path_wg: str,
    probability_model: str,
    sigma: float,
    seed: int,
    n_days: int,
    blocksize: int,
    path_trajectory: str,
    n_days_min: int | None = None,
) -> None:
    """Simulate and save a time series with a weather generator.

    Parameters
    ----------
    path_wg : str
        Path of the directory of the weather generator.
    probability_model : str
        Name of a probability model to be used when sampling analogs.
    sigma : float
        Sigma parameter of the proability model used during sampling.
    seed : int
        Seed for the random sampling of the time series.
    n_days : int
        Number of days in the final time series. Gets rounded to
        conform with the selected blocksize.
    blocksize : int
        Size of the contiguous blocks of states during the sampling of the time series.
    path_trajectory : str
        Path to store the trajectory in.
    n_days_min : int | None, optional
        To be used in combination with the "KeepMinimalNDays"  probability model, avoids
        sampling states closer than `n_days_min` from the true sample. By default None.

    Raises
    ------
    ValueError
        If an invalid name of a probability model was specified.
    """
    # Load weather generator
    wg = WeatherGenerator.load(path_wg)

    # Configure probability model based on transition model type
    if probability_model == "NoRestrictions":
        probability_model = NormalProbabilityModel(sigma=sigma)
    elif probability_model == "KeepMinimalNDays":
        assert n_days_min is not None, (
            "n_days_min cannot be none if transition model KeepMinimalNDays is chosen."
        )
        probability_model = NormalProbabilityKeepMinimalNDays(
            sigma=sigma, n_days_min=n_days_min
        )
    else:
        raise ValueError(
            f"Invalid argument for probability_model type: {probability_model}"
        )

    # Initialize random number generator
    rng = np.random.default_rng(seed)

    # Calculate number of simulation steps
    n_steps = int(np.ceil(n_days / blocksize))

    # Generate trajectory
    traj = wg.sample_trajectory(
        blocksize=blocksize,
        probability_model=probability_model,
        stepper_class=StandardStepper,
        n_steps=n_steps,
        rng=rng,
        show_progressbar=False,
    )

    # Save trajectory to NetCDF file
    traj.to_netcdf(path_trajectory)


@snakemake_handler
def main(snakemake: Any) -> None:
    all_params = snakemake.params.all_params
    tracked_params = snakemake.params.tracked_params
    seed = int(snakemake.wildcards.seed)
    sigma = float(snakemake.wildcards.sigma)
    blocksize = int(snakemake.wildcards.blocksize)

    # Update tracked parameters with current run values
    tracked_params["seed"] = seed
    tracked_params["sigma"] = sigma
    tracked_params["blocksize"] = blocksize

    # Save parameters to output file
    with open(snakemake.output.params, "w") as f:
        yaml.dump(tracked_params, f, default_flow_style=False, sort_keys=False)

    simulate_trajectory(
        path_wg=snakemake.input.path_wg,
        probability_model=all_params["simulate_trajectory.probability_model"],
        sigma=sigma,
        seed=seed,
        n_days=all_params["simulate_trajectory.n_days"],
        blocksize=blocksize,
        path_trajectory=snakemake.output.nc_trajectory,
        n_days_min=all_params["simulate_trajectory.n_days_min"],
    )


if __name__ == "__main__":
    main(snakemake=snakemake)  # noqa: F821
