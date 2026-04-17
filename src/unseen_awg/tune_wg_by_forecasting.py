"""Module for tuning weather generator parameters by optimizing forecast accuracy."""

from typing import Any, Literal, Tuple

import joblib
import numpy as np
import optuna
import xarray as xr
import yaml
from icecream import ic
from numpy.typing import NDArray
from xskillscore import crps_ensemble

from unseen_awg.data_classes import InitTimeLeadTimeMemberState
from unseen_awg.probability_models import (
    NormalProbabilityNotLargerThanFixedDate,
    ProbabilityModel,
)
from unseen_awg.snakemake_utils import snakemake_handler
from unseen_awg.time_steppers import StandardStepper, TimeStepper
from unseen_awg.utils import get_map_valid_n_day_transitions
from unseen_awg.weather_generator import WeatherGenerator


def get_gt_coords(sel_valid_time: np.datetime64, ds: xr.Dataset) -> dict[str, Any]:
    """For a given valid_time, get (ensemble_member, lead_time, init_time)
    that allow extracting the ground truth sample from the dataset.

    Parameters
    ----------
    sel_valid_time : np.datetime64
        The selected valid time.
    ds : xr.Dataset
        The dataset the weather generator is trained on.

    Returns
    -------
    dict[str, Any]
        Dictionary with ensemble_member, init_time, and lead_time keys.

    Raises
    ------
    NotImplementedError
        If the dataset type is not supported.
    """
    if ds.type == "era5":
        assert sel_valid_time in ds.init_time + ds.lead_time, (
            f"Selected time {sel_valid_time} is not in dataset."
        )
        return {
            "ensemble_member": 0,
            "init_time": ds.init_time.data[0],
            "lead_time": sel_valid_time - ds.init_time.data[0],
        }
    elif ds.type == "reforecasts":
        assert sel_valid_time in ds.init_time, (
            f"Selected time {sel_valid_time} is not an init_time in the dataset."
        )
        return {
            "ensemble_member": 0,
            "init_time": sel_valid_time,
            "lead_time": np.timedelta64(0, "D"),
        }
    else:
        raise NotImplementedError(f"Invalid dataset type: {ds.type}")


def get_gt(sel_valid_time: np.datetime64, ds: xr.Dataset) -> xr.Dataset:
    """Get ground truth data for a given valid time.

    Parameters
    ----------
    sel_valid_time : np.datetime64
        The selected valid time.
    ds : xr.Dataset
        The dataset containing the data.

    Returns
    -------
    xr.Dataset
        Dataset with the ground truth data for the selected time.
    """
    return ds.sel(get_gt_coords(sel_valid_time=sel_valid_time, ds=ds))


def persistence_forecast(ds_t0: xr.Dataset, lead_time: np.timedelta64) -> xr.Dataset:
    """Create persistence a forecast.

    Parameters
    ----------
    ds_t0 : xr.Dataset
        Initial dataset at time t0.
    lead_time : np.timedelta64
        Lead time to add to the valid time.

    Returns
    -------
    xr.Dataset
        Persistence forecast, i.e. initial data with modified valid_time coordinate.
    """
    return ds_t0.assign_coords({"valid_time": ds_t0.valid_time + lead_time})


def crps_persistence(
    ground_truth: xr.Dataset,
    ds_t0: xr.Dataset,
    forecast_lead_time: np.timedelta64,
    var: str,
) -> xr.Dataset:
    """Calculate CRPS (Continuous Ranked Probability Score) for a persistence forecast.

    Parameters
    ----------
    ground_truth : xr.Dataset
        Ground truth observations.
    ds_t0 : xr.Dataset
        Initial dataset at time t0.
    forecast_lead_time : np.timedelta64
        Lead time for the forecast.
    var : str
        Variable name to compute CRPS for.

    Returns
    -------
    xr.Dataset
        CRPS values for the persistence forecast.
    """
    return crps_ensemble(
        observations=ground_truth[[var]].load(),
        forecasts=persistence_forecast(ds_t0=ds_t0, lead_time=forecast_lead_time)[[var]]
        .load()
        .expand_dims(
            "member"
        ),  # member dimension trivial because we just project init sample into future
        dim=(),
    )


def crps_climatology(
    ground_truth: xr.Dataset,
    ds_t0: xr.Dataset,
    forecast_lead_time: np.timedelta64,
    ds_full: xr.Dataset,
    var: str,
) -> xr.Dataset:
    """Calculate CRPS (Continuous Ranked Probability Score) for a climatology forecast.

    Parameters
    ----------
    ground_truth : xr.Dataset
        Ground truth observations.
    ds_t0 : xr.Dataset
        Initial dataset at time t0.
    forecast_lead_time : np.timedelta64
        Lead time for the forecast.
    ds_full : xr.Dataset
        Full dataset for climatology calculation.
    var : str
        Variable name to compute CRPS for.

    Returns
    -------
    xr.Dataset
        CRPS values for the climatology forecast.
    """
    doy_target = (ds_t0.valid_time + forecast_lead_time).dt.dayofyear.squeeze()
    ds_stacked = ds_full.stack(member=("ensemble_member", "init_time", "lead_time"))

    res_chunks = []

    # to avoid memory errors, process each longitude separately
    for lon in ds_full.longitude:
        res_chunks.append(
            crps_ensemble(
                ground_truth[[var]].sel(longitude=lon).load(),
                ds_stacked[[var]]
                .sel(longitude=lon)
                .where(ds_stacked.valid_time.dt.dayofyear == doy_target.data, drop=True)
                .load(),
                dim=(),
            ).expand_dims("longitude", axis=-1)
        )
    return xr.concat(res_chunks, dim="longitude")


def analog_ensemble_forecast(
    wg: WeatherGenerator,
    probability_model: ProbabilityModel,
    initial_datapoint: InitTimeLeadTimeMemberState,
    lead_time: np.timedelta64,
    n_members: int,
    blocksize: int,
    rng: np.random.Generator,
    stepper_class: TimeStepper = StandardStepper,
) -> xr.Dataset:
    """Generate an ensemble forecast using analog sampling.

    Parameters
    ----------
    wg : WeatherGenerator
        Weather generator instance.
    probability_model : ProbabilityModel
        Probability model for the weather generator's analog selection.
    initial_datapoint : InitTimeLeadTimeMemberState
        Initial state for the forecast.
    lead_time : np.timedelta64
        Forecast lead time.
    n_members : int
        Number of ensemble members to be created in the forecast.
    blocksize : int
        Block size for weather generator sampling.
    rng : np.random.Generator
        Random number generator.
    stepper_class : TimeStepper, optional
        Time stepper class, by default StandardStepper.

    Returns
    -------
    xr.Dataset
        Trajectories of ensemble forecast.
    """
    # n_steps:
    n_steps = int(lead_time / (blocksize * np.timedelta64(1, "D"))) + 1

    trajs = xr.Dataset(
        {
            "lead_time": (
                ("forecast"),
                np.full((n_members), fill_value=np.timedelta64(0, "D")).astype(
                    "timedelta64[ns]"
                ),
            ),
            "init_time": (
                ("forecast"),
                np.full((n_members), fill_value=np.datetime64("2000-01-01", "ns")),
            ),
            "ensemble_member": (("forecast"), np.full((n_members), fill_value=np.nan)),
        }
    )

    for i in range(n_members):
        trajs.loc[{"forecast": i}] = wg.sample_trajectory(
            blocksize=blocksize,
            probability_model=probability_model,
            stepper_class=stepper_class,
            n_steps=n_steps,
            rng=rng,
            initialization=initial_datapoint,
        ).isel(out_time=-1)
    return trajs


def get_n_valid_forecast_start_points(
    n: int,
    wg: WeatherGenerator,
    rng: np.random.Generator,
    forecast_lead_time: np.timedelta64 = np.timedelta64(0, "D"),
    min_timedelta_from_dataset_start: np.timedelta64 = np.timedelta64(366, "D"),
    with_replacement: bool = True,
    balance_months: bool = True,
) -> xr.DataArray:
    """Select n valid ground truth data points to start forecasts from.

    Parameters
    ----------
    n : int
        Number of start points to select.
    wg : WeatherGenerator
        Weather generator instance.
    rng : np.random.Generator
        Random number generator used in the selection.
    forecast_lead_time : np.timedelta64, optional
        Forecast lead time, by default np.timedelta64(0, "D")
    min_timedelta_from_dataset_start : np.timedelta64, optional
        Samples are only considered as valid if min_timedelta_from_dataset_start
        from the start of the dataset, by default np.timedelta64(366, "D").
        This is because the analog weather generator relies only on analogs with
        valid_time smaller than the forecast start time. Therefore, on the end of the
        dataset, the set of analogs would be very small.
    with_replacement : bool, optional
        Whether to sample with replacement, by default True.
    balance_months : bool, optional
        Whether to balance months in sampling, by default True.

    Returns
    -------
    xr.DataArray
        DataArray of valid forecast start points.
    """
    transition_map = get_map_valid_n_day_transitions(
        wg.ds_similarities.init_time.load(),
        n=int(forecast_lead_time / np.timedelta64(1, "D")),
    )
    # if they all have the same init time, we have an ERA5-like WG.
    if (
        (wg.ds_similarities.init_time == wg.ds_similarities.init_time[0, 0, 0])
        | np.isnat(wg.ds_similarities.init_time)
    ).all():  # ERA5 case
        ss_valid_forecast_start_points = wg.ds_similarities.valid_time.data[
            (~np.isnan(transition_map.next_sample)).any(dim="sample")
        ]
        ss_valid_forecast_start_points = xr.Dataset(
            {
                "init_time": (
                    "datapoint",
                    np.array(
                        len(ss_valid_forecast_start_points)
                        * [wg.ds_similarities.init_time[0, 0, 0].data]
                    ),
                ),
                "lead_time": (
                    "datapoint",
                    ss_valid_forecast_start_points
                    - wg.ds_similarities.init_time[0, 0, 0].data,
                ),
                "ensemble_member": (
                    "datapoint",
                    0 * np.arange(len(ss_valid_forecast_start_points)),
                ),
            },
            coords={
                "datapoint": (
                    "datapoint",
                    np.arange(len(ss_valid_forecast_start_points)),
                )
            },
        )
    else:  # reforecast case
        # get a mask of the (doy, year) points for which we have
        # at least one sample with lead_time 0
        m = (wg.ds_similarities.lead_time == np.timedelta64(0, "D")).any(dim="sample")

        # extract valid times for this:
        vt_0 = wg.ds_similarities.valid_time.load().data[m]

        # get time to be predicted:
        vt_new = vt_0 + forecast_lead_time
        # transform back to year and dayofyear:
        yr = xr.DataArray(vt_new).dt.year
        doy = xr.DataArray(vt_new).dt.dayofyear
        # keep subset of those points for which we also have
        # a point with lead time zero at the new date.
        lead_times_new = wg.ds_similarities.lead_time.load().sel(year=yr, dayofyear=doy)
        m_new_lead_times = (lead_times_new == np.timedelta64(0, "D")).any(dim="sample")

        ss_valid_forecast_start_points = xr.Dataset(
            data_vars={
                "init_time": ("datapoint", vt_0[m_new_lead_times.data]),
                "lead_time": (
                    "datapoint",
                    np.repeat(np.timedelta64(0, "D"), repeats=m_new_lead_times.sum()),
                ),
                "ensemble_member": (
                    "datapoint",
                    np.repeat(0, repeats=m_new_lead_times.sum()),
                ),
            },
            coords={
                "datapoint": (
                    "datapoint",
                    np.arange(m_new_lead_times.sum()),
                )
            },
        )

    # to make sure we don't run out of candidates when sampling the trajectory,
    # allow setting a minimum time interval from dataset start.
    ss_valid_times = (
        ss_valid_forecast_start_points.init_time
        + ss_valid_forecast_start_points.lead_time
    )
    ss_valid_forecast_start_points = ss_valid_forecast_start_points.where(
        ss_valid_times > ss_valid_times.min() + min_timedelta_from_dataset_start,
        drop=True,
    )

    assert len(ss_valid_forecast_start_points.datapoint) >= n, (
        f"To make selection without replacement, require {n} datapoints, but only "
        + "{len(ss_valid_forecast_start_points.datapoint)} valid samples available."
    )
    ss_valid_forecast_start_points["valid_time"] = (
        ss_valid_forecast_start_points["init_time"]
        + ss_valid_forecast_start_points["lead_time"]
    )

    # in reforecast dataset, samples aren't distributed equally across months.
    # Therefore allow rebalancing them:
    if balance_months:
        months, counts = np.unique(
            ss_valid_forecast_start_points.valid_time.dt.month, return_counts=True
        )
        counts = xr.DataArray(counts, coords={"month": months})
        unnormalized_probabilities = 1 / counts.sel(
            month=ss_valid_forecast_start_points.valid_time.dt.month
        )
        p = unnormalized_probabilities / unnormalized_probabilities.sum()
    else:
        p = np.ones_like(ss_valid_forecast_start_points)
        p = p / p.sum()

    return ss_valid_forecast_start_points.isel(
        datapoint=rng.choice(
            np.arange(len(ss_valid_forecast_start_points.datapoint)),
            size=n,
            p=p,
            replace=with_replacement,
        )
    )


def crps_wg_forecasts(
    ds: xr.Dataset,
    wg: WeatherGenerator,
    n_members: int,
    probability_model: ProbabilityModel,
    forecast_init_time: np.datetime64,
    forecast_lead_time: np.timedelta64,
    blocksize: int,
    rng: np.random.Generator,
    stepper_class: TimeStepper = StandardStepper,
) -> xr.Dataset:
    """Compute CRPS (Continuous Ranked Probability Score) for analog forecasts.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset the weather generator samples from.
    wg : WeatherGenerator
        Weather generator instance.
    n_members : int
        Number of members in the analog forecast ensemble.
    probability_model : ProbabilityModel
        Probability model for analog selection.
    forecast_init_time : np.datetime64
        Initial time for the analog forecast.
    forecast_lead_time : np.timedelta64
        Lead time for the analog forecast.
    blocksize : int
        Block size (days) for weather generator sampling.
    rng : np.random.Generator
        Random number generator.
    stepper_class : TimeStepper, optional
        Time stepper class, by default StandardStepper

    Returns
    -------
    xr.Dataset
        CRPS for the weather generator's forecasts.
    """
    forecasts = analog_ensemble_forecast(
        wg=wg,
        probability_model=probability_model,
        initial_datapoint=InitTimeLeadTimeMemberState(
            **get_gt_coords(forecast_init_time, ds=ds)
        ),
        lead_time=forecast_lead_time,
        n_members=n_members,
        blocksize=blocksize,
        rng=rng,
        stepper_class=stepper_class,
    )

    fc_data = ds.sel(forecasts)
    gt_data = get_gt(sel_valid_time=forecast_init_time + forecast_lead_time, ds=ds)

    return crps_ensemble(
        observations=gt_data.load(),
        forecasts=fc_data.load(),
        member_dim="forecast",
        dim=(),
    )


def eval_climatology_persistence_forecast(
    sampled_timesteps: xr.DataArray,
    ds: xr.Dataset,
    ds_rechunk: xr.Dataset,
    var: str,
    forecast_lead_time: np.timedelta64,
) -> Tuple[NDArray, NDArray]:
    """Evaluate climatology and persistence forecasts.

    Parameters
    ----------
    sampled_timesteps : xr.DataArray
        Timesteps to initialize forecasts from.
    ds : xr.Dataset
        Dataset the weather generator samples from.
    ds_rechunk : xr.Dataset
        Rechunked dataset to allow faster computation of climatology forecasts.
    var : str
        Name of variable to evaluate.
    forecast_lead_time : np.timedelta64
        Forecast lead time.

    Returns
    -------
    Tuple[NDArray, NDArray]
        Tuple of (climatology_CRPS, persistence_CRPS) arrays.
    """
    crps_clim = []
    for i in range(len(sampled_timesteps.datapoint)):
        vt = (
            sampled_timesteps.isel(datapoint=i).init_time
            + sampled_timesteps.isel(datapoint=i).lead_time
        ).data
        crps_clim.append(
            crps_climatology(
                ds_full=ds_rechunk,
                ds_t0=get_gt(sel_valid_time=vt, ds=ds_rechunk),
                forecast_lead_time=forecast_lead_time,
                var=var,
                ground_truth=get_gt(
                    sel_valid_time=vt + forecast_lead_time, ds=ds_rechunk
                ),
            )
            .weighted(np.cos(np.deg2rad(ds_rechunk.latitude)))
            .mean()
            .compute()[var]
        )

    crps_clim = xr.concat(crps_clim, dim="datapoint").mean("datapoint")

    # same for persistence
    crps_pers = []
    for i in range(len(sampled_timesteps.datapoint)):
        vt = (
            sampled_timesteps.isel(datapoint=i).init_time
            + sampled_timesteps.isel(datapoint=i).lead_time
        ).data
        crps_pers.append(
            crps_persistence(
                ds_t0=get_gt(sel_valid_time=vt, ds=ds),
                forecast_lead_time=forecast_lead_time,
                ground_truth=get_gt(sel_valid_time=vt + forecast_lead_time, ds=ds),
                var=var,
            )
            .weighted(np.cos(np.deg2rad(ds.latitude)))
            .mean()
            .compute()[var]
        )

    crps_pers = xr.concat(crps_pers, dim="datapoint").mean("datapoint")

    return crps_clim.data, crps_pers.data


def eval_analogue_forecast_skill(
    sigma: float,
    type_probability_model: str,
    wg: WeatherGenerator,
    sampled_timesteps: xr.DataArray,
    ds: xr.Dataset,
    var: str,
    n_analogs: int,
    forecast_lead_time: np.timedelta64,
    blocksize: int,
    rng: np.random.Generator,
    stepper_class: TimeStepper = StandardStepper,
) -> NDArray:
    """Evaluate the skill of analog forecasts.

    Parameters
    ----------
    sigma : float
        Standard deviation parameter for the probability model.
    type_probability_model : str
        Type of probability model to use.
    wg : WeatherGenerator
        Weather generator instance.
    sampled_timesteps : xr.DataArray
        Timesteps to initialize forecasts from.
    ds : xr.Dataset
        Dataset the weather generator samples from.
    var : str
        Name of variable to evaluate.
    n_analogs : int
        Number of analog forecasts to create.
    forecast_lead_time : np.timedelta64
        Forecast lead time.
    blocksize : int
        Block size for sampling.
    rng : np.random.Generator
        Random number generator.
    stepper_class : TimeStepper, optional
        Time stepper class, by default StandardStepper.

    Returns
    -------
    NDArray
        Mean CRPS value for the analog forecasts.
    """
    res = np.full(shape=len(sampled_timesteps.datapoint), fill_value=np.nan)
    assert (
        type_probability_model == "NormalProbabilityNotLargerThanFixedDate"
    )  # for now, only allow this model - assures that no data from "future" is used.

    for i, ts in enumerate(sampled_timesteps.valid_time.data):
        r = crps_wg_forecasts(
            ds=ds[[var]],
            wg=wg,
            n_members=n_analogs,
            probability_model=NormalProbabilityNotLargerThanFixedDate(
                sigma, date_max=ts
            ),
            forecast_init_time=ts,
            forecast_lead_time=forecast_lead_time,
            blocksize=blocksize,
            rng=rng,
            stepper_class=stepper_class,
        )[var]

        res[i] = r.weighted(np.cos(np.deg2rad(r.latitude))).mean().compute()

    return np.mean(res)


def eval_wg(
    seed: int,
    forecast_lead_time_days: int,
    n_analogs: int,
    sigma_min: float,
    sigma_max: float,
    use_log_scale: bool,
    n_sampled_inits: int,
    min_timedelta_from_dataset_start: int,
    probability_model_type: str,
    var: str,
    blocksize: int,
    n_optuna_trials: int,
    path_ds: str,
    path_ds_rechunk: str,
    path_wg: str,
    path_study: str,
    ds_type: Literal["era5", "reforecasts"],
) -> None:
    """Evaluate weather generator in terms of its accuracy as ensemble forecast model.

    This function also runs climatology and persistance baseline forecasts and conducts
    an optuna study to optimize parameters of the weather generator and the
    trajectory sampling.

    Parameters
    ----------
    seed : int
        Seed used in sampling forecast initializations
    forecast_lead_time_days : int
        Lead time of the analog forecasts in days.
    n_analogs : int
        Number of ensemble members used in the analog ensemble forecast.
    sigma_min : float
        Lower bound of range of possible values of weather generator's sigma parameter.
    sigma_max : float
        Upper bound of range of possible values of weather generator's sigma parameter.
    use_log_scale : bool
        Use a logarithmic scaling during the optimization of the parameter sigma.
    n_sampled_inits : int
        Number of forecast initializations analog forecasts should be started from.
        Performance is averaged across initialization times.
    min_timedelta_from_dataset_start : int
        In some settings, the weather generator only uses analogs from timesteps before
        the forecast initialization. To avoid having few/zero analogs available, choose
        initialization times only min_timedelta_from_dataset_start days after the start
        of the dataset.
    probability_model_type : str
        A probability model to be used when sampling analogs.
    var : str
        Name of the variable to be evaluated.
    blocksize : int
        Blocksize of the simulated time series.
    n_optuna_trials : int
        Number of optimization steps in optuna.
    path_ds : str
        Path under which the dataset used during the evaluation is stored.
    path_ds_rechunk : str
        Path for dataset similar to the path_ds one but use chunking along longitudes.
    path_wg : str
        Path of the directory of the weather generator.
    path_study : str
        Path to write the optuna study to.
    ds_type : Literal[&quot;era5&quot;, &quot;reforecasts&quot;]
        Dataset used (ERA5 reanalysis or reforecasts).

    Raises
    ------
    ValueError
        If the weather generator contains negative values in its `lag` dimension.
    """
    forecast_lead_time = (np.timedelta64(1, "D") * forecast_lead_time_days).astype(
        "timedelta64[ns]"
    )
    ds = xr.open_zarr(path_ds, decode_timedelta=True)
    ds = ds.assign_attrs({"type": ds_type})

    if "valid_time" not in ds.coords:
        ds = ds.assign_coords({"valid_time": ds.init_time + ds.lead_time})
    ds_rechunk = xr.open_zarr(path_ds_rechunk, decode_timedelta=True)
    ds_rechunk = ds_rechunk.assign_attrs({"type": ds_type})
    if "valid_time" not in ds_rechunk.coords:
        ds_rechunk = ds_rechunk.assign_coords(
            {"valid_time": ds_rechunk.init_time + ds_rechunk.lead_time}
        )

    wg = WeatherGenerator.load(wg_path=path_wg)

    min_lag = xr.open_zarr(wg.path_dataset).lag.min() / np.timedelta64(1, "D")
    if min_lag < 0:
        raise ValueError(
            f"Dataset with negative lag (lag={min_lag.data}) supplied."
            + "Tuning necessitates comparisons with ground truth state at lead_time 0."
            + "Therefore, tuning isn't possible for weather generators with lag < 0."
            + "Lag < 0 imply accessing negative lead_time for ground truth sample."
        )

    rng = np.random.default_rng(seed=seed)

    sampled_timesteps = get_n_valid_forecast_start_points(
        n=n_sampled_inits,
        wg=wg,
        rng=rng,
        forecast_lead_time=forecast_lead_time,
        min_timedelta_from_dataset_start=np.timedelta64(
            min_timedelta_from_dataset_start, "D"
        ),
    )

    def objective(trial):
        sigma = trial.suggest_float(
            "sigma",
            sigma_min,
            sigma_max,
            log=use_log_scale,
        )
        pm_type = trial.suggest_categorical(
            "probability_model_type", probability_model_type
        )
        return eval_analogue_forecast_skill(
            sigma=sigma,
            type_probability_model=pm_type,
            wg=wg,
            sampled_timesteps=sampled_timesteps,
            ds=ds,
            var=var,
            n_analogs=n_analogs,
            forecast_lead_time=forecast_lead_time,
            blocksize=blocksize,
            rng=rng,
        )

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_optuna_trials, show_progress_bar=True)

    crps_clim, crps_pers = eval_climatology_persistence_forecast(
        sampled_timesteps=sampled_timesteps,
        ds=ds,
        ds_rechunk=ds_rechunk,
        var=var,
        forecast_lead_time=forecast_lead_time,
    )

    study.set_user_attr("crps_clim", crps_clim)
    study.set_user_attr("crps_pers", crps_pers)

    best_trial = study.best_trial
    ic(
        f"Best trial: sigma={best_trial.params['sigma']}, "
        + "probability_model_type={best_trial.params['probability_model_type']},"
        + " value={best_trial.value}",
    )

    joblib.dump(study, path_study)


@snakemake_handler
def main(snakemake: Any) -> None:
    all_params = dict(snakemake.params.all_params)
    tracked_params = dict(snakemake.params.tracked_params)

    seed = int(snakemake.wildcards.n_analogs)
    forecast_lead_time_days = int(snakemake.wildcards.forecast_lead_time_days)
    n_analogs = int(snakemake.wildcards.n_analogs)
    tracked_params["seed"] = seed
    tracked_params["n_analogs"] = n_analogs
    tracked_params["forecast_lead_time_days"] = forecast_lead_time_days

    with open(snakemake.output.params, "w") as f:
        yaml.dump(tracked_params, f, default_flow_style=False, sort_keys=False)

    eval_wg(
        seed=seed,
        forecast_lead_time_days=forecast_lead_time_days,
        n_analogs=n_analogs,
        sigma_min=all_params["tune_weather_generator.sigma_min"],
        sigma_max=all_params["tune_weather_generator.sigma_max"],
        use_log_scale=all_params["tune_weather_generator.use_log_scale"],
        n_sampled_inits=all_params["tune_weather_generator.N_sampled_inits"],
        min_timedelta_from_dataset_start=all_params[
            "tune_weather_generator.min_timedelta_from_dataset_start"
        ],
        probability_model_type=all_params[
            "tune_weather_generator.probability_model_type"
        ],
        var=all_params["tune_weather_generator.var"],
        blocksize=all_params["tune_weather_generator.blocksize"],
        n_optuna_trials=all_params["tune_weather_generator.N_optuna_trials"],
        path_ds=snakemake.input.path_ds,
        path_ds_rechunk=snakemake.input.path_ds_rechunk,
        path_wg=snakemake.input.dir_wg,
        path_study=snakemake.output.study_pkl,
        ds_type=snakemake.config["dataset_type"],
    )


if __name__ == "__main__":
    main(snakemake=snakemake)  # noqa: F821
