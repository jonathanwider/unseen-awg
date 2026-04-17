rule weather_generator:
    input:
        zarr_year_dayofyear=os.path.join(
            get_preprocessed_dir("circulation", "{merge_ds_type}"),
            "data_year_dayofyear_{hash_combine_circulation}_{hash_merge_restructure_ds}.zarr",
        ),
    output:
        dir_wg=directory(
            os.path.join(
                config["paths"]["dir_wgs"],
                "wg_{merge_ds_type}_{hash_combine_circulation}_{hash_merge_restructure_ds}_{hash_wg}",
            )
        ),
    params:
        all_params=get_all_params_dict("weather_generator"),
        tracked_params=get_params_dict_for_saving("weather_generator"),
    resources:
        runtime=4320,
        mem_mb_per_cpu=GB_TO_MB * 8,
        cpus_per_task=8,
    conda:
        "unseen_awg"
    log:
        **get_log_paths(
            config["paths"]["dir_wgs"],
            "weather_generator",
            "_{merge_ds_type}_{hash_combine_circulation}_{hash_merge_restructure_ds}_{hash_wg}",
        ),
    script:
        f"{DIR_CODE_CORE}/src/unseen_awg/weather_generator.py"


rule tune_weather_generator:
    input:
        dir_wg=os.path.join(
            config["paths"]["dir_wgs"],
            "wg_{merge_ds_type}_{hash_combine_circulation}_{hash_merge_restructure_ds}_{hash_wg}",
        ),
        path_ds_rechunk=os.path.join(
            get_preprocessed_dir("circulation", "{merge_ds_type}"),
            "rechunk_combined_{hash_combine_circulation}.zarr",
        ),
        path_ds=os.path.join(
            get_preprocessed_dir("circulation", "{merge_ds_type}"),
            "combined_{hash_combine_circulation}.zarr",
        ),
    output:
        study_pkl=os.path.join(
            config["paths"]["dir_wgs"],
            "wg_{merge_ds_type}_{hash_combine_circulation}_{hash_merge_restructure_ds}_{hash_wg}_tune",
            "{seed}_{n_analogs}_{forecast_lead_time_days}_optuna_study.pkl",
        ),
        params=os.path.join(
            config["paths"]["dir_wgs"],
            "wg_{merge_ds_type}_{hash_combine_circulation}_{hash_merge_restructure_ds}_{hash_wg}_tune",
            "{seed}_{n_analogs}_{forecast_lead_time_days}_params.yaml",
        ),
    params:
        all_params=get_all_params_dict("tune_weather_generator"),
        tracked_params=get_params_dict_for_saving("tune_weather_generator"),
    resources:
        runtime=12000,
        mem_mb_per_cpu=GB_TO_MB * 16,
        cpus_per_task=1,
    conda:
        "unseen_awg"
    log:
        **get_log_paths(
            config["paths"]["dir_wgs"],
            "tune_weather_generator",
            "_{merge_ds_type}_{hash_combine_circulation}_{hash_merge_restructure_ds}"
            + "_{hash_wg}_{seed}_{n_analogs}_{forecast_lead_time_days}",
        ),
    script:
        f"{DIR_CODE_CORE}/src/unseen_awg/tune_wg_by_forecasting.py"


rule simulate_trajectory:
    input:
        path_wg=os.path.join(
            config["paths"]["dir_wgs"],
            "wg_{merge_ds_type}_{hash_combine_circulation}_{hash_merge_restructure_ds}_{hash_wg}",
        ),
    output:
        nc_trajectory=os.path.join(
            config["paths"]["dir_simulations"],
            "wg_{merge_ds_type}_{hash_combine_circulation}_{hash_merge_restructure_ds}_{hash_wg}_{hash_traj}/",
            "{seed}_{sigma}_{blocksize}/trajectory.nc",
        ),
        params=os.path.join(
            config["paths"]["dir_simulations"],
            "wg_{merge_ds_type}_{hash_combine_circulation}_{hash_merge_restructure_ds}_{hash_wg}_{hash_traj}/",
            "{seed}_{sigma}_{blocksize}/params.yaml",
        ),
    params:
        all_params=get_all_params_dict("simulate_trajectory"),
        tracked_params=get_params_dict_for_saving("simulate_trajectory"),
    log:
        **get_log_paths(
            os.path.join(
                config["paths"]["dir_simulations"],
                "wg_{merge_ds_type}_{hash_combine_circulation}_{hash_merge_restructure_ds}_{hash_wg}_{hash_traj}/",
            ),
            "simulate_trajectory",
            "_{seed}_{sigma}_{blocksize}",
        ),
    resources:
        runtime=2500,
        mem_mb_per_cpu=GB_TO_MB * 16,
        cpus_per_task=1,
    conda:
        "unseen_awg"
    script:
        f"{DIR_CODE_CORE}/src/unseen_awg/simulate_trajectory.py"
