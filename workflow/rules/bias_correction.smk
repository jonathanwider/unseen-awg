
rule compute_climatology:
    input:
        zarr_rechunk=os.path.join(
            get_preprocessed_dir("{var_type}", "{dataset_type}"),
            "rechunk_{name}.zarr",
        ),
    output:
        nc_clim=os.path.join(
            config["paths"]["dir_preprocessed_datasets"],
            "climatology_preprocessed_{var_type}_{dataset_type}/{name}_"
            + "{hash_compute_climatology}.nc",
        ),
        params=os.path.join(
            config["paths"]["dir_preprocessed_datasets"],
            "climatology_preprocessed_{var_type}_{dataset_type}",
            "params-clim_{name}_{hash_compute_climatology}.yaml",
        ),
    params:
        all_params=get_all_params_dict("compute_climatology"),
        tracked_params=get_params_dict_for_saving("compute_climatology"),
    resources:
        runtime=2880,
        mem_mb_per_cpu=GB_TO_MB * 64,
        cpus_per_task=1,
    conda:
        "unseen_awg"
    log:
        **get_log_paths(
            os.path.join(config["paths"]["dir_preprocessed_datasets"],
            "climatology_preprocessed_{var_type}_{dataset_type}"),
            "compute_climatology",
            "_{name}_{hash_compute_climatology}",
        ),
    script:
        f"{DIR_CODE_CORE}/src/unseen_awg/preprocessing/compute_climatology.py"


rule compute_bias:
    input:
        path_era5=os.path.join(
            get_preprocessed_dir("{var_type}", "era5"),
            "rechunk_combined_{hash_era5}.zarr",
        ),
        path_reforecasts=os.path.join(
            get_preprocessed_dir("{var_type}", "reforecasts"),
            "rechunk_combined_{hash_re}.zarr",
        ),
    output:
        reforecasts=directory(
            os.path.join(
                os.path.join(
                    get_preprocessed_dir("{var_type}", "reforecasts"),
                    "bias_{hash_era5}_{hash_re}_{hash_bias}",
                ),
                "reforecasts_{i_lead_time}_mode_{split_mode}_n_{n_partitions}.zarr",
            )
        ),
        era5=directory(
            os.path.join(
                os.path.join(
                    get_preprocessed_dir("{var_type}", "reforecasts"),
                    "bias_{hash_era5}_{hash_re}_{hash_bias}",
                ),
                "era5_{i_lead_time}_mode_{split_mode}_n_{n_partitions}.zarr",
            )
        ),
        params=os.path.join(
            get_preprocessed_dir("{var_type}", "reforecasts"),
            "params-bias_{hash_era5}_{hash_re}_{i_lead_time}_mode_{split_mode}_"
            + "{hash_bias}_n_{n_partitions}.yaml",
        ),
    params:
        all_params=get_all_params_dict("compute_bias"),
        tracked_params=get_params_dict_for_saving("compute_bias"),
    resources:
        runtime=600,
        mem_mb_per_cpu=GB_TO_MB * 16,
        cpus_per_task=1,
    conda:
        "unseen_awg"
    log:
        **get_log_paths(
            get_preprocessed_dir("{var_type}", "reforecasts"),
            "compute_bias",
            "_{hash_era5}_{hash_re}_{i_lead_time}_{split_mode}_{hash_bias}_n{n_partitions}",
        ),
    script:
        f"{DIR_CODE_CORE}/src/unseen_awg/preprocessing/compute_bias.py"


rule merge_bias:
    input:
        reforecasts_tmp=TMP_BIAS_CORRECTION_REFORECASTS,
        era5_tmp=TMP_BIAS_CORRECTION_ERA5,
        path_reforecasts=os.path.join(
            get_preprocessed_dir("{var_type}", "reforecasts"),
            "combined_{hash_re}.zarr",
        ),
    output:
        era5=directory(
            os.path.join(
                os.path.join(
                    get_preprocessed_dir("{var_type}", "reforecasts"),
                    "bias_{hash_era5}_{hash_re}_{hash_bias}",
                ),
                "era5_mode_{split_mode}_n_{n_partitions}.zarr",
            )
        ),
        reforecasts=directory(
            os.path.join(
                os.path.join(
                    get_preprocessed_dir("{var_type}", "reforecasts"),
                    "bias_{hash_era5}_{hash_re}_{hash_bias}",
                ),
                "reforecasts_mode_{split_mode}_n_{n_partitions}.zarr",
            )
        ),
    log:
        **get_log_paths(
            get_preprocessed_dir("{var_type}", "reforecasts"),
            "merge_bias",
            "_{hash_era5}_{hash_re}_{hash_bias}_{split_mode}_n_{n_partitions}",
        ),
    resources:
        runtime=360,
        mem_mb_per_cpu=GB_TO_MB * 16,
        cpus_per_task=1,
    conda:
        "unseen_awg"
    script:
        f"{DIR_CODE_CORE}/src/unseen_awg/preprocessing/merge_bias_results.py"


rule correct_bias:
    input:
        era5=os.path.join(
            os.path.join(
                get_preprocessed_dir("{var_type}", "reforecasts"),
                "bias_{hash_era5}_{hash_re}_{hash_bias}",
            ),
            "era5_mode_chronological_n_1.zarr",
        ),
        reforecasts=os.path.join(
            os.path.join(
                get_preprocessed_dir("{var_type}", "reforecasts"),
                "bias_{hash_era5}_{hash_re}_{hash_bias}",
            ),
            "reforecasts_mode_chronological_n_1.zarr",
        ),
        dataset_to_be_corrected=os.path.join(
            get_preprocessed_dir("{var_type}", "reforecasts"),
            "combined_{hash_re}.zarr",
        ),
    output:
        path_reforecasts=directory(
            os.path.join(
                get_preprocessed_dir("{var_type}", "reforecasts"),
                "combined-corrected_{hash_era5}_{hash_re}_{hash_bias}_"
                + "{hash_correct_bias}.zarr",
            )
        ),
        params=os.path.join(
            get_preprocessed_dir("{var_type}", "reforecasts"),
            "params_combined-corrected_{hash_era5}_{hash_re}_{hash_bias}_"
            + "{hash_correct_bias}.yaml",
        ),
    params:
        all_params=get_all_params_dict("correct_bias"),
        tracked_params=get_params_dict_for_saving("correct_bias"),
    log:
        **get_log_paths(
            get_preprocessed_dir("{var_type}", "reforecasts"),
            "correct_bias",
            "_{hash_era5}_{hash_re}_{hash_bias}_{hash_correct_bias}",
        ),
    resources:
        runtime=360,
        mem_mb_per_cpu=GB_TO_MB * 16,
        cpus_per_task=4,
    conda:
        "unseen_awg"
    script:
        f"{DIR_CODE_CORE}/src/unseen_awg/preprocessing/correct_bias.py"
