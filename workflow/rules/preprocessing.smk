rule preprocess_circulation_reforecasts:
    output:
        zarr=directory(
            os.path.join(
                get_preprocessed_dir("circulation", "reforecasts"),
                "combined_{hash_preprocess_circulation_reforecasts}.zarr",
            )
        ),
        params=os.path.join(
            get_preprocessed_dir("circulation", "reforecasts"),
            "params_combined_{hash_preprocess_circulation_reforecasts}.yaml",
        ),
    params:
        all_params=get_all_params_dict("preprocess_circulation_reforecasts"),
        tracked_params=get_params_dict_for_saving("preprocess_circulation_reforecasts"),
    resources:
        runtime=2880,  # 2 days in minutes
        mem_mb_per_cpu=GB_TO_MB * 64,
        cpus_per_task=1,
    conda:
        "unseen_awg"
    log:
        **get_log_paths(
            get_preprocessed_dir("circulation", "reforecasts"),
            "preprocess_circulation_reforecasts",
            "_{hash_preprocess_circulation_reforecasts}",
        ),
    script:
        f"{DIR_CODE_CORE}/src/unseen_awg/preprocessing/preprocess_reforecasts.py"


rule preprocess_impact_variables_reforecasts:
    output:
        zarr=directory(
            os.path.join(
                get_preprocessed_dir("impact_variables", "reforecasts"),
                "combined_{hash_preprocess_impact_variables_reforecasts}.zarr",
            )
        ),
        params=os.path.join(
            get_preprocessed_dir("impact_variables", "reforecasts"),
            "params_combined_{hash_preprocess_impact_variables_reforecasts}.yaml",
        ),
    params:
        all_params=get_all_params_dict("preprocess_impact_variables_reforecasts"),
        tracked_params=get_params_dict_for_saving(
            "preprocess_impact_variables_reforecasts"
        ),
    resources:
        runtime=2880,  # 2 days in minutes
        mem_mb_per_cpu=GB_TO_MB * 64,
        cpus_per_task=1,
    conda:
        "unseen_awg"
    log:
        **get_log_paths(
            get_preprocessed_dir("impact_variables", "reforecasts"),
            "preprocess_impact_variables_reforecasts",
            "_{hash_preprocess_impact_variables_reforecasts}",
        ),
    script:
        f"{DIR_CODE_CORE}/src/unseen_awg/preprocessing/preprocess_reforecasts.py"


rule preprocess_era5_single:
    output:
        nc_file=os.path.join(
            get_preprocessed_dir("", "era5"), "{var_type}-{year}-{var}-{grid}.nc"
        ),
    resources:
        runtime=30,
        mem_mb_per_cpu=GB_TO_MB * 8,
        cpus_per_task=1,
    conda:
        "unseen_awg"
    log:
        **get_log_paths(
            get_preprocessed_dir("", "era5"),
            "preprocess_era5_single",
            "_{var_type}-{year}-{var}-{grid}",
        ),
    script:
        f"{DIR_CODE_CORE}/src/unseen_awg/preprocessing/ds_from_era5_single.py"


rule preprocess_circulation_era5:
    input:
        ERA5_CIRCULATION_FILES,
    output:
        zarr=directory(
            os.path.join(
                get_preprocessed_dir("circulation", "era5"),
                "combined_{hash_preprocess_circulation_era5}.zarr",
            )
        ),
        params=os.path.join(
            get_preprocessed_dir("circulation", "era5"),
            "params_combined_{hash_preprocess_circulation_era5}.yaml",
        ),
    params:
        all_params=get_all_params_dict("preprocess_circulation_era5"),
        tracked_params=get_params_dict_for_saving("preprocess_circulation_era5"),
    conda:
        "unseen_awg"
    resources:
        runtime=60,
        mem_mb_per_cpu=GB_TO_MB * 16,
        cpus_per_task=1,
    log:
        **get_log_paths(
            get_preprocessed_dir("circulation", "era5"),
            "preprocess_circulation_era5",
            "_{hash_preprocess_circulation_era5}",
        ),
    script:
        f"{DIR_CODE_CORE}/src/unseen_awg/preprocessing/ds_from_era5.py"


rule preprocess_impact_variables_era5:
    input:
        ERA5_IMPACT_VARIABLES_FILES,
    output:
        zarr=directory(
            os.path.join(
                get_preprocessed_dir("impact_variables", "era5"),
                "combined_{hash_preprocess_impact_variables_era5}.zarr",
            )
        ),
        params=os.path.join(
            get_preprocessed_dir("impact_variables", "era5"),
            "params_combined_{hash_preprocess_impact_variables_era5}.yaml",
        ),
    params:
        all_params=get_all_params_dict("preprocess_impact_variables_era5"),
        tracked_params=get_params_dict_for_saving("preprocess_impact_variables_era5"),
    conda:
        "unseen_awg"
    resources:
        runtime=60,
        mem_mb_per_cpu=GB_TO_MB * 64,
        cpus_per_task=1,
    log:
        **get_log_paths(
            get_preprocessed_dir("impact_variables", "era5"),
            "preprocess_impact_variables_era5",
            "_{hash_preprocess_impact_variables_era5}",
        ),
    script:
        f"{DIR_CODE_CORE}/src/unseen_awg/preprocessing/ds_from_era5.py"


rule rechunk_ds:
    input:
        path_zarr=os.path.join(
            get_preprocessed_dir("{var_type}", "{dataset_type}"),
            "combined{name}.zarr",
        ),
    output:
        path_rechunk=directory(
            os.path.join(
                get_preprocessed_dir("{var_type}", "{dataset_type}"),
                "rechunk_combined{name}.zarr",
            )
        ),
    resources:
        runtime=5760,  # 4 days in minutes
        mem_mb_per_cpu=GB_TO_MB * 16,
        cpus_per_task=8,
    conda:
        "unseen_awg"
    log:
        **get_log_paths(
            get_preprocessed_dir("{var_type}", "{dataset_type}"),
            "rechunk_ds",
            "{name}",
        ),
    script:
        f"{DIR_CODE_CORE}/src/unseen_awg/preprocessing/rechunk_ds.py"


# bring data in format that allows more efficient processing in weather generator.


rule merge_restructure_ds:
    input:
        path_zarr=os.path.join(
            get_preprocessed_dir("circulation", "{merge_ds_type}"),
            "combined_{hash_combine_circulation}.zarr",
        ),
    output:
        nc_file=os.path.join(
            get_preprocessed_dir("circulation", "{merge_ds_type}"),
            "restructured_{hash_combine_circulation}_{hash_merge_restructure_ds}.nc",
        ),
        params=os.path.join(
            get_preprocessed_dir("circulation", "{merge_ds_type}"),
            "params_restructured_{hash_combine_circulation}_{hash_merge_restructure_ds}.yaml",
        ),
        nc_mu_sigma=os.path.join(
            get_preprocessed_dir("circulation", "{merge_ds_type}"),
            "mu_sigma_{hash_combine_circulation}_{hash_merge_restructure_ds}.nc",
        ),
    params:
        all_params=get_all_params_dict("merge_restructure_ds"),
        tracked_params=get_params_dict_for_saving("merge_restructure_ds"),
    resources:
        runtime=120,  # 2 h in minutes
        mem_mb_per_cpu=GB_TO_MB * 64,
        cpus_per_task=1,
    conda:
        "unseen_awg"
    log:
        **get_log_paths(
            get_preprocessed_dir("circulation", "{merge_ds_type}"),
            "merge_restructure_ds",
            "_{hash_combine_circulation}_{hash_merge_restructure_ds}",
        ),
    script:
        f"{DIR_CODE_CORE}/src/unseen_awg/preprocessing/merge_restructure_reforecasts.py"


# further restucturing
rule to_year_dayofyear_format:
    input:
        nc_file=os.path.join(
            get_preprocessed_dir("circulation", "{merge_ds_type}"),
            "restructured_{hash_combine_circulation}_{hash_merge_restructure_ds}.nc",
        ),
    output:
        zarr_year_dayofyear=directory(
            os.path.join(
                get_preprocessed_dir("circulation", "{merge_ds_type}"),
                "data_year_dayofyear_{hash_combine_circulation}_{hash_merge_restructure_ds}.zarr",
            )
        ),
    resources:
        runtime=120,  # 2 h in minutes
        mem_mb_per_cpu=GB_TO_MB * 1,
        cpus_per_task=1,
    conda:
        "unseen_awg"
    log:
        **get_log_paths(
            get_preprocessed_dir("circulation", "{merge_ds_type}"),
            "to_year_dayofyear_format",
            "_{hash_combine_circulation}_{hash_merge_restructure_ds}",
        ),
    script:
        f"{DIR_CODE_CORE}/src/unseen_awg/preprocessing/year_dayofyear_format.py"
