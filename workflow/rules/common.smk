import hashlib
import json
import os
import numpy as np


# Constants
GB_TO_MB = 1024


def get_tracked_params(rule_name):
    params_config = config.get(rule_name, {})
    all_params = params_config.get("all_params", [])
    untracked_params = params_config.get("untracked_params", [])
    return [param for param in all_params if param not in (untracked_params or [])]


def get_all_params(rule_name):
    params_config = config.get(rule_name, {})
    return params_config.get("all_params", [])


def get_param_value(param_path):
    """
    Get a parameter value from config using dot notation or direct key.
    Examples:
      - "n_days_max" -> config["n_days_max"]
      - "paths.output_dir" -> config["paths"]["output_dir"]
    """
    keys = param_path.split(".")
    value = config
    for key in keys:
        value = value[key]
    return value


def get_params_hash(rule_name):
    """Generate a hash from the tracked parameters for a specific rule"""
    tracked_params = get_tracked_params(rule_name)
    params_dict = {}
    for param in tracked_params:
        try:
            params_dict[param] = get_param_value(param)
        except (KeyError, TypeError) as e:
            raise ValueError(f"Parameter '{param}' not found in config: {e}")

    # Convert to JSON string for consistent hashing
    params_str = json.dumps(params_dict, sort_keys=True)
    # Create short hash (first 8 characters)
    return hashlib.md5(params_str.encode()).hexdigest()[:8]


def get_params_dict_for_saving(rule_name):
    """Get the full parameters dictionary for saving (only tracked params)"""
    tracked_params = get_tracked_params(rule_name)
    # Save only tracked parameters
    params_dict = {}
    for param in tracked_params:
        try:
            params_dict[param] = get_param_value(param)
        except (KeyError, TypeError) as e:
            raise ValueError(f"Parameter '{param}' not found in config: {e}")
    return params_dict


def get_all_params_dict(rule_name):
    """Get the full parameters dictionary for saving (only tracked params)"""
    tracked_params = get_all_params(rule_name)
    # Save only tracked parameters
    params_dict = {}
    for param in tracked_params:
        try:
            params_dict[param] = get_param_value(param)
        except (KeyError, TypeError) as e:
            raise ValueError(f"Parameter '{param}' not found in config: {e}")
    return params_dict


def get_log_paths(base_dir, rule_name, wildcards_str=""):
    """Generate standardized log paths."""
    log_dir = os.path.join(base_dir, "logs")
    return {
        "stdout": os.path.join(log_dir, f"log_{rule_name}{wildcards_str}.out"),
        "stderr": os.path.join(log_dir, f"log_{rule_name}{wildcards_str}.err"),
    }


def get_preprocessed_dir(var_type, dataset_type):
    return os.path.join(
        config["paths"]["dir_preprocessed_datasets"],
        f"preprocessed_{var_type}_{dataset_type}",
    )


def validate_config() -> None:
    # Validate required config keys
    required_keys = ["dataset_type", "paths"]
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required config key: {key}")

    # Validate dataset_type
    valid_types = ["era5", "reforecasts"]
    if config["dataset_type"] not in valid_types:
        raise ValueError(
            f"Invalid dataset_type: {config['dataset_type']}. Must be one of {valid_types}"
        )

    # Validate paths
    required_paths = [
        "dir_preprocessed_datasets",
        "dir_wgs",
        "dir_simulations",
        "dir_code_core",
    ]
    for path_key in required_paths:
        if path_key not in config.get("paths", {}):
            raise ValueError(f"Missing required path config: paths.{path_key}")


def get_era5_files(rulename):
    var_config = config[rulename]
    return expand(
        os.path.join(
            get_preprocessed_dir("", "era5"), "{var_type}-{year}-{var}-{grid}.nc"
        ),
        var_type=var_config["var_type"],
        year=range(config["era5_year_min"], config["era5_year_max"] + 1),
        var=var_config["used_vars"],
        grid=var_config["target_grid"],
    )


# Helper function for bias correction files
def get_bias_correction_files(dataset, var_type="{var_type}"):
    return expand(
        os.path.join(
            get_preprocessed_dir(var_type, "reforecasts"),
            "bias_{hash_era5}_{hash_re}_{hash_bias}",
            f"{dataset}" + "_{i_lead_time}_mode_{split_mode}_n_{n_partitions}.zarr",
        ),
        i_lead_time=np.arange(
            config["lead_time_days_low"],
            config["lead_time_days_high"] + 1,
        ),
        allow_missing=True,
    )
