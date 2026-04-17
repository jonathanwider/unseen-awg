"""
Module for defining standard geographical grids used in weather generation and
evaluation of weather generator.
"""

from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr
import yaml

# Get the directory containing this file
current_dir: Path = Path(__file__).parents[2]

lon_attrs = {
    "standard_name": "longitude",
    "long_name": "longitude",
    "units": "degrees_east",
}
lat_attrs = {
    "standard_name": "latitude",
    "long_name": "latitude",
    "units": "degrees_north",
}

# Load paths
with open(f"{current_dir}/configs/paths.yaml", "r") as file:
    paths: Any = yaml.safe_load(file)["paths"]

# Standard grid for analog selection with fine resolution
GRID_ANALOGS: xr.Dataset = xr.Dataset(
    {
        "latitude": (["latitude"], np.arange(30, 74, 2.5), lat_attrs),
        "longitude": (["longitude"], np.arange(-80, 42.5, 2.5), lon_attrs),
    }
)

# Standard grid for analog selection with medium resolution
GRID_ANALOGS_SMALL: xr.Dataset = xr.Dataset(
    {
        "latitude": (["latitude"], np.arange(30, 74, 5.0), lat_attrs),
        "longitude": (["longitude"], np.arange(-80, 45, 5.0), lon_attrs),
    }
)

# Standard grid for analog selection with coarse resolution
GRID_ANALOGS_TINY: xr.Dataset = xr.Dataset(
    {
        "latitude": (["latitude"], np.arange(30, 74, 25.0), lat_attrs),
        "longitude": (["longitude"], np.arange(-80, 45, 25.0), lon_attrs),
    }
)

# Original grid for impact variables
GRID_IMPACTS: xr.Dataset = xr.open_dataset(paths["path_original_grid_impact_variables"])

# Grid for impact variables focused on European region
GRID_IMPACTS_AR6_EU: xr.Dataset = xr.open_dataset(
    paths["path_original_grid_impact_variables"]
).sel(
    latitude=slice(72.6, 30), longitude=slice(-10, 40)
)  # Note: latitudes are stored in decreasing order

# Tiny grid for European region with coarse resolution
GRID_IMPACTS_AR6_EU_TINY: xr.Dataset = xr.Dataset(
    {
        "latitude": (["latitude"], np.arange(30, 72.6, 5.0), lat_attrs),
        "longitude": (["longitude"], np.arange(-10, 40, 5.0), lon_attrs),
    }
)
