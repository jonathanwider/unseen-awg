# Installation

Conda is used to install the required dependencies because at the time of writing the installation of the required xESMF package through pip wasn't straightforward.

To install and activate the conda environment, navigate to the package's base directory and run:

```Bash
conda env create -f env.yml
conda activate unseen-awg
pip install -e .  # Install the unseen-awg package in development mode
```

The `unseen_awg` source code is installed in editable mode.

Alternatively, we also provide the environment as conda-lock.yml, created with `conda-lock`. To install it:

```Bash
conda install -c conda-forge conda-lock
conda-lock install -n unseen-awg conda-lock.yml
conda activate unseen-awg
pip install -e .
```

## Adapting predefined paths
Base directories for different components of the unseen-awg workflow are specified in `configs/`. Use the provided `configs/paths_template.yaml` to create a `configs/paths.yaml` containing your own paths.