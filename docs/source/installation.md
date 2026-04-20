# Installation

Conda is used to manage and install required dependencies.

To install and activate the provided conda environment, navigate *unseen-awg*'s base directory and run:

```Bash
conda env create -f env.yml
conda activate unseen-awg
pip install -e .  # Install the unseen-awg package in development mode
```

Alternatively, we also provide the environment as `conda-lock.yml`, created with `conda-lock`. To install it:
```Bash
conda install -c conda-forge conda-lock
conda-lock install -n unseen-awg conda-lock.yml
conda activate unseen-awg
pip install -e .
```
Both methods should yield similar results but they may differ in the precise versions of packages being installed. In both cases, the *unseen-awg* source code is installed in editable mode.

(configs-paths)=
## Defining paths in `configs/paths.yaml`
A file `configs/paths.yaml` is used as a central location for storing paths to base directories for different components of the *unseen-awg* workflow. If you want to make use of this workflow, use the provided `configs/paths_template.yaml` to create a `configs/paths.yaml` containing paths to directories of your choice.