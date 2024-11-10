# Template for Isaac Lab Projects

[![IsaacSim](https://img.shields.io/badge/IsaacSim-4.2.0-silver.svg)](https://docs.omniverse.nvidia.com/isaacsim/latest/overview.html)
[![Isaac Lab](https://img.shields.io/badge/IsaacLab-1.2.0-silver)](https://isaac-sim.github.io/IsaacLab)
[![Python](https://img.shields.io/badge/python-3.10-blue.svg)](https://docs.python.org/3/whatsnew/3.10.html)
[![Linux platform](https://img.shields.io/badge/platform-linux--64-orange.svg)](https://releases.ubuntu.com/20.04/)
[![Windows platform](https://img.shields.io/badge/platform-windows--64-orange.svg)](https://www.microsoft.com/en-us/)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://pre-commit.com/)
[![License](https://img.shields.io/badge/license-MIT-yellow.svg)](https://opensource.org/license/mit)

## Overview

Isaac Lab extension for performing a transparent liquid pouring task using RL and image processing through [PourIt](https://github.com/hetolin/PourIt).
Based on the Isaac Lab [extension template](https://github.com/isaac-sim/IsaacLabExtensionTemplate).

**Work In Progress...**


**Keywords:** extension, template, isaaclab, liquid, pouring, pourit

## Installation

- Install Isaac Lab by following the [installation guide](https://isaac-sim.github.io/IsaacLab/source/setup/installation/index.html). We recommend using the conda installation as it simplifies calling Python scripts from the terminal.

- Clone the repository separately from the Isaac Lab installation (i.e. outside the `IsaacLab` directory) and use he access token (this is a private repository for now):

```bash
# Option 1: HTTPS
git clone https://github.com/robegi/IsaacLabPouringExtension.git
```

- To rename from `pouring_ext` to a custom name:

```bash
# Enter the repository
cd IsaacLabExtensionTemplate
# Rename all occurrences of pouring_ext (in files/directories) to your_fancy_extension_name
python scripts/rename_template.py your_fancy_extension_name
```

- Using a python interpreter that has Isaac Lab installed, install the library

```bash
python -m pip install -e exts/pouring_ext
```

Now the extension can be executed from any Isaac Lab script by importing it:

```python
import pouring_ext # Custom extension
```

**Example**: to execute skrl to train the model, use:

```bash
# Enter Isaac Lab folder
cd IsaacLab
# Execute the training script, after importing the extensions
./isaaclab.sh -p source/standalone/workflows/skrl/train.py --task Isaac-Franka-Pouring-Direct-v0 --num_envs 1 --device cpu --disable_fabric --enable_cameras 
```

This uses the default python interpreter. To use a custom one, simply replace ```./isaaclab.sh -p``` with ```python``` inside a conda environment.


