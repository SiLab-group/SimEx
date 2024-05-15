# SimEx

## Install and run jupyter notbook
1. Create venv/conda environment and install the dependencies
```bash
# Creates environment in the .venv directory
python3 -m venv .venv
# Activate the environment
source .venv/bin/activate
# Install dependencies
pip3 install -r requirementsAMY.txt

# Conda env and install requirements
conda create -n simex-env python=3.10
conda install -n simex-env requirementsAMY.txt
```
2. Run jupyter-lab in the envirnoment
3. Adjust path for the sumo and the models in `notebooks/global_settings.py` sumovsls variables.
4. Run `notebooks/SimEx_SUMOvsl_notebook.ipynb`