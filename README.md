# SimEx
This repository contains Structured simulation framework.

## Run Simex-sumo simulation
To run the simulation the jupyter notebook or normal python script can be used. Follow the instructions in the 
notebook to setup the settings for the sumo and model path.
1. VSL: `notebooks/SimEx_sumo_vsl_notebook.ipynb` or `notebooks/sumo_vsl_run.py`
2. NOVSL: `notebooks/SimEx_sumo_novsl_notebook.ipynb` or `notebooks/sumo_novsl_run.py`
3. Plots from the saved pickle objects: `notebooks/SimEX_sumo_plot.ipynb`

## Install and run jupyter notebook
1. Create venv environment and install the dependencies Linux:
```bash
# Creates environment in the .venv directory
python3 -m venv .venv
# Activate the environment
source .venv/bin/activate
# Install dependencies
pip3 install -r requirementsAMY.txt
```
2. Create venv environment and install the dependencies Windows. Tested only with VSCODE:
   - Run vscode
   - Open the Notebook to run: Select kernel -> Create new environment .venv -> Select requriementsAmy.txt -> Run the jupyter notebook
3. Run jupyter-lab in the environment
4. Adjust path for the sumo and the models in `notebooks/global_settings.py` sumovsls variables.
```python
# Path to the model used
# Path for the sumo
sumovsls = {"model_path": "C:/Users/kusic/Desktop/SSF/SUMOVSL/SPSC_MD/model_MD/",
           "sumo_path": "C:/Program Files (x86)/Eclipse/Sumo/bin/sumo"}
```
5. Run `notebooks/SimEx_sumo_vsl_notebook.ipynb` or `notebooks/SimEx_sumo_novsl_notebook.ipynb`
6. For the overal plots adjustments from pkl objects run `notebooks/SimEx_sumo_plot.ipynb`
