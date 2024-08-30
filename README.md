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
4. Set the sumo model path and the sumo binary path in the `notebooks/sumo_config.ini`.
```bash
# Path to the model used
# Path for the sumo
[SUMO]
MODEL_PATH = /home/amy/tmp/repos/SimEx/model_MD/
SUMO_PATH = /usr/share/sumo/bin/sumo
```
5. Run `notebooks/SimEx_sumo_vsl_notebook.ipynb` or `notebooks/SimEx_sumo_novsl_notebook.ipynb`
6. For the overall plots adjustments from pkl objects run `notebooks/SimEx_sumo_plot.ipynb`

## Time comparison for parallelization of simulation runs

| Solution                        | VSL runtime (min)   | NOVSL runtime (min)|
|:--------------------------------|:-------------------:|:------------------:|
| ProcessPoolExecutor             |  9.776681780815125  | 11.462244868278503 |
| ThreadPoolExecutor with Process |  9.74341140985489   | 11.33800235191981  |
| No parallelization              | 33.91311665376028   |  36.22358120282491 |