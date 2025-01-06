# SimEx
This repository contains Systematic exploration tool.

## Run Simex-sumo simulation
To run the simulation the jupyter notebook or normal python script can be used. Follow the instructions in the 
notebook to setup the settings for the sumo and model path.
1. VSL: `notebooks/SimEx_sumo_vsl_notebook.ipynb` or `notebooks/sumo_vsl_run.py`
2. NOVSL: `notebooks/SimEx_sumo_novsl_notebook.ipynb` or `notebooks/sumo_novsl_run.py`
3. Loop for training controller can be run in `notebooks/SimEx_loop.ipynb` or `notebooks/vsl_loop.py`.

## Install and run jupyter notebook
All the sumo related functions were tested and run with sumo 1.21.0 be aware that different versions of sumo can 
give different results.
1. Create venv environment and install the dependencies Linux:
```bash
# Creates environment in the .venv directory
python3 -m venv .venv
# Activate the environment
source .venv/bin/activate
# Install dependencies
pip3 install -r requirements.txt
```
2. Create venv environment and install the dependencies Windows. Tested only with VSCODE:
   - Run vscode
   - Open the Notebook to run: Select kernel -> Create new environment .venv -> Select requriements.txt -> Run the jupyter notebook
3. Run jupyter-lab in the environment
4. Set the sumo model path and the sumo binary path in the `notebooks/sumo_config.ini`.
```bash
# Path to the model used
# Path for the sumo
[SUMO]
# taken root path of directory + MODEL_PATH
MODEL_PATH = /model_MD/
SUMO_PATH = /usr/share/sumo/bin/sumo
```
5. Setting of the default parameters in the `notebooks/global_settings.py` and per instance when calling `run_simex` function located in `notebooks/simex.py` module.
```python
from simulator import Simulator
from validator import Validator
from modifier import Modifier
from simex import Simex
# All default parameters can be overriden here when calling the run_simex function
simex_novsl_old = Simex(instance_name='NOVSL_old_notebook', smoothen=False)
base_file_novsl_not = simex_novsl_old.run_simex(simulator_function=Simulator.sumo_simulator_novsl,
                                                modifier=Modifier.modifierA,
                                                validator=Validator.local_exploration_validator_A, parallel=True)
print(f"Run finished. CSV file is {base_file_novsl_not}")
```
6. Run `notebooks/SimEx_sumo_vsl_notebook.ipynb` or `notebooks/SimEx_sumo_novsl_notebook.ipynb`
7. The results of each simulation run are saved into the results directory, which name is defined in `notebooks/global_settings.py`.



## Time comparison for parallelization of simulation runs

| Solution                        | VSL runtime (min)   | NOVSL runtime (min)|
|:--------------------------------|:-------------------:|:------------------:|
| ProcessPoolExecutor             |  9.776681780815125  | 11.462244868278503 |
| ThreadPoolExecutor with Process |  9.74341140985489   | 11.33800235191981  |
| No parallelization              | 33.91311665376028   |  36.22358120282491 |