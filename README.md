# SimEx
This repository contains Systematic exploration tool. This tool was used in the Evaluation of the traffic controller use case. This usecase can be found in the branch [traffic_simulator_case](https://github.com/SiLab-group/SimEx/tree/traffic_simulator_case) [1].

[1] K. Kušić et al., "Evaluation of Traffic Controller Performance via Systematic Exploration," 2024 International Symposium ELMAR, Zadar, Croatia, 2024, pp. 165-168, doi: [10.1109/ELMAR62909.2024.10694499](https://ieeexplore.ieee.org/document/10694499).


## TODO
Make Simex Library with proper directory structure and not notebooks. Use the notebooks only as an example.

## Run Simex-sumo simulation
To run the simulation the jupyter notebook or normal python script can be used. Follow the instructions in the 
notebook to setup the settings for the sumo and model path.
1. Test notebook: `notebooks/SimEx_test_notebook.ipynb`

## Install and run jupyter notebook
1. Create venv environment and install the dependencies Linux:
```bash
# Creates environment in the .venv directory
python3 -m venv .venv
# Activate the environment
source .venv/bin/activate
# Install dependencies
pip3 install -r requirements.txt
```
1.Create venv environment and install the dependencies Windows. Tested only with VSCODE:
   - Run vscode
   - Open the Notebook to run: Select kernel -> Create new environment .venv -> Select requirements.txt -> Run the jupyter notebook
3. Run jupyter-lab in the environment
5. Setting of the default parameters in the `notebooks/global_settings.py` and per instance when calling `run_simex` function located in `notebooks/simex.py` module.
```python
from simulator import Simulator
from validator import Validator
from modifier import Modifier
from simex import Simex
# All default parameters can be overriden here when calling the run_simex function
sim= Simex(instance_name='Test', smoothen=False)
base_file = sim.run_simex(simulator_function=Simulator.sim_func_A,
                                                modifier=Modifier.modifierA,
                                                validator=Validator.local_exploration_validator_A, parallel=True)
print(f"Run finished. CSV file is {base_file}")
```
The file `simex.py` contains instance of simex which can be copied out to run modifier, simulator, validator decoupled without whole simex instance.
6. Run `notebooks/SimEx_test_notebook.ipynb` or `simex_run.py`
7. The results of each simulation run are saved into the results directory, which name is defined in `notebooks/global_settings.py`.



## Time comparison for parallelization of simulation runs

| Solution                        | VSL runtime (min)   | NOVSL runtime (min)|
|:--------------------------------|:-------------------:|:------------------:|
| ProcessPoolExecutor             |  9.776681780815125  | 11.462244868278503 |
| ThreadPoolExecutor with Process |  9.74341140985489   | 11.33800235191981  |
| No parallelization              | 33.91311665376028   |  36.22358120282491 |
