# SimEx

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
3. Run jupyter-lab in the envirnoment
4. Adjust path for the sumo and the models in `notebooks/global_settings.py` sumovsls variables.
5. Run `notebooks/SimEx_sumo_vsl_notebook.ipynb` or `notebooks/SimEx_sumo_novsl_notebook.ipynb`
6. For the overal plots adjustments from pkl objects run `notebooks/SimEx_sumo_plot.ipynb`
