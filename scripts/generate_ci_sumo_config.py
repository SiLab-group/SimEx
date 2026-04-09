"""Generate sumo_config.ini for CI using environment variables."""
import os
import subprocess

workspace = os.environ["GITHUB_WORKSPACE"]

result = subprocess.run(
    ["python", "-c", "import sumo, os; print(os.path.join(os.path.dirname(sumo.__file__), 'bin', 'sumo'))"],
    capture_output=True, text=True
)
sumo_bin = result.stdout.strip()

config = f"""[SUMO]
MODEL_PATH = {workspace}/model_MD/
SUMO_PATH = {sumo_bin}

[MARL]
ROOT_PATH = {workspace}/
RESULTS_PATH = {workspace}/marl_training_file/
MODEL_PATH = {workspace}/marl_model_MD/
RUN = 0
END = 6000
VSL = 1
"""

for dest in ["sumo_config.ini", "examples/marl_vsl/sumo_config.ini"]:
    with open(dest, "w") as f:
        f.write(config)
    print(f"Written {dest}")
