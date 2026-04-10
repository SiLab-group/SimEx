"""Generate sumo_config.ini for CI using environment variables."""
import os
import subprocess
import sys

workspace = os.environ.get("GITHUB_WORKSPACE")
if not workspace:
    print("ERROR: GITHUB_WORKSPACE environment variable is not set.", file=sys.stderr)
    sys.exit(1)

result = subprocess.run(
    ["python", "-c", "import sumo, os; print(os.path.join(os.path.dirname(sumo.__file__), 'bin', 'sumo'))"],
    capture_output=True, text=True
)
sumo_bin = result.stdout.strip()
if not sumo_bin or result.returncode != 0:
    print("ERROR: Could not locate sumo binary. Is eclipse-sumo installed?", file=sys.stderr)
    sys.exit(1)

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
