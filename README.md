# SimEx - Systematic Exploration Tool
A Python package for systematic exploration of simulation models. This tool was used in the evaluation of traffic controller use cases and provides a framework for systematic parameter space exploration with adaptive sampling.

<img width="3330" height="878" alt="logic_schema-1" src="https://github.com/user-attachments/assets/83fa8bf7-05c5-4074-9a49-df960e6351d6" />

## Reference

K. Kušić et al., "Evaluation of Traffic Controller Performance via Systematic Exploration," 2024 International Symposium ELMAR, Zadar, Croatia, 2024, pp. 165-168, doi: [10.1109/ELMAR62909.2024.10694499](https://ieeexplore.ieee.org/document/10694499).

## Installation

[//]: # (### From PyPI &#40;when published&#41;)

[//]: # (```bash)

[//]: # (pip install simex)

[//]: # (```)

### From Source
```bash
git clone https://github.com/SiLab-group/SimEx.git
cd SimEx

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install in development mode
pip install -e .

# Or run the setup script
chmod +x setup_venv.sh
./setup_venv.sh
```

### For Development
```bash
# Install with development dependencies
pip install -e ".[dev]"

# Or install from requirements
pip install -r requirements-dev.txt
```

## Quick Start

```python
from simex import Simex
from simex.components.simulator import Simulator
from simex.components.modifier import Modifier
from simex.components.validator import Validator

sim = Simex(instance_name='Func_A', smoothen=False)
file = sim.run_simex(
    simulator_function=Simulator.sim_func_A,
    modifier=Modifier.modifierA,
    validator=Validator.local_exploration_validator_A
)
print(f"Run finished. CSV file is {file}")
```

### Custom Settings

```python
from simex import Simex
from simex.config.settings import SimexSettings

settings = SimexSettings(
    instance_name='custom_run',
    domain_min_interval=1000,
    domain_max_interval=5000,
    modifier_incremental_unit=10,
    vfs_threshold_y_fitting=20,
    ops_sigmoid_tailing=True
)
sim = Simex(instance_name='custom_run', smoothen=True)
sim.settings = settings
```

### Built-in Components

| Type | Name | Description |
|------|------|-------------|
| Simulator | `sim_func_A` | Cubic function with noise |
| Simulator | `sim_func_B` | Sinusoidal function with noise |
| Simulator | `sim_func_C` | Sine + linear function with noise |
| Modifier | `modifierA` | Quadratic transformation with rescaling |
| Modifier | `modifierB` | Linear transformation with rescaling |
| Modifier | `modifierC` | Cubic transformation with rescaling |
| Validator | `local_exploration_validator_A` | Polynomial fitting with unfit interval detection |

## Command Line Usage

```bash
simex-run                              # basic run
python examples/simex_run.py custom    # custom settings
python examples/simex_run.py compare   # parallel vs sequential
```

## Development

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=simex

# Run specific test file
pytest tests/test_simex.py -v
```

### Code Formatting

```bash
# Format code with black
black simex/ tests/ examples/

# Check code style
flake8 simex/ tests/ examples/
```

### Package Structure

```
simex/
├── simex/                  # Main package
│   ├── core/              # Core SimEx loop
│   ├── components/        # Simulators, modifiers, validators
│   ├── controllers/       # Modifier, simulator, validator controllers
│   ├── utils/             # Logging
│   └── config/            # Settings
├── examples/              # Usage examples
│   ├── simex_run.py       # Basic example
│   └── marl_vsl/          # SUMO/MARL VSL examples
└── tests/
```

## Configuration

### Key Settings

- `domain_min_interval` / `domain_max_interval`: Exploration domain bounds
- `modifier_incremental_unit`: Minimum step size for exploration
- `modifier_data_point`: Initial step size for point generation
- `vfs_threshold_y_fitting`: Y-axis threshold for curve fitting
- `vfs_degree` / `vfs_max_deg`: Polynomial fitting degree range
- `max_workers`: Number of parallel processes for simulation

### Logging and Output

Results are written to `results_dir_NAME_timestamp/` and include:
- **CSV files**: Final results with fitted polynomial coefficients
- **PDF plots**: Visualization of fitted curves and unfit intervals
- **Log files**: Detailed execution logs
- **Pickle files**: Intermediate results for analysis

## Performance Comparison

The tool supports both sequential and parallel execution:

| Solution                 | Runtime Improvement |
|:-------------------------|:-------------------:|
| ProcessPoolExecutor      | ~3.5x faster        |
| Sequential (baseline)    | 1x                  |

## Examples

### Basic usage (`examples/simex_run.py`)
Runs SimEx with built-in mock simulators (`sim_func_A/B`) — no external dependencies required. Good starting point to understand the SimEx loop.

```bash
python examples/simex_run.py           # basic run
python examples/simex_run.py custom    # custom settings
python examples/simex_run.py compare   # parallel vs sequential
```

### Notebook (`examples/notebooks/SimEx_test_notebook.ipynb`)
Interactive walkthrough of the SimEx API.

### MARL VSL examples (`examples/marl_vsl/`)
Traffic controller examples using SUMO via TraCI. Requires a `sumo_config.ini` copied and filled in from `sumo_config_example.ini`.

**Simple VSL** (`run_vsl_simple.py`): single SimEx run with a rule-based proportional VSL controller. No training required.
```bash
cd examples/marl_vsl
python run_vsl_simple.py
```

**MARL VSL loop** (`run_vsl_loop.py`): iterative SimEx loop that trains MARL agents on demand regions where the VSL controller underperforms a no-VSL baseline. Pre-trained weights are provided in `marl_training_file/`.
```bash
cd examples/marl_vsl
python run_vsl_loop.py
```

**Notebook** (`SimEx_loop.ipynb`): interactive version of the MARL VSL loop with a standalone cell for analysing bad regions from an existing VSL output CSV.

#### MARL VSL setup
1. Copy the example config: `cp sumo_config_example.ini sumo_config.ini`
2. Edit `sumo_config.ini` with your local SUMO binary and model paths
3. Set `SUMO_HOME` to your SUMO installation (e.g. `export SUMO_HOME=/path/to/sumo`)

## Support

For issues and questions:
- GitHub Issues: [https://github.com/SiLab-group/SimEx/issues](https://github.com/SiLab-group/SimEx/issues)
- Email: amy.liffey@hevs.ch
