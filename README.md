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

### Basic Usage

```python
import time
from simex import Simex, Simulator, Modifier, Validator

# Create and run simulation
before = time.time()
sim = Simex(instance_name='Func_A', smoothen=False)
file = sim.run_simex(
    simulator_function=Simulator.sim_func_A,
    modifier=Modifier.modifierA,
    validator=Validator.local_exploration_validator_A
)

print(f"Run finished. CSV file is {file}")
print(f"Run time: {(time.time()-before)/60} minutes")
```

### Custom Configuration

```python
from simex import SimexSettings, Simex

# Custom settings
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

# Run with custom settings
file = sim.run_simex(
    simulator_function=Simulator.sim_func_B,
    modifier=Modifier.modifierB,
    validator=Validator.local_exploration_validator_A
)
```

### Available Components

#### Simulators
- `Simulator.sim_func_A`: Cubic function with noise
- `Simulator.sim_func_B`: Sinusoidal function with noise  
- `Simulator.sim_func_C`: Complex function with sine and linear components

#### Modifiers
- `Modifier.modifierA`: Quadratic transformation with rescaling
- `Modifier.modifierB`: Linear transformation with rescaling
- `Modifier.modifierC`: Cubic transformation with rescaling

#### Validators
- `Validator.local_exploration_validator_A`: Polynomial fitting with unfit interval detection

## Command Line Usage

After installation, you can use the command line interface:

```bash
# Run basic example
simex-run

# Run with custom settings
python examples/simex_run.py custom

# Compare parallel vs sequential
python examples/simex_run.py compare
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
│   ├── core/              # Core functionality
│   ├── components/        # Modifiers, simulators, validators
│   ├── controllers/       # Control logic
│   ├── utils/            # Utilities and logging
│   └── config/           # Configuration
├── tests/                # Test suite
├── examples/            # Usage examples
└── docs/               # Documentation
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

The tool generates into the results_dir_NAME_timestamp directore following files:
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

See the `examples/` directory for:
- Basic usage example: simex_run.py
- `notebooks`: SimEx_test_notebook.ipynb

## Support

For issues and questions:
- GitHub Issues: [https://github.com/SiLab-group/SimEx/issues](https://github.com/SiLab-group/SimEx/issues)
- Email: amy.liffey@hevs.ch
