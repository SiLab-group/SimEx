from modifier import Modifier
from simulator import Simulator
from validator import Validator


components = {
    'modifier': Modifier.modifierA,
    'simulator': Simulator.sim_func_B,
    'sumo_simulator': Simulator.sumo_simulator,
    'validator': Validator.local_exploration_validator_A
}
