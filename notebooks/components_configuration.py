from modifier import Modifier
from simulator import Simulator
from validator import Validator


components = {
    'modifier': Modifier.modifierA,
    'simulator': Simulator.sim_func_B,
    'sumo_simulator_vsl': Simulator.sumo_simulator_vsl,
    'sumo_simulator_novsl': Simulator.sumo_simulator_novsl,
    'validator': Validator.local_exploration_validator_A
}
