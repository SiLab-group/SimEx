from modifier import Modifier
from simulator import Simulator
from validator import Validator


components = {
    'modifierA': Modifier.modifierA,
    'modifierB': Modifier.modifierB,
    'simulator': Simulator.sim_func_B,
    'sumo_simulator_vsl': Simulator.sumo_simulator_vsl,
    'sumo_simulator_novsl': Simulator.sumo_simulator_novsl,
    'sumo_simulator': Simulator.sumo_simulator,
    'validator': Validator.local_exploration_validator_A
}
