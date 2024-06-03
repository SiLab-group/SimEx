from modifier import Modifier
from simulator import Simulator
from validator import Validator


components = {
    'modifierA': Modifier.modifierA,
    'modifierB': Modifier.modifierB,
    'simulator': Simulator.sim_func_B,
    'sumo_simulator_vsl': Simulator.sumo_simulator_vsl,
    'sumo_simulator_novsl': Simulator.sumo_simulator_novsl,
    'sumo_simulator_vsl_old': Simulator.sumo_simulator_vsl_old,
    'sumo_simulator_novsl_old': Simulator.sumo_simulator_novsl_old,
    'validator': Validator.local_exploration_validator_A
}
