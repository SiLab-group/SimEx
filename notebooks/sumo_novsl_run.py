#!/usr/bin/env python3

from simulator import Simulator
from validator import Validator
from modifier import Modifier
from simex import Simex

if __name__ == '__main__':
    # Run simex
    print("Running simex.")
    simex_novsl_old = Simex(instance_name='NOVSL_old_script', smoothen=False)
    base_file_novsl = simex_novsl_old.run_simex(simulator_function=Simulator.sumo_simulator_novsl,
                                                modifier=Modifier.modifierA,
                                                validator=Validator.local_exploration_validator_A)
    print(f"Run finished. CSV file is {base_file_novsl}")