#!/usr/bin/env python3

from simulator import Simulator
from validator import Validator
from modifier import Modifier
from simex import run_simex

if __name__ == '__main__':
    # Run simex
    print("Running simex.")
    base_file = run_simex(simulator_function=Simulator.sumo_simulator_novsl, modifier=Modifier.modifierA, validator=Validator.local_exploration_validator_A,instance_name='NOVSL_script')
    print(f"Run finished. CSV file is {base_file}")