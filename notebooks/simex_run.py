#!/usr/bin/python3

from simulator import Simulator
from validator import Validator
from modifier import Modifier
from simex import Simex
import time
before = time.time()
# Run simex
print("Running simex.")
sim = Simex(instance_name='Func_A', smoothen=False)
file = sim.run_simex(simulator_function=Simulator.sim_func_A, modifier=Modifier.modifierA,
                     validator=Validator.local_exploration_validator_A)
print(f"Run finished. CSV file is {file}")
now = time.time()
print(f"Run time: {(now-before)/60}")
