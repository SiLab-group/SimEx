#!/usr/bin/env python3
"""
Simple VSL example: single SimEx run with a rule-based VSL controller.

Uses a proportional VSL controller (no MARL) to set speed limits based on
traffic density. The simulator sweeps traffic demand values and returns TTS.

Configure paths via sumo_config.ini (see sumo_config_example.ini).
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, os.path.dirname(__file__))

from simex import Simex
from simex.components.validator import Validator
from simex.components.modifier import Modifier
from marl_simulator import Simulator


if __name__ == '__main__':
    print("Running simple VSL SimEx.")

    sim = Simex(instance_name="simple_vsl", smoothen=True)
    result_file = sim.run_simex(
        simulator_function=Simulator.sumo_simulator_vsl,
        modifier=Modifier.modifierA,
        validator=Validator.local_exploration_validator_A,
        parallel=False
    )

    print(f"Done. Results: {result_file}")
