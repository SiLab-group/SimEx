#!/usr/bin/env python3
"""
MARL VSL example: iterative SimEx loop with MARL-based VSL controller.

Workflow:
1. Load a pre-computed baseline (no-VSL) SimEx result CSV.
2. Run SimEx with the MARL VSL simulator to evaluate control performance.
3. Identify "bad regions" where VSL underperforms the baseline.
4. Retrain agents on the bad regions and repeat.

Configure paths via sumo_config.ini (see sumo_config_example.ini).
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, os.path.dirname(__file__))
import pandas as pd
import numpy as np

from simex import Simex
from simex.components.validator import Validator
from simex.components.modifier import Modifier
from simex.config.settings import SumoVsl
from marl_simulator import Simulator
from marl.training import Controller
from marl.performance import automatic_performance


if __name__ == '__main__':
    print("Running MARL VSL SimEx loop.")

    results_path = os.getenv('MARL_RESULTS_PATH', SumoVsl.marl_results_path or '/tmp/marl_training/')
    iterations_of_training = 3000
    os.environ['MARL_PATH_TRAIN'] = results_path

    # Load baseline (no-VSL) SimEx output — run once offline or point to existing file
    path = os.path.dirname(os.path.abspath(__file__))
    base_file = os.getenv(
        'MARL_BASE_FILE',
        os.path.join(path, 'base_file', 'simex_output-MARL_novsl-20250102-201414.csv')
    )

    df_baseline = pd.read_csv(base_file)
    print(f"Base file loaded: {df_baseline.head()}")
    dataset_baseline = df_baseline.to_numpy()

    counter = 0
    MaxIteration = 5
    newTrainingData = []
    trainingData = [i for i in range(2500, 4100, 100)]
    load_trained_controller = True
    start_training = 6000

    while True:
        counter += 1

        # Train controller on current training data
        if counter > 1 and not load_trained_controller:
            trainingData = trainingData + newTrainingData
            controller = Controller(results_path)
            controller.marl_vsl_training(trainingData)
        elif counter > 1 and load_trained_controller:
            print(f"============= New training data {newTrainingData}")
            trainingData = newTrainingData
            controller = Controller(results_path)
            last_run = controller.marl_vsl_training(
                trainingData,
                run_start=start_training,
                iterations=iterations_of_training,
                vsl_on=1
            )
            print(f"Training ended at run {last_run}")
            os.environ['END_RUN_MARL'] = str(start_training + iterations_of_training)
            start_training = start_training + iterations_of_training

        # Run SimEx with the (re)trained VSL controller
        simex_loop_vsl = Simex(instance_name=f"VSL_marl_loop-{counter}", smoothen=True)
        control_file = simex_loop_vsl.run_simex(
            simulator_function=Simulator.marl_vsl_simulator,
            modifier=Modifier.modifierA,
            validator=Validator.local_exploration_validator_A,
            parallel=False
        )

        df_control = pd.read_csv(control_file)
        dataset_control = df_control.to_numpy()

        # Find traffic demand regions where VSL does not improve over baseline
        bad_regions = automatic_performance(
            dataset_baseline,
            dataset_control,
            incremnet_step_for_x=10,
            max_order_of_polynom=9,
            tolerance_in_diffrence=12
        )
        print("Bad regions:", bad_regions)

        if bad_regions:
            newTrainingData = [item for i, j in bad_regions for item in np.arange(i, j, 5)]
            print("New training data:", newTrainingData)

        if not newTrainingData or counter > MaxIteration:
            break

    os.environ['END_RUN_MARL'] = str(6000)
    print("Done.")
