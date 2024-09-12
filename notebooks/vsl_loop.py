#!/usr/bin/env python3

import pandas as pd

from DLASIUT_find_best_scenarios import automatic_performance
from simulator import Simulator
from validator import Validator
from modifier import Modifier
from simex import run_simex

# Run baseline NOVSL case
base_file = run_simex(simulator_function=Simulator.sumo_simulator_novsl, modifier=Modifier.modifierA, validator=Validator.local_exploration_validator_A,instance_name='NOVSL_base_case')
# collect name of the csv file
if base_file:
    df_baseline = pd.read_csv(base_file)
    # Main Function receives np.arrays
    dataset_baseline = df_baseline.to_numpy()
    counter = 0
    MaxIteration = 10
    newTrainingData = []
    while True:
        counter +=1
        # trainingData = trainingData + newTrainingData

        # Insert training of the controller
        # controller.doTraining(trainingData)
        controller = Simulator.sumo_simulator_vsl
        # Run simex for the retrained controller
        control_file = run_simex(simulator_function=controller, modifier=Modifier.modifierA, validator=Validator.local_exploration_validator_A,instance_name='VSL')
        df_control = pd.read_csv(control_file)
        dataset_control = df_control.to_numpy()
        # Run martin script on the csv files
        _incremnet_step_for_x = 10
        _max_order_of_polynom = 9
        _tolerance_in_diffrence=12
        bad_regions = automatic_performance(dataset_baseline,dataset_control,incremnet_step_for_x=_incremnet_step_for_x,max_order_of_polynom=_max_order_of_polynom,tolerance_in_diffrence=_tolerance_in_diffrence)
        print(bad_regions)
        if bad_regions:
            from modifier_controller import ModifierController
            import os
            from modifier import Modifier
            from global_settings import SimexSettings,timestamp
            from logger_utils import Logger

            set = SimexSettings(instance_name="modifier_controller")

            resultdir = f"results_dir_{set.instance_name}-{timestamp}"
            # Create directory for the results
            script_dir = os.path.abspath('')
            results_dir = os.path.join(script_dir, resultdir)
            set.results_dir = results_dir
            if not os.path.isdir(results_dir):
                os.makedirs(results_dir)
                print(f"Results dir {results_dir}")

            log = Logger(filename=os.path.join(results_dir, f"{set.log_filename}-{timestamp}.txt"), simex_settings=set)
            modifier = ModifierController(logger=log, settings=set)
            newTrainingData = modifier.control(intervals_list=bad_regions, selected_modifier=Modifier.modifierA, do_plot=set.do_plot)
            print(newTrainingData)
        if not newTrainingData and counter<MaxIteration:
            break

