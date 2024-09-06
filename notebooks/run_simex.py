#!/usr/bin/env python3

import os

def run_simex(simulator_function, instance_name):
    # IMPORT LIBRARIES
    import os
    import argparse
    import numpy as np
    # Set instance name
    os.environ['INSTANCE_NAME'] = instance_name

    from global_settings import simexSettings, mds, timestamp, fs
    resultdir = f"results_dir_{instance_name}-{timestamp}"
    # Create directory for the results
    script_dir = os.path.abspath('')
    results_dir = os.path.join(script_dir, resultdir)

    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)

    from components_configuration import components
    from validator_controller import ValidatorController
    from modifier_controller import ModifierController
    from simulator_controller import SimulatorController
    from logger_utils import Logger

    import pickle
    import datetime

    def save_object(obj, filename):
        with open(filename, 'wb') as outp:  # Overwrites any existing file.
            pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)

    validator_controller_vsl = ValidatorController()
    logger = Logger()
    logger_main_arguments = {}
    is_main_func = True
    # Initialize interval list for the first iteration

    intervals_list = [[mds['domain_min_interval'], mds['domain_max_interval']]]
    # Timestamp for the validator pickle file
    count = 0

    while is_main_func:
        # Calls Modifier Controller
        # NOTE: intervals_list type is set to np.int64 due to: https://github.com/numpy/numpy/issues/8433 on windows
        mod_outcome = ModifierController.control(intervals_list=intervals_list,
                                                 selected_modifier=components['modifierA'],
                                                 do_plot=simexSettings['do_plot'])
        mod_x_list = mod_outcome[0]
        checked_intervals = mod_outcome[1]
        print("MAIN mod outcome", mod_outcome)

        # breaks loop if iterations end by granularity reached
        if not mod_x_list:  # FALSE IF ['modifier_data_point'] < mdv['modifier_incremental_unit']:
            logger_main_arguments['log_contex'] = 'overall MAIN stats'
            logger_main_arguments['main_status'] = 'no generated points'
            logger_main_arguments['remaining_unfit_intervals'] = checked_intervals
            logger.log_main(logger_main_arguments)
            break

        # Calls Simulator
        mod_x, sim_y_list = SimulatorController.simulate_parallel(mod_x_list, selected_simulator=simulator_function)
        print(f"MODX {mod_x} and sim_y_list {sim_y_list}")
        assert len(mod_x) == len(sim_y_list)

        print("MAIN modx", mod_x)

        # Calls Validator controller
        intervals_list = validator_controller_vsl.validate(mod_x_list=np.array(mod_x), sim_y_list=np.array(sim_y_list),
                                                           selected_validator=components['validator'],
                                                           global_interval=[mds["domain_min_interval"],
                                                                            mds["domain_max_interval"]])
        print("MAIN interval list from VAL:", intervals_list)
        # Loop number ( Loop-1,Loop-2..etc)
        count += 1
        save_object(validator_controller_vsl, os.path.join(results_dir, f"vc_vsl_loop-{count}-{timestamp}.pkl"))

        # Updates interval_list to new range output from validator controller
        # No more unfit intervals -> write MAIN log
        if not intervals_list:
            is_main_func = False
            logger_main_arguments['log_contex'] = 'overall MAIN stats'
            logger_main_arguments['main_status'] = 'no unfit intervals'
            logger.log_main(logger_main_arguments)

    # MAIN cycle completed/interrupted -> write OVERALL statistics
    logger_main_arguments['log_contex'] = 'Overall Stats'
    logger_main_arguments['main_status'] = 'end cycle'
    logger.log_main(logger_main_arguments)

    # Save data for the last plot located in logger object
    save_object(logger.all_fit_intervals_data,
                os.path.join(results_dir, f"logger-vsl_script-fitted_intervals-{timestamp}.pkl"))
    # If not empty
    if logger.remaining_unfit_intervals:
        save_object(logger.remaining_unfit_intervals,
                    os.path.join(results_dir, f"logger-vsl_script-unfitted_intervals-{timestamp}.pkl"))
    # print(f"Logger object saved with timestamp {timestamp}")
    file = f"{fs['csv_filename']}-{timestamp}.csv"
    file_path = os.path.join(results_dir, file)
    print(f"{file_path}")
    return file_path

import subprocess
import pandas as pd
# import os
# os.environ['INSTANCE_NAME'] = 'LOOP_script'
# from global_settings import simexSettings
# script_dir = os.path.abspath('')
# results_dir = os.path.join(script_dir, f'{simexSettings["results_dir"]}')
#
# if not os.path.isdir(results_dir):
#     os.makedirs(results_dir)

from DLASIUT_find_best_scenarios import automatic_performance
# from components_configuration import components
from simulator import Simulator

# scripts = [ 'sumo_novsl_run.py', 'sumo_vsl_run.py']
# # Run scripts
# def run_script(script) -> str:
#     cmd = ['python3', script[0]]
#     lines = subprocess.run(cmd,stdout=subprocess.PIPE).stdout.splitlines()
#     print(f" LINES: {lines}")
#     baseline_file = None
#     for line in lines:
#         output = "{}".format(line.rstrip().decode("utf-8"))
#         if 'simex_output' in output:
#             baseline_file =  output
#     return baseline_file

# Run baseline NOVSL case

# Rewrite function to accept different types of input data and different simulator function
os.environ['INSTANCE_NAME'] = 'NOVSL_base'
base_file = run_simex(simulator_function=Simulator.sumo_simulator_novsl, instance_name='NOVSL_base_case')
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
        # Run simex for the retrained controller replace components['']
        os.environ['INSTANCE_NAME'] = f'VSL_{counter}'
        control_file = run_simex(simulator_function=Simulator.sumo_simulator_vsl, instance_name='VSL')
        df_control = pd.read_csv(control_file)
        dataset_control = df_control.to_numpy()
        # Run martin script on the csv files
        _incremnet_step_for_x = 10
        _max_order_of_polynom = 9
        _tolerance_in_diffrence=12
        newTrainingData = automatic_performance(dataset_baseline,dataset_control,incremnet_step_for_x=_incremnet_step_for_x,max_order_of_polynom=_max_order_of_polynom,tolerance_in_diffrence=_tolerance_in_diffrence)
        print(newTrainingData)
        if not newTrainingData and counter<MaxIteration:
            break