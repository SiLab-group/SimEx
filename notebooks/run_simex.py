import os
import numpy as np
from global_settings import SimexSettings, Mds, timestamp, Fs


# class SimexRunner():
#     def __init__(self, name):
#         self.instance_name = name

def run_simex(simulator_function, instance_name):
    SimexSettings.instance_name=instance_name
    print(f"Instance name {SimexSettings.instance_name}")
    resultdir = f"results_dir_{SimexSettings.instance_name}-{timestamp}"
    # Create directory for the results
    script_dir = os.path.abspath('')
    results_dir = os.path.join(script_dir, resultdir)

    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)

    from logger_utils import Logger
    from validator_controller import ValidatorController
    from modifier_controller import ModifierController
    from simulator_controller import SimulatorController

    import pickle
    import datetime

    def save_object(obj, filename):
        with open(filename, 'wb') as outp:  # Overwrites any existing file.
            pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)

    logger = Logger()
    validator_controller = ValidatorController(logger)
    modifier_controller = ModifierController(logger)
    # logger = Logger()
    logger_main_arguments = {}
    is_main_func = True
    # Initialize interval list for the first iteration


    intervals_list = [[Mds.domain_min_interval, Mds.domain_max_interval]]
    # Timestamp for the validator pickle file
    count = 0

    while is_main_func:
        from components_configuration import components
        # Calls Modifier Controller
        # NOTE: intervals_list type is set to np.int64 due to: https://github.com/numpy/numpy/issues/8433 on windows
        mod_outcome = modifier_controller.control(intervals_list=intervals_list, selected_modifier=components['ModifierA'],
                                                 do_plot=SimexSettings.do_plot)
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
        intervals_list = validator_controller.validate(mod_x_list=np.array(mod_x), sim_y_list=np.array(sim_y_list),
                                                           selected_validator=components['validator'],
                                                           global_interval=[Mds.domain_min_interval,
                                                                            Mds.domain_max_interval])
        print("MAIN interval list from VAL:", intervals_list)
        # Loop number ( Loop-1,Loop-2..etc)
        count += 1
        save_object(validator_controller, os.path.join(results_dir, f"vc_vsl_loop-{count}-{timestamp}.pkl"))

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
    save_object(logger.all_fit_intervals_data
                ,os.path.join(results_dir ,f"logger-vsl_script-fitted_intervals-{timestamp}.pkl"))
    # If not empty
    if logger.remaining_unfit_intervals:
        save_object(logger.remaining_unfit_intervals
                    ,os.path.join(results_dir ,f"logger-vsl_script-unfitted_intervals-{timestamp}.pkl"))
    # print(f"Logger object saved with timestamp {timestamp}")
    file = f"{Fs.csv_filename}-{timestamp}.csv"
    file_path = os.path.join(results_dir ,file)
    print(f"{file_path}")
    return file_path
