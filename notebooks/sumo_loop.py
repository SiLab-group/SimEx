import subprocess
import pandas as pd
import os, pickle, datetime

os.environ['INSTANCE_NAME'] = 'LOOP_script'
from global_settings import simexSettings

script_dir = os.path.abspath('')
results_dir = os.path.join(script_dir, f'{simexSettings["results_dir"]}')

if not os.path.isdir(results_dir):
    os.makedirs(results_dir)


def save_object(obj, filename):
    with open(filename, 'wb') as outp:  # Overwrites any existing file.
        pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)


from DLASIUT_find_best_scenarios import automatic_performance

scripts = ['sumo_vsl_run.py', 'sumo_novsl_run.py']
loop = True
mod_outcome = ()
while loop:
    files = []
    # RUn scripts
    for script in scripts:
        print(f"Scripts running {script}")
        if mod_outcome:
            # Pass the mod_outcome pkl name file to the subprocess
            filename = f"mod_{os.environ['INSTANCE_NAME']}_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}.pkl"
            final_filename = os.path.join(simexSettings['results_dir'], filename)
            save_object(mod_outcome, final_filename)
            print(f"Pickle saved {final_filename}")
            cmd = ['python3', script, '--intervals_pkl', final_filename]
        else:
            cmd = ['python3', script]
        lines = subprocess.run(cmd, stdout=subprocess.PIPE).stdout.splitlines()
        print(f" LINES: {lines}")
        for line in lines:
            output = "{}".format(line.rstrip().decode("utf-8"))
            if 'simex_output' in output:
                print(f"Output file append: {output}")
                files.append(output)

    # collect name of the csv file
    if len(files) == 2:
        df_baseline = pd.read_csv(files[1])
        df_control = pd.read_csv(files[0])
        # Main Function receives np.arrays
        dataset_baseline = df_baseline.to_numpy()
        dataset_control = df_control.to_numpy()

        # Run martin script on the csv files
        _incremnet_step_for_x = 10
        _max_order_of_polynom = 9
        _tolerance_in_diffrence = 12
        results = automatic_performance(dataset_baseline, dataset_control, incremnet_step_for_x=_incremnet_step_for_x,
                                        max_order_of_polynom=_max_order_of_polynom,
                                        tolerance_in_diffrence=_tolerance_in_diffrence)
        print(f"Results: {results}")

    else:
        print(f" File lenght: {len(files)} for {files}  \n BREAKING")
        results = []
        # break

    if results:
        # get moddifiers ?? providing the initial x values to the script???
        from modifier_controller import ModifierController
        from components_configuration import components
        print(f"Entering modifiers with {results}")
        mod_outcome = ModifierController.control(intervals_list=results, selected_modifier=components['modifierA'],
                                                 do_plot=simexSettings['do_plot'])
        print(f"MOD OUTCOME IS {mod_outcome}")
    else:

        loop = False
        print(f"LOOP is over?? {loop}")