# Overall SimEx settings
# possible modes exploration (this one) and exploitation (mod with prob threes)
import os

simexSettings = {"do_plot": False,  # No special meaning at the moment. TODO: Should be refactored.
                 "extensive_search": False,  #  Complete exploration setting modifier_data_point to 1 and enabling extensive iteration
                 "extensive_iteration": False,  #  Gets enabled when extensive search is True. TODO: should be refactored
                 "SimEx_mode": "exploration"}  # Only exploration implemented

# Modifier Domain Settings
mds = {"domain_min_interval": 2500,
        "domain_max_interval": 4000,
        "modifier_incremental_unit": 25,  # Minimal incremental unit is the smallest allowed step_size. Note: If extensive search True then minimal increment is set to 1
        "modifier_data_point": 100,  # Data point step size on the X axis in the first round. In next iterations
                                   # modifier_data_point = modifier_data_point - modifier_incremental_unit until modifier_data_point < minimal_increment_unit.
       }

# Validator Function Settings
# For each fitted function we calculate Mean squared error(MSE):
# https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_error.html
# MSE ( (y_values, current_y_pred) + penality_weight * np.sum(current_coeff[:-1] ** 2) ) and compare it to the previous
# We consider improvement acceptable if: (previous_mse - current_mse) >= improvement_threshold
vfs = {'threshold_y_fitting': 15,  # Threshold on the y axis
       'threshold_x_interval': 0.80,  # For unfit point expand by threshold_x_interval to each side to close unfit interval
       'degree': 2,  # Minimum degree for exploration. We start with polyfit in x^degree
       'max_deg': 9,  # Max degree for exploration to which degree we try to fit function x^max_degree
       'early_stop': True,  # if early_stop = True and improvement is not acceptable by increasing dimension, we stop
       'improvement_threshold': 0.1,  # Sufficient improvement threshold (previous_mse - current_mse) >= improvement_threshold
       'penality_weight': 1, # Penalty for MSE to avoid overfitting with high dimension polynomial
       'x_labels': 'Traffic volume [veh/h]',  # Y axis label name validator graph
       'y_labels': 'TTS [veh$\cdot$h]',  # Y axis label name validator graph
       'title': f'Fitted Curve with unfit Intervals for {os.environ["INSTANCE_NAME"]}',  # Title for validator graph
       'figsize_x': 12,  # X size of the figure
       'figsize_y': 6,  # Y size of the figure
       'font_size': 12  # Fontsize in the figure
       }

# Overall plot settings (the last plot with all the functions)
ops = {
    'x_labels': 'Traffic volume [veh/h]',
    'y_labels': 'TTS [veh$\cdot$h]',
    'title': f'Total fitted curves for {os.environ["INSTANCE_NAME"]} case',
    'figsize_x': 10,  # X size of the figure
    'figsize_y': 5,   # Y size of the figure
    'linewidth': 3  # Linewidth for the functions plotted
    }

## Data and settings for log purposes ##
# These settings are filled during the runtime and used as a global data structure for the logger statistics.
# Modifier Global Statistics 
mgs = {"points_generated_total": 0, # Number of generated points TODO: Should be refactored
       "points_generation_intervals": 0, # Number of intervals generated TODO: Should be refactored
       "mod_iterations": 0}  # Number of modifier iterations TODO: Should be refactored

# Validator Global statistics
vgs = {"points_fitting_total": 0,  # Not used TODO: Should be refactored
       "points_unfitting_total": 0,  # Not used TODO: Should be refactored
       "intervals_unfit_total": 0}  # Not used TODO: Should be refactored

# Filename settings
fs = {
    "log_filename": f"LOG-{os.environ['INSTANCE_NAME']}",
    "csv_filename": f"simex_output-{os.environ['INSTANCE_NAME']}"
    }
# Logger Granularity Settings
# log_granularity:
# 0 only general stats
# 1 minimal log
# 2 medium
# 3 detailed)
lgs = {"log_granularity": 3}

# SUMOvsl settings
sumovsls = {"model_path": "C:/Users/kusic/Desktop/SSF/SUMOVSL/SPSC_MD/model_MD/",
            "sumo_path": "C:/Program Files (x86)/Eclipse/Sumo/bin/sumo"}
# sumovsls = {"model_path": "/home/amy/tmp/repos/SimEx/model_MD/",
#             "sumo_path": "/usr/share/sumo/bin/sumo"}