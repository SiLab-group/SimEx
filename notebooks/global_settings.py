# Overall SimEx settings
# possible modes exploration (this one) and exploitation (mod with prob threes)
simexSettings = {"do_plot": False,  # No special meaning at the moment. TODO: Should be refactored.
                 "extensive_search": False,  #  Complete exploration setting modifier_data_point to 1 and enabling extensive iteration
                 "extensive_iteration": False,  #  Gets enabled when extensive search is True. TODO: should be refactored
                 "SimEx_mode": "exploration"}  # Only exploration implemented

# Modifier Domain Settings
mds = {"domain_min_interval": 2500,
       "domain_max_interval": 4000,
       "modifier_incremental_unit": 25,  # Second round modifier_data_point - modifier_incremental_unit to make smaller granularity. Note: If extensive search True then increment set to 1
       "modifier_data_point": 100  # Data points on the X axis in the first round. Step size (100,200,300,..)
       }

# Validator Function Settings
# When fitting function we calculate MSE for each fitted function:
# MSE ( (y_values, current_y_pred) + penality_weight * np.sum(current_coeff[:-1] ** 2) ) and compare it to the previous
# We consider improvement acceptable if: (previous_mse - current_mse) >= improvement_threshold
vfs = {'threshold_y_fitting': 15,  # Threshold on the y axis
       'threshold_x_interval': 0.80,  # For unfit point expand by threshold_x_interval to each side to close unfit interval
       'degree': 2,  # Minimum degree for exploration. We start with polyfit in x^degree
       'max_deg': 9,  # Max degree for exploration to which degree we try to fit function x^max_degree
       'early_stop': True,  # if early_stop = True and improvement is not acceptable by increasing dimension, we stop
       'improvement_threshold': 0.1,  # Sufficient improvement threshold (previous_mse - current_mse) >= improvement_threshold
       'penality_weight': 1}  # Penalty for MSE to avoid overfitting with high dimension polynomial

## Data and settings for log purposes ##
# These settings are filled during the runtime and used as a global data structure for the logger statistics.

# Modifier Global Statistics 
mgs = {"points_generated_total": 0, # Number of generated points
       "points_generation_intervals": 0, # Number of intervals generated
       "mod_iterations": 0}  # Number of modifier iterations

# Validator Global statistics
vgs = {"points_fitting_total": 0,  # Not used
       "points_unfitting_total": 0,  # Not used
       "intervals_unfit_total": 0}  # Not used

# Logger Granularity Settings
# log_granularity:
# 0 only general stats
# 1 minimal log
# 2 medium
# 3 detailed)
lgs = {"log_granularity": 3}

# DDL: (for everything) END OF MARCH
# TODO: ?? - 04/24 a some point, introduce exploitation mode (just from modifier to sim and no validator)

# SUMOvsl settings
sumovsls = {"model_path": "C:/Users/kusic/Desktop/SSF/SUMOVSL/SPSC_MD/model_MD/",
            "sumo_path": "C:/Program Files (x86)/Eclipse/Sumo/bin/sumo"}
# sumovsls = {"model_path": "/home/amy/tmp/repos/SimEx/model_MD/",
#             "sumo_path": "/usr/share/sumo/bin/sumo"}