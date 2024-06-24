# Overall SimEx settings
# possible modes exploration (this one) and exploitation (mod with prob threes)
simexSettings = {"do_plot": False,  # No special meaning at the moment. Should be refactored.
                 "extensive_search": False,  #  Complete exploration setting modifier_data_point to 1 and enabling extensive iteration
                 "extensive_iteration": False,  #  Gets enabled when extensive search is True
                 "SimEx_mode": "exploration"}  # Only exploration implemented

# Modifier Domain Settings
mds = {"domain_min_interval": 2500,
       "domain_max_interval": 4000,
       "modifier_incremental_unit": 25,  # Second round modifier_data_point - modifier_incremental_unit to make smaller granularity. Note: If extensive search True then increment set to 1
       "modifier_data_point": 100  # Data points on the X axis in the first round. Step size (100,200,300,..)
       }

# Validator Function Settings
vfs = {'threshold_y_fitting': 15,  # Threshold on the y axis
       'threshold_x_interval': 0.80,  # Percentage on x how good or bad point
       'degree': 2,  # Minimum degree for exploration
       'max_deg': 9,  # Max degree for exploration
       'early_stop': True,  # Validator: if True and not sufficient improvement by increasing dimension, we stop
       'improvement_threshold': 0.1,  # Sufficient improvement threshold
       'penality_weight': 1}  # Penalty for MSE to avoid overfitting with high dimension polynomial

## Data and settings for log purposes ##

# Modifier Global Statistics 
mgs = {"points_generated_total": 0,
       "points_generation_intervals": 0,
       "mod_iterations": 0}

# Validator Global statistics
vgs = {"points_fitting_total": 0,
       "points_unfitting_total": 0,
       "intervals_unfit_total": 0}

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