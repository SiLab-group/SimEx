from Modifiers import Modifiers

# Overall SimEx settings
# possible modes exploration (this one) and exploitation (mod with prob threes)
simexSettings={"do_plot":False,
               "extensive_search":False,
               "extensive_iteration":False, 
               "SimEx_mode":"exploration"}

# Modifier Domain Settings
mds={"domain_min_interval":1,
     "domain_max_interval":500,
     "modifier_incremental_unit":3,
     "modifier_data_point":50}

# Modifier Function Settings
mfs={'selectedModifier':Modifiers.modifierA}

# Validator Function Settings
vfs={"threshold_y_fitting":5,
     "threshold_x_interval":0.75}


## Data and settings for log purposes ##

# Modifier Global Statistics 
mgs={"points_generated_total":0, 
     "points_generation_intervals":0, 
     "mod_iterations":0}

# Validator Global statistics
vgs={"points_fitting_total":0, 
     "points_unfitting_total":0, 
     "intervals_unfit_total":0}

# Logger Granularity Settings
# log_granularity:
# 0 only general stats
# 1 minimal log
# 2 medium
# 3 detailed)
lgs={"log_granularity":3}



# DDL: (for everything) END OF MARCH
# TODO: ?? - 04/24 a some point, introduce exploitation mode (just from modifier to sim and no validator)