import numpy as np
import matplotlib.pyplot as plt
from Modifiers import Modifiers
from global_settings import mdv,mds,lgs
from global_settings import simexSettings
from Logger import Logger

logger = Logger() 

class ModifierController:

    def modifierController(intervals_list, local_modifier=Modifiers.modifierA, do_plot=simexSettings["do_plot"],verbalinfo = 0):
             
        # Function to control modifiers given the input and the selected modifier function. Option to plot or not. 
        print("[MODC]: *** Entering Modifier controller ***")
        print("[MODC]: intervals list: ",intervals_list)
        all_intervals_mod = []
        logger_modifier_arguments = {}
        
        # Check if it's possible to generate more data points
        if (mdv["modifier_data_point"] < mdv["modifier_incremental_unit"]):
            if simexSettings["extensive_search"] is True and simexSettings["extensive_iteration"] is False:
                mdv["modifier_data_point"] = 1
                simexSettings["extensive_iteration"] = True
            else:  
                # Exit the function if not possible
                return False, intervals_list
                       
        mds["mod_iterations"] +=1
        for i, (interval_min_tick, interval_max_tick) in enumerate(intervals_list):

            # Generate data points (incremental ticks and function modified x values) within the specified interval
            mod_ticks = np.arange(interval_min_tick, interval_max_tick, mdv["modifier_data_point"])
            mod_x = local_modifier(mod_ticks, new_min=interval_min_tick, new_max=interval_max_tick)
            all_intervals_mod.append(mod_x)
        
        current_iteration_points_number = sum(len(sublist) for sublist in all_intervals_mod)
        mds["points_generation_intervals"] += len(all_intervals_mod)
        mds["points_generated_total"] += current_iteration_points_number
        
        #Modifier Logging
        logger_modifier_arguments["log_contex"] = "internal MOD stats"
        logger_modifier_arguments["current_iteration_points_number"] = current_iteration_points_number
        logger_modifier_arguments["all_intervals_mod"] = all_intervals_mod
        logger_modifier_arguments["intervals_list"] = intervals_list
        logger.log_modifier(logger_modifier_arguments)
        
    
        
        
        
        # update the mdv to decrease the interdatapoint distance for the next iteration
        mdv["modifier_data_point"] = mdv["modifier_data_point"] - mdv["modifier_incremental_unit"]
        
        if do_plot == True:
            # Plot the generated data points
            for mod_x in all_intervals_mod:
                plt.scatter(mod_x, np.ones(np.shape(mod_x)))
                plt.show()
        
        # print('  * Mod_x:   ',all_interval_mod)
        # print('  * Mod_x shape:   ',np.shape(all_interval_mod))
        return all_intervals_mod,intervals_list
