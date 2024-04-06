import numpy as np
import matplotlib.pyplot as plt
#from Modifiers import Modifiers
from global_settings import mds,mgs,lgs
from global_settings import simexSettings
from Logger import Logger

logger = Logger() 

class ModifierController:

    def modifierController(intervals_list, selectedModifier, do_plot):
        print("Modifier...")     
        # Function to control modifiers given the input and the selected modifier function. Option to plot or not. 
        #print("[MODC]: *** Entering Modifier controller ***")
        print("[MODC]: intervals list: ",intervals_list)
        all_intervals_mod = []
        logger_modifier_arguments = {}
        
        # Check if it's possible to generate more data points
        if (mds["modifier_data_point"] < mds["modifier_incremental_unit"]):
            if simexSettings["extensive_search"] is True and simexSettings["extensive_iteration"] is False:
                mds["modifier_data_point"] = 1
                simexSettings["extensive_iteration"] = True
            else:  
                # Exit the function if not possible
                return False, intervals_list
                       
        mgs["mod_iterations"] +=1
        for i, (interval_min_tick, interval_max_tick) in enumerate(intervals_list):

            # Generate data points (incremental ticks and function modified x values) within the specified interval
            print("[MODC]: (interval_min_tick, interval_max_tick): ",(interval_min_tick, interval_max_tick))
            mod_ticks = np.arange(interval_min_tick, interval_max_tick, mds["modifier_data_point"])
        
            # mod_ticks = np.arange(mds["domain_min_interval"], mds["domain_max_interval"], mds["modifier_data_point"])
            print("[MODC]: mod_ticks: ",mod_ticks)

            mod_x = selectedModifier(mod_ticks, new_min=interval_min_tick, new_max=interval_max_tick)
            print("[MODC]: mod_x: ",mod_x)

            all_intervals_mod.append(mod_x)
        
        current_iteration_points_number = sum(len(sublist) for sublist in all_intervals_mod)
        mgs["points_generation_intervals"] += len(all_intervals_mod)
        mgs["points_generated_total"] += current_iteration_points_number
        
        #Modifier Logging
        logger_modifier_arguments["log_contex"] = "internal MOD stats"
        logger_modifier_arguments["current_iteration_points_number"] = current_iteration_points_number
        logger_modifier_arguments["all_intervals_mod"] = all_intervals_mod
        logger_modifier_arguments["intervals_list"] = intervals_list
        logger.log_modifier(logger_modifier_arguments)
        
    
        
        
        
        # update the mdv to decrease the interdatapoint distance for the next iteration
        mds["modifier_data_point"] = mds["modifier_data_point"] - mds["modifier_incremental_unit"]
        
        if do_plot == True:
            # Plot the generated data points
            for mod_x in all_intervals_mod:
                plt.scatter(mod_x, np.ones(np.shape(mod_x)))
                plt.show()
        
        # print('  * Mod_x:   ',all_interval_mod)
        # print('  * Mod_x shape:   ',np.shape(all_interval_mod))
        return all_intervals_mod,intervals_list
