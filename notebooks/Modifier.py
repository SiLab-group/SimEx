import numpy as np
import matplotlib.pyplot as plt
from global_settings import mdv,mds,lgs
from global_settings import simexSettings
from Logger import Logger

logger = Logger() 

class Modifier:


    def rescaler(old_list, new_min, new_max):
        if not np.any(old_list):
            return []  # handle empty list case

        old_min = min(old_list)
        old_max = max(old_list)

        if old_min == old_max:
            # Handle the case when all elements in old_list are the same
            return [new_min] * len(old_list)

        new_values = []
        for old_value in old_list:
            denominator = old_max - old_min
            if denominator != 0:
                scaled_value = (((old_value - old_min) * (new_max - new_min)) / denominator) + new_min
                new_values.append(scaled_value)
            else:
                # Handle the case when the range is zero
                new_values.append(new_min)
        return new_values

    

    def local_modifier_A(x,new_min,new_max):
        temp = x**2
        temp = Modifier.rescaler(temp,new_min,new_max)
        return temp


    def local_modifier_B(x,new_min,new_max):
        temp = x*2/3
        temp = Modifier.rescaler(temp,new_min,new_max)
        return temp


    def modifier_controller(ranges_list, local_modifier=local_modifier_A, do_plot=simexSettings["do_plot"],verbalinfo = 0):
             
        # Function to control modifiers given the input and the selected modifier function. Option to plot or not. 
        print("[MODC]: *** Entering Modifier controller ***")
        print("[MODC]: intervals list: ",ranges_list)
        all_intervals_mod = []
        
        # Check if it's possible to generate more data points
        if (mdv["modifier_data_point"] < mdv["modifier_incremental_unit"]):
            if simexSettings["extensive_search"] is True and simexSettings["extensive_iteration"] is False:
                mdv["modifier_data_point"] = 1
                simexSettings["extensive_iteration"] = True
            else:  
                temp_log_tot_points = "Total generated points: "+str(mds["points_generated_total"])
                temp_log_tot_ranges = "Total ranges used for points generation: "+ str(mds["points_generation_ranges"])
                logger.log_modifier("   ***   MOD overall stats    ***   ")
                logger.log_modifier(temp_log_tot_ranges)
                logger.log_modifier(temp_log_tot_points)
                return False  # Exit the function if not possible
            
        mds["mod_iterations"] +=1
        for i, (interval_min_range, interval_max_range) in enumerate(ranges_list):

            #print("[MOD]: iteration: ",i,", interval: ",interval_min_range,"-",interval_max_range)

            # Generate data points (incremental ticks and function modified x values) within the specified interval
            mod_ticks = np.arange(interval_min_range, interval_max_range, mdv["modifier_data_point"])
            mod_x = local_modifier(mod_ticks, new_min=interval_min_range, new_max=interval_max_range)
            all_intervals_mod.append(mod_x)
        
        current_iteration_points_number = sum(len(sublist) for sublist in all_intervals_mod)
        mds["points_generation_ranges"] += len(all_intervals_mod)
        mds["points_generated_total"] += current_iteration_points_number
        if lgs["log_granularity"] > 0:
            temp_log="[MOD]: Iteration "+str(mds["mod_iterations"])+" has generated "+str(current_iteration_points_number)+" points in "+str(len(all_intervals_mod))+" range(s)"
            logger.log_modifier(temp_log)
        
        if lgs["log_granularity"] > 1:
            temp_log="[MOD]:   * The range(s) are: "+str(ranges_list)
            logger.log_modifier(temp_log)

            # add ranges min-max
        if lgs["log_granularity"] > 2:
            for i,sublist in enumerate(all_intervals_mod):
                temp_log="[MOD]:      * The points of the range "+str(i)+" are: "+str(sublist)
                logger.log_modifier(temp_log)

        
        
        
        # update the mdv to decrease the interdatapoint distance for the next iteration
        mdv["modifier_data_point"] = mdv["modifier_data_point"] - mdv["modifier_incremental_unit"]
        
        if do_plot == True:
            # Plot the generated data points
            for mod_x in all_intervals_mod:
                plt.scatter(mod_x, np.ones(np.shape(mod_x)))
                plt.show()
        
        # print('  * Mod_x:   ',all_interval_mod)
        # print('  * Mod_x shape:   ',np.shape(all_interval_mod))
        return all_intervals_mod
