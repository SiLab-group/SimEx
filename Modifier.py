import numpy as np
import matplotlib.pyplot as plt
from global_settings import mdv

# Define mdv as a global variable
# mdv={"domain_min_range":1, "domain_max_range":100, "modifier_incremental_unit":3, "modifier_data_point":10}

# mdv = global_settings.mdv
#MODIFIER .py file eventually
# global mdv
class Modifier:

    def rescaler(old_list,new_min,new_max):
        new_values=[]
        for old_value in old_list:
            new_values.append((((old_value - min(old_list)) * (new_max - new_min)) / (max(old_list) - min(old_list)))+ new_min)
        return new_values
    

    def local_modifier_A(x,new_min=0,new_max=100):
        temp = x**2
        temp = Modifier.rescaler(temp,new_min,new_max)
        return temp


    def local_modifier_B(x,new_min=0,new_max=100):
        temp = x*2/3
        temp = Modifier.rescaler(temp,new_min,new_max)
        return temp


    def modifier_controller(range_list, local_modifier=local_modifier_A, do_plot=False):
        
        # Function to control modifiers given the input and the selected modifier function. Option to plot or not. 
        
        print("\nModifier controller...")
        print('  * Interval: ',range_list)
        all_interval_mod = []
        
        # Check if it's possible to generate more data points
        if mdv["modifier_data_point"] < mdv["modifier_incremental_unit"]:
            return False  # Exit the function if not possible
        
        for i in range(len(range_list)):
            print('iterations within Modifier: ',i)
            interval_min_range = range_list[i][0] #global_settings.mdv["domain_min_range"]
            interval_max_range = range_list[i][1] #global_settings.mdv["domain_max_range"]
            
            # Generate data points (incremental ticks and function modified x values) within the specified interval
            mod_ticks = np.arange(interval_min_range, interval_max_range, mdv["modifier_data_point"])
            mod_x = local_modifier(mod_ticks, new_min=interval_min_range,new_max=interval_max_range)
            
            # Normalize the function outputs to fit within the interval range
            # mod_x = rescaler(mod_x, new_max=interval_max_range, new_min=interval_min_range)
            all_interval_mod.append(mod_x)
        
        # update the mdv to decrease the interdatapoint distance for the next iteration
        mdv["modifier_data_point"] = mdv["modifier_data_point"] - mdv["modifier_incremental_unit"]
        
        if do_plot == True:
            # Plot the generated data points
            for mod_x in all_interval_mod:
                plt.scatter(mod_x, np.ones(np.shape(mod_x)))
                plt.show()
        
        print('  * Mod_x:   ',all_interval_mod)
        return all_interval_mod
