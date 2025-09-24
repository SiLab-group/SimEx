import numpy as np
import matplotlib.pyplot as plt
from ..core.settings import mgs

class ModifierController:

    def __init__(self, logger, settings):
        self.logger = logger
        self.settings = settings

    def control(self, intervals_list, selected_modifier, do_plot):
        print("Modifier...")
        # Function to control modifiers given the input and the selected modifier function. Option to plot or not. 
        # print("[MODC]: *** Entering Modifier controller ***")
        #print("[MODC]: intervals list: ", intervals_list)
        all_intervals_mod = []
        logger_modifier_arguments = {}

        # Check if it's possible to generate more data points
        if self.settings.modifier_data_point < self.settings.modifier_incremental_unit:
            if self.settings.extensive_search is True and self.settings.extensive_iteration is False:
                self.settings.modifier_data_point = 1
                self.settings.extensive_iteration = True
            else:
                # Exit the function if not possible
                return False, intervals_list
        mgs["mod_iterations"] += 1

        mod_ticks = np.arange(self.settings.domain_min_interval, self.settings.domain_max_interval, self.settings.modifier_data_point)
        for _, (interval_min_tick, interval_max_tick) in enumerate(intervals_list):

            # Generate data points (incremental ticks and function modified x values) within the specified interval
            print("[MODC]: (interval_min_tick, interval_max_tick): ", (interval_min_tick, interval_max_tick))

            # Extract mod_ticks that are within intervals_list
            mod_filtered_ticks = [tick for tick in mod_ticks if interval_min_tick < tick < interval_max_tick]
            # If not empty
            if np.any(mod_filtered_ticks):
                print("[MODC]: mod_ticks: ", mod_filtered_ticks)
                # FIX: Set type to np.int64 due to issue on windows: https://github.com/numpy/numpy/issues/8433
                # Fixed in numpy 2.0
                mod_x = selected_modifier(mod_filtered_ticks, new_min=np.min(np.array(mod_filtered_ticks, dtype=np.int64)),
                                          new_max=np.max(np.array(mod_filtered_ticks, dtype=np.int64)))

                print("[MODC]: mod_x: ", mod_x)
                print(f"[MODC]: mod_x: ", {len(mod_x)})
                if self.settings.add_first_last_points:
                    new_mod_x = [float(interval_min_tick)] + mod_x + [float(interval_max_tick)]
                    all_intervals_mod.append(new_mod_x)
                else:
                    all_intervals_mod.append(mod_x)

        current_iteration_points_number = sum(len(sublist) for sublist in all_intervals_mod)
        mgs["points_generation_intervals"] += len(all_intervals_mod)
        mgs["points_generated_total"] += current_iteration_points_number

        # all_intervals_mod = [item for sublist in all_intervals_mod for item in sublist]

        # Modifier Logging
        logger_modifier_arguments["log_contex"] = "internal MOD stats"
        logger_modifier_arguments["current_iteration_points_number"] = current_iteration_points_number
        logger_modifier_arguments["all_intervals_mod"] = all_intervals_mod
        logger_modifier_arguments["intervals_list"] = intervals_list
        self.logger.log_modifier(logger_modifier_arguments)

        # update the mdv to decrease the interdatapoint distance for the next iteration
        self.settings.modifier_data_point = self.settings.modifier_data_point - self.settings.modifier_incremental_unit

        if do_plot:
            # Plot the generated data points
            for mod_x in all_intervals_mod:
                plt.scatter(mod_x, np.ones(np.shape(mod_x)))
                plt.show()

        # print('  * Mod_x:   ',all_interval_mod)
        # print('  * Mod_x shape:   ',np.shape(all_interval_mod))
        return all_intervals_mod, intervals_list
