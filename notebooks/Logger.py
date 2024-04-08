import matplotlib.pyplot as plt
import numpy as np
import re

from datetime import datetime
from global_settings import lgs,mgs

all_fit_intervals_data = []
remaining_unfit_intervals = []

class Logger:
    
    def __init__(self, filename="LOG-"):
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self.filename = f"{filename}{timestamp}.txt"
        self._open_file()

    def _open_file(self):
        self.file = open(self.filename, 'a')

    def _close_file(self):
        if self.file and not self.file.closed:
            self.file.close()

    def _write_log(self, level, message):
        if not level:
            self.file.write(message)
            self.file.flush()
        else:
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            log_entry = f"{timestamp} - {level} - {message}\n"
            self.file.write(log_entry)
            self.file.flush()  # Ensure the message is written immediately

    # def _write_results(self):
    #     #timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
    #     for element in all_fit_intervals_data:
    #         result_entry = f"GI: {element['fit_interval']} | FF: {element['fitting_function']} | PTs: {element['fit_points']}\n"
    #         self.file.write(result_entry)
    #         self.file.flush()  # Ensure the message is written immediately
        
    #     if not remaining_unfit_intervals:
    #         result_entry = "No unfit interval(s) left.\n"
    #         self.file.write(result_entry)
    #         self.file.flush()  # Ensure the message is written immediately
    #     else:
    #         for element in remaining_unfit_intervals:
    #             result_entry = f"UR: {element['unfit_interval']} | UPTs: {element['unfit_points']}\n"
    #             self.file.write(result_entry)
    #             self.file.flush()  # Ensure the message is written immediately
        
    def _plot_results(self,all_fit_intervals_data,remaining_unfit_intervals):
        
        #FI_intervals = [element for element in all_intervals if len(element.keys()) > 1]
        #UI_intervals = [element for element in all_intervals if len(element.keys()) < 1]

        _, ax = plt.subplots(figsize=(10, 5))
   

        # Plot FI intervals with their fitting functions
        for element in all_fit_intervals_data:
            interval = element['interval']
            fitting_function_str = element['fitting_function']
            
            # Convert the string into a function
            terms = re.findall(r'([+-]?\s*\d+\.?\d*(?:e[+-]?\d+)?)(x\^\d+)?', fitting_function_str.replace(' ', ''))
            coefficients = [0]*10  # Initialize a list of 10 zeros
            for term in terms:
                coef = float(term[0])
                if term[1]:  # If there is an 'x' term
                    exponent = int(term[1][2:])  # Get the exponent
                    while len(coefficients) <= exponent:  # Expand the list if needed
                        coefficients.append(0)
                    coefficients[exponent] = coef  # Assign the coefficient to the corresponding position in the list
                else:  # If there is no 'x' term, it's the constant term
                    coefficients[0] = coef
            fitting_function = np.poly1d(coefficients[::-1])  # Reverse the list to match the order expected by np.poly1d
            
            x = np.linspace(interval[0], interval[1], 400)  # Adjust the number of points as needed
            y = fitting_function(x)
            ax.plot(x, y, label=f'Interval: {interval}')
            plt.ylim([-100,100])
            ax.set_xticks(np.arange(*ax.get_xlim(), (ax.get_xlim()[1]-ax.get_xlim()[0])/20))
            #ax.set_xticks(list(ax.get_xticks()) + [interval[0], interval[1]])
            
        for element in remaining_unfit_intervals:
            #bad_interval = element['interval']
            #ax.axvspan(bad_interval[0], bad_interval[1], color='gray', alpha=0.3, label='unfit Interval')
            ax.axvspan(*element['interval'], color='gray', alpha=0.3, label='unfit Interval')


    
    
    def _write_results(self):
        
        if not remaining_unfit_intervals:
            all_fit_intervals_data.sort(key=lambda x: x['interval'][0])
            result_entry = "No unfit interval(s) left.\n"
            self.file.write(result_entry)
            self.file.flush()  # Ensure the message is written immediately
            
            for element in all_fit_intervals_data:
                result_entry = f"FI: {str(element['interval']):<20} | FF: {str(element['fitting_function']):<30} | PTs: {str(element['fit_points']):<50}\n"
                self.file.write(result_entry)
                self.file.flush()  # Ensure the message is written immediately
        else:    
            #OVERALL SORTED
            all_intervals = all_fit_intervals_data + remaining_unfit_intervals
            all_intervals.sort(key=lambda x: x['interval'][0])

            for element in all_intervals:
                if (len(element.keys()) > 1):
                    result_entry = f"FI: {str(element['interval']):<40} | FF: {str(element['fitting_function']):<30} | PTs: {str(element['fit_points']):<50}\n"
                else:
                    result_entry = f"UI: {str(element['interval']):<40} | \n"
                self.file.write(result_entry)
                self.file.flush()  # Ensure the message is written immediately
        self._plot_results(all_fit_intervals_data,remaining_unfit_intervals)
        
        
        
        


    def log_main(self, logger_arguments):
        #TODO: log simEx settings
        if logger_arguments["log_contex"] == "overall MAIN stats" and logger_arguments["main_status"] == "begin cycle":
            message = ("   ***   main cycle STARTED   ***   \n")
            self._write_log(False, message)
            
        if logger_arguments["log_contex"] == "Overall Stats" and logger_arguments["main_status"] == "end cycle":
            message = ("\n\n   ***   OVERALL STATS   ***   \n")
            self._write_log(False, message)
            message = "MOD - Total generated points: " + str(mgs["points_generated_total"])
            self._write_log('[MAIN]: ', message)
            message = "MOD - Total intervals used for points generation: " + str(mgs["points_generation_intervals"])
            self._write_log('[MAIN]: ', message)
            message = ("   ***   main cycle ENDED   ***   ")
            self._write_log('[MAIN]: ', message)
            message = ("\n\n   ***   RESULTS   ***   \n")
            self._write_log(False, message)
            self._write_results()
        
        if logger_arguments["log_contex"] == "overall MAIN stats" and logger_arguments["main_status"] == "no generated points":
            for element in logger_arguments.get("remaining_unfit_intervals"):
                new_unfit_entry = {"interval": element }
                remaining_unfit_intervals.append(new_unfit_entry)
            message = ("   ***   main cycle INTERRUPTED: No more points from Modifier   ***   ")
            self._write_log('[MAIN]: ', message)
            message = ("   Remaining unfit intervals: "+str(logger_arguments['remaining_unfit_intervals']))
            self._write_log('[MAIN]: ', message)
        
        if logger_arguments["log_contex"] == "overall MAIN stats" and logger_arguments["main_status"] == "no unfit intervals":
            message = ("   ***   main cycle COMPLETED: No more unfit points/intervals from Validator   ***   ")
            self._write_log('[MAIN]: ', message)

    def log_modifier(self, logger_arguments):
        
        if logger_arguments["log_contex"] == "internal MOD stats":
            current_iteration_points_number = logger_arguments.get("current_iteration_points_number")
            all_intervals_mod = logger_arguments.get("all_intervals_mod")
            intervals_list = logger_arguments.get("intervals_list")

            if lgs["log_granularity"] > 0:
                message = "Iteration " + str(mgs["mod_iterations"]) + " has generated " + str(
                    current_iteration_points_number) + " points in " + str(len(all_intervals_mod)) + " interval(s)"
                self._write_log('[MOD]: ', message)

            if lgs["log_granularity"] > 1:
                message = "   * The interval(s) are: " + str(intervals_list)
                self._write_log('[MOD]: ', message)

            # add intervals min-max
            if lgs["log_granularity"] > 2:
                for i, sublist in enumerate(all_intervals_mod):
                    message = "      * The x value(s) of the interval " + str(i) + " is/are: " + str(sublist)
                    self._write_log('[MOD]: ', message)



    def log_simulator(self, message):
        self._write_log('ERROR', message)


    def log_validator(self, logger_arguments):
        
        if logger_arguments["log_contex"] == "internal VAL stats" and "validator_intervals" in logger_arguments:
            validator_intervals = logger_arguments.get("validator_intervals")
            if lgs["log_granularity"] > 0:
                message = "   * Found "+ str(len(validator_intervals)) +" unfit interval(s)"
                self._write_log('[VAL]: ', message)

            if lgs["log_granularity"] > 1:
                message = "     * The fit function is:  FUNCTION HERE"
                self._write_log('[VAL]: ', message)

            # add interval min-max
            if lgs["log_granularity"] > 2:
                message = "     * The unfit interval(s) are: " + str(validator_intervals)
                self._write_log('[VAL]: ', message)
                
                
        if logger_arguments["log_contex"] == "internal VAL stats" and "new_unfit_interval" in logger_arguments:
            new_unfit_interval = logger_arguments.get("new_unfit_interval")
            unfit_points = logger_arguments.get("unfit_points")
            if lgs["log_granularity"] > 0:
                message = "   * The new unfit interval is: " + str(new_unfit_interval)
                self._write_log('[VAL]: ', message)

            if lgs["log_granularity"] > 1:
                pass

            # add intervals min-max
            if lgs["log_granularity"] > 2:
                message = "      * Points are: " + str(unfit_points)
                self._write_log('[VAL]: ', message)
        # logs the fit intervals, fitting functions, and related points
        if logger_arguments["log_contex"] == "fit_VAL_stats" and "fit_interval" in logger_arguments:
            new_fit_entry = {
            "interval": logger_arguments.get("fit_interval"),
            "fitting_function": logger_arguments.get("fitting_function"),
            "fit_points": logger_arguments.get("fit_points")}
            all_fit_intervals_data.append(new_fit_entry)
            
        

    def close(self):
        self._close_file()
