import os
import re
import csv
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
from global_settings import lgs, mgs, vfs, ops

all_fit_intervals_data = []
remaining_unfit_intervals = []


def get_coefficients(interval):
    # Convert the string into a function array of terms
    terms = re.findall(r'([+-]?\s*\d+\.?\d*(?:e[+-]?\d+)?)(x\^\d+)?', interval['fitting_function'].replace(' ', ''))
    # For each element if x present, we extract exponent
    coefficients = [0] * (vfs['max_deg']+1)  # Initialize a list for coefficients
    for term in terms:
        coef = float(term[0])
        if term[1]:  # If there is an 'x' term
            exponent = int(term[1][2:])  # Get the exponent
            while len(coefficients) <= exponent:  # Expand the list if needed
                coefficients.append(0)
            # Assign the coefficient to the corresponding position in the list
            coefficients[exponent] = coef
        else:  # If there is no 'x' term, it's the constant term
            coefficients[0] = coef
    return coefficients


class Logger:

    def __init__(self, filename="LOG-"):
        self.remaining_unfit_intervals = []
        self.all_fit_intervals_data = []
        self.timestamp = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        self.filename = f"{filename}{self.timestamp}.txt"
        self._open_file()

    def _open_file(self):
        self.file = open(self.filename, 'a')

    def _close_file(self):
        if self.file and not self.file.closed:
            self.file.close()

    def _write_log(self, level, message):
        if level:
            timestamp = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
            message = f"{timestamp} - {level} - {message}\n"
        self.file.write(message)
        self.file.flush()  # Ensure the message is written immediately

    def _plot_results(self, all_fit_intervals_data, remaining_unfit_intervals):
        # Create graph
        _, ax = plt.subplots(figsize=(10, 5))
        # Remember color for the same fitted functions
        colors = {}
        # Plot FI intervals with their fitting functions
        for element in all_fit_intervals_data:
            interval = element['interval']
            fitting_function_str = element['fitting_function']

            coefficients = get_coefficients(element)
            # Reverse the list to match the order expected by np.poly1d
            fitting_function = np.poly1d(coefficients[::-1])

            # Adjust the number of points as needed
            x = np.linspace(interval[0], interval[1], 400)
            y = fitting_function(x)
            if fitting_function_str in colors.keys():
                ax.plot(x, y, linewidth=3, label=f'Interval: [{round(interval[0]), round(interval[1])}]',
                        color=colors[fitting_function_str])
            else:
                ax.plot(x, y, linewidth=3, label=f'Interval: [{round(interval[0]), round(interval[1])}]')
                color = ax.get_lines()[-1].get_color()
                colors[fitting_function_str] = color
            # plt.ylim([-100, 100])
            # ax.set_xticks(np.arange(*ax.get_xlim(),
            #                         (ax.get_xlim()[1] - ax.get_xlim()[0]) / 20))

        for element in remaining_unfit_intervals:
            ax.axvspan(*element['interval'], color='gray',
                       alpha=0.3, label='unfit Interval')

        plt.xlabel(ops['x_labels'])
        plt.ylabel(ops['y_labels'])
        plt.title(ops['title'])
        plt.legend()
        plt.show()

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
            # OVERALL SORTED
            all_intervals = all_fit_intervals_data + remaining_unfit_intervals
            all_intervals.sort(key=lambda x: x['interval'][0])

            for element in all_intervals:
                if len(element.keys()) > 1:
                    result_entry = f"FI: {str(element['interval']):<40} | FF: {str(element['fitting_function']):<30} | PTs: {str(element['fit_points']):<50}\n"
                else:
                    result_entry = f"UI: {str(element['interval']):<40} | \n"
                self.file.write(result_entry)
                self.file.flush()  # Ensure the message is written immediately
        self.all_fit_intervals_data = all_fit_intervals_data
        self.remaining_unfit_intervals = remaining_unfit_intervals
        # Write results to csv file
        self.write_csv_file()
        # Plot results
        self._plot_results(all_fit_intervals_data, remaining_unfit_intervals)

    def log_main(self, logger_arguments):
        # TODO: log simEx settings
        if logger_arguments["log_contex"] == "overall MAIN stats" and logger_arguments["main_status"] == "begin cycle":
            message = "   ***   main cycle STARTED   ***   \n"
            self._write_log(False, message)

        if logger_arguments["log_contex"] == "Overall Stats" and logger_arguments["main_status"] == "end cycle":
            message = "\n\n   ***   OVERALL STATS   ***   \n"
            self._write_log(False, message)
            message = "MOD - Total generated points: " + \
                      str(mgs["points_generated_total"])
            self._write_log('[MAIN]: ', message)
            message = "MOD - Total intervals used for points generation: " + \
                      str(mgs["points_generation_intervals"])
            self._write_log('[MAIN]: ', message)
            message = "   ***   main cycle ENDED   ***   "
            self._write_log('[MAIN]: ', message)
            message = "\n\n   ***   RESULTS   ***   \n"
            self._write_log(False, message)
            self._write_results()

        if logger_arguments["log_contex"] == "overall MAIN stats" and logger_arguments[
            "main_status"] == "no generated points":
            for element in logger_arguments.get("remaining_unfit_intervals"):
                new_unfit_entry = {"interval": element}
                remaining_unfit_intervals.append(new_unfit_entry)
            message = (
                "   ***   main cycle INTERRUPTED: No more points from Modifier   ***   ")
            self._write_log('[MAIN]: ', message)
            message = ("   Remaining unfit intervals: " +
                       str(logger_arguments['remaining_unfit_intervals']))
            self._write_log('[MAIN]: ', message)

        if logger_arguments["log_contex"] == "overall MAIN stats" and logger_arguments[
            "main_status"] == "no unfit intervals":
            message = (
                "   ***   main cycle COMPLETED: No more unfit points/intervals from Validator   ***   ")
            self._write_log('[MAIN]: ', message)

    def log_modifier(self, logger_arguments):

        if logger_arguments["log_contex"] == "internal MOD stats":
            current_iteration_points_number = logger_arguments.get(
                "current_iteration_points_number")
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
                    message = "      * The x value(s) of the interval " + \
                              str(i) + " is/are: " + str(sublist)
                    self._write_log('[MOD]: ', message)

    def log_simulator(self, message):
        self._write_log('ERROR', message)

    def log_validator(self, logger_arguments):

        if logger_arguments["log_contex"] == "internal VAL stats" and "validator_intervals" in logger_arguments:
            validator_intervals = logger_arguments.get("validator_intervals")
            if lgs["log_granularity"] > 0:
                message = "   * Found " + \
                          str(len(validator_intervals)) + " unfit interval(s)"
                self._write_log('[VAL]: ', message)

            if lgs["log_granularity"] > 1:
                message = "     * The fit function is:  FUNCTION HERE"
                self._write_log('[VAL]: ', message)

            # add interval min-max
            if lgs["log_granularity"] > 2:
                message = "     * The unfit interval(s) are: " + \
                          str(validator_intervals)
                self._write_log('[VAL]: ', message)

        if logger_arguments["log_contex"] == "internal VAL stats" and "new_unfit_interval" in logger_arguments:
            new_unfit_interval = logger_arguments.get("new_unfit_interval")
            unfit_points = logger_arguments.get("unfit_points")
            if lgs["log_granularity"] > 0:
                message = "   * The new unfit interval is: " + \
                          str(new_unfit_interval)
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

    def write_csv_file(self):
        with open(f'simex_output-{os.environ["INSTANCE_NAME"]}-{self.timestamp}.csv', 'w') as f:
            # Create the csv writer
            writer = csv.writer(f)
            # Create header for the CSV file based on the global_settings configuration
            header = ['interval_start', 'interval_end']
            # Append header reversed max_degree9,max_degree8...max_degree0 range defined in global settings
            [header.append(f'exponent_max_degree{i}') for i in reversed(range(0, vfs['max_deg'] + 1))]
            writer.writerow(header)
            for interval in self.all_fit_intervals_data:
                # For each element if x present, we extract exponent
                coefficients = get_coefficients(interval)
                # For given interval
                row = [interval['interval'][0], interval['interval'][1]]
                # Append all the exponents in reversed order
                [row.append(c) for c in reversed(coefficients)]
                writer.writerow(row)

            # For unfit intervals append 0
            for u_interval in self.remaining_unfit_intervals:
                row = [u_interval['interval'][0], u_interval['interval'][1]]
                [row.append(0) for i in range(0, vfs['max_deg']+1)]
                writer.writerow(row)
            print(f'Data written to the csv file simex_output-{os.environ["INSTANCE_NAME"]}-{self.timestamp}.csv')
