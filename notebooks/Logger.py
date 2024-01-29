from datetime import datetime
from global_settings import lgs,mds

results_data = [
    {"good_range": (0, 1), "fitting_function": "f(x) = x^2", "good_points": [(0, 0), (0.5, 0.25), (1, 1)]},
    {"good_range": (1, 2), "fitting_function": "f(x) = x^3", "good_points": [(1, 1), (1.5, 3.375), (2, 8)]},
]

unfit_residuals = []

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
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_entry = f"{timestamp} - {level} - {message}\n"
        self.file.write(log_entry)
        self.file.flush()  # Ensure the message is written immediately

    def _write_results(self):
        #timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        for element in results_data:
            result_entry = f"GI: {element['good_range']} | FF: {element['fitting_function']} | PTs: {element['good_points']}\n"
            self.file.write(result_entry)
            self.file.flush()  # Ensure the message is written immediately
        
        if not unfit_residuals:
            result_entry = "No unfit range(s) left.\n"
            self.file.write(result_entry)
            self.file.flush()  # Ensure the message is written immediately
        else:
            for element in unfit_residuals:
                result_entry = f"UR: {element['unfit_range']} | UPTs: {element['unfit_points']}\n"
                self.file.write(result_entry)
                self.file.flush()  # Ensure the message is written immediately

    def log_main(self, logger_arguments):
        #TODO: log simEx settings
        #TODO: log MAIN stats (i.e., iterations, stop condition, etc.)
        if logger_arguments["log_contex"] == "overall MAIN stats" and logger_arguments["main_status"] == "begin cycle":
            message = ("   ***   main cycle STARTED   ***   ")
            self._write_log('[MAIN]: ', message)
            
        if logger_arguments["log_contex"] == "Overall Stats" and logger_arguments["main_status"] == "end cycle":
            message = ("\n\n   ***   OVERALL STATS   ***   ")
            self._write_log('[MAIN]: ', message)
            message = "MOD - Total generated points: " + str(mds["points_generated_total"])
            self._write_log('[MAIN]: ', message)
            message = "MOD - Total ranges used for points generation: " + str(mds["points_generation_ranges"])
            self._write_log('[MAIN]: ', message)
            message = ("   ***   main cycle ENDED   ***   ")
            self._write_log('[MAIN]: ', message)
            message = ("\n\n   ***   RESULTS   ***   ")
            self._write_log('[MAIN]: ', message)
            self._write_results()
        
        if logger_arguments["log_contex"] == "overall MAIN stats" and logger_arguments["main_status"] == "no generated points":
            message = ("   ***   main cycle INTERRUPTED: No more points from Modifier   ***   ")
            self._write_log('[MAIN]: ', message)
        
        if logger_arguments["log_contex"] == "overall MAIN stats" and logger_arguments["main_status"] == "no unfit intervals":
            message = ("   ***   main cycle COMPLETED: No more unfit points/intervals from Validator   ***   ")
            self._write_log('[MAIN]: ', message)

    def log_modifier(self, logger_arguments):
        
        if logger_arguments["log_contex"] == "internal MOD stats":
            current_iteration_points_number = logger_arguments.get("current_iteration_points_number")
            all_intervals_mod = logger_arguments.get("all_intervals_mod")
            ranges_list = logger_arguments.get("ranges_list")

            if lgs["log_granularity"] > 0:
                message = "Iteration " + str(mds["mod_iterations"]) + " has generated " + str(
                    current_iteration_points_number) + " points in " + str(len(all_intervals_mod)) + " range(s)"
                self._write_log('[MOD]: ', message)

            if lgs["log_granularity"] > 1:
                message = "   * The range(s) are: " + str(ranges_list)
                self._write_log('[MOD]: ', message)

            # add ranges min-max
            if lgs["log_granularity"] > 2:
                for i, sublist in enumerate(all_intervals_mod):
                    message = "      * The points of the range " + str(i) + " are: " + str(sublist)
                    self._write_log('[MOD]: ', message)

        # if logger_arguments["log_contex"] == "overall MOD stats":
        #     message = ("   ***   Overall Stats   ***   ")
        #     self._write_log('[MOD]: ', message)
        #     message = "Total generated points: " + str(mds["points_generated_total"])
        #     self._write_log('[MOD]: ', message)
        #     message = "Total ranges used for points generation: " + str(mds["points_generation_ranges"])
        #     self._write_log('[MOD]: ', message)
            

        
    
        

    def log_simulator(self, message):
        self._write_log('ERROR', message)
    
    def log_validator(self, logger_arguments):
        
        if logger_arguments["log_contex"] == "internal VAL stats" and "validator_ranges" in logger_arguments:
            validator_ranges = logger_arguments.get("validator_ranges")
            if lgs["log_granularity"] > 0:
                message = "   * Found "+ str(len(validator_ranges)) +" unfit range(s)"
                self._write_log('[VAL]: ', message)

            if lgs["log_granularity"] > 1:
                message = "   * The fit function is:  FUNCTION HERE"
                self._write_log('[VAL]: ', message)

            # add ranges min-max
            if lgs["log_granularity"] > 2:
                message = "   * The unfit range(s) are: " + str(validator_ranges)
                self._write_log('[VAL]: ', message)
                
                
        if logger_arguments["log_contex"] == "internal VAL stats" and "local_unfit_range" in logger_arguments:
            local_unfit_range = logger_arguments.get("local_unfit_range")
            unfit_points = logger_arguments.get("unfit_points")
            if lgs["log_granularity"] > 0:
                message = "   * The local unfit range is: " + str(local_unfit_range)
                self._write_log('[VAL]: ', message)

            if lgs["log_granularity"] > 1:
                pass

            # add ranges min-max
            if lgs["log_granularity"] > 2:
                message = "      * Points are: " + str(unfit_points)
                self._write_log('[VAL]: ', message)
        

    def close(self):
        self._close_file()
