from datetime import datetime
from global_settings import lgs,mds


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

    def log_main(self, logger_arguments):
        #TODO: log simEx settings
        #TODO: log MAIN stats (i.e., iterations, stop condition, etc.)
        if logger_arguments["log_contex"] == "overall MAIN stats" and logger_arguments["main_status"] == "begin cycle":
            message = ("   ***   main cycle STARTED   ***   ")
            self._write_log('[MAIN]: ', message)
            
        if logger_arguments["log_contex"] == "Overall Stats" and logger_arguments["main_status"] == "end cycle":
            message = ("   ***   OVERALL STATS   ***   ")
            self._write_log('[MAIN]: ', message)
            message = "MOD - Total generated points: " + str(mds["points_generated_total"])
            self._write_log('[MAIN]: ', message)
            message = "MOD - Total ranges used for points generation: " + str(mds["points_generation_ranges"])
            self._write_log('[MAIN]: ', message)
            message = ("   ***   main cycle ENDED   ***   ")
            self._write_log('[MAIN]: ', message)
        
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
        
        least_fit_points = logger_arguments.get("least_fit_points")
        unfitting_ranges = logger_arguments.get("unfitting_ranges")
        if lgs["log_granularity"] > 0:
            message = "   * The unfit range(s) are: " + str(unfitting_ranges)
            self._write_log('[VAL]: ', message)

        if lgs["log_granularity"] > 1:
            message = "   * The function fit function is:  FUNCTION HERE"
            self._write_log('[VAL]: ', message)

        # add ranges min-max
        if lgs["log_granularity"] > 2:
            for i, sublist in enumerate(least_fit_points):
                message = "      * The points of the range " + str(i) + " are: " + str(sublist)
                self._write_log('[VAL]: ', message)
        

    def close(self):
        self._close_file()
