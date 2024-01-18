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

    def log_main(self, message):
        #TODO: log simEx settings
        #TODO: log MAIN stats (i.e., iterations, stop condition, etc.)
        self._write_log('[MAIN]:', message)

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

        if logger_arguments["log_contex"] == "overall MOD stats":
            message = ("   ***   Overall Stats   ***   ")
            self._write_log('[MOD]: ', message)
            message = "Total generated points: " + str(mds["points_generated_total"])
            self._write_log('[MOD]: ', message)
            message = "Total ranges used for points generation: " + str(mds["points_generation_ranges"])
            self._write_log('[MOD]: ', message)
            

        
        
        
        
        
        
        

    def log_simulator(self, message):
        self._write_log('ERROR', message)
    
    def log_validator(self, message):
        self._write_log('ERROR', message)
        

    def close(self):
        self._close_file()
