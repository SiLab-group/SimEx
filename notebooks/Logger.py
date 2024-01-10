from datetime import datetime


class Logger:
    
    def __init__(self, filename='my_log.txt'):
        self.filename = filename
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

    def log_modifier(self, message):
        self._write_log('[MOD]: ', message)

    def log_simulator(self, message):
        self._write_log('ERROR', message)
    
    def log_validator(self, message):
        self._write_log('ERROR', message)
        

    def close(self):
        self._close_file()
