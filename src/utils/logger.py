import logging
from datetime import datetime

class Logger:
    def __init__(self, name: str, log_file: str = None, level=logging.INFO):
        """
        Initialize the logger.

        Args:
            name (str): Name of the logger.
            log_file (str): Optional path to a log file. If None, logs will only be output to the console.
            level (int): Logging level (e.g., logging.INFO, logging.DEBUG).
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        
        # Set up console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)

        # Set up file handler if a file path is provided
        if log_file:
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(level)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)

    def info(self, message: str):
        """Log an info-level message."""
        self.logger.info(message)

    def error(self, message: str):
        """Log an error-level message."""
        self.logger.error(message)
    def warn(self, message: str):
        """Log a warning-level message."""
        self.logger.warning(message)
    def debug(self, message: str):
        """Log a debug-level message."""
        self.logger.debug(message)
