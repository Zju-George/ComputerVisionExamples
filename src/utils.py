import sys

class Logger():
    def __init__(self, loginfo='DEBUG'):
        self.loginfo = loginfo

    @staticmethod
    def info(msg):
        print(f'[INFO] {msg}')
    
    @staticmethod
    def debug(msg):
        print(f'[DEBUG] {msg}')

    @staticmethod
    def warning(msg):
        print(f'[WARNING] {msg}')
    
    @staticmethod
    def error(msg):
        print(f'[ERROR] {msg}')
        sys.exit(1)
    