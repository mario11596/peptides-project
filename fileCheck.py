import os
import configparser

config = configparser.ConfigParser()
config.read('config.ini')
config = config['default']

# check if file exists and delete
filepath_raw = config['output_location']


def check_file():
    if os.path.exists(filepath_raw):
        os.remove(filepath_raw)
    else:
        print("File does not exists")
    return