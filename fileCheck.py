import os
import configparser
import pandas as pd

config = configparser.ConfigParser()
config.read('config.ini')
config = config['default']

# check if file exists and delete
filepath_raw = config['output_location']
filepath_raw_filter = config['output_location']


def make_empty_file():
    transform_data = pd.DataFrame()
    transform_data.to_csv(filepath_raw)

    transform_data_filter = pd.DataFrame()
    transform_data_filter.to_csv(filepath_raw)
    return


def check_file():
    if os.path.exists(filepath_raw):
        os.remove(filepath_raw)
    else:
        print("File does not exists")

    if os.path.exists(filepath_raw_filter):
        os.remove(filepath_raw_filter)
    else:
        print("File does not exists")

    make_empty_file()
    return
