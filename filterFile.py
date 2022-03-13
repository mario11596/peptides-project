import configparser
import pandas as pd
import numpy as np
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer


config = configparser.ConfigParser()
config.read('config.ini')
config = config['default']

# read output file
filepath_raw = config['output_location']
data_file = pd.read_csv(filepath_or_buffer=filepath_raw)

filter_file = config['output_location-filter']
# remove all columns which results are overflow (+ MDEC-24, VSA_EState9)
overflow_columns = ['MDEC-22', 'MDEC-23', 'MDEC-33', 'MDEO-11', 'MDEC-12', 'MDEC-13', 'MDEN-12', 'MDEN-22']


# check if each columns contains same values, calculation mean values
def filter_columns_file():
    for each_columns in data_file.loc[:, ~data_file.columns.isin(['FAST form', 'SMILE form'])]:
        data_file[each_columns].replace('None', np.nan, inplace=True)

        #check_variance(each_columns)

        if(data_file[each_columns] == data_file[each_columns][0]).all() or data_file[each_columns].isnull().all():
            data_file.drop(each_columns, axis=1, inplace=True)
        elif data_file[each_columns].isnull().any():
            mean_result_with_nan(each_columns)

    data_file.drop(overflow_columns, axis=1, inplace=True)
    data_file.to_csv(filter_file, index=False)
    return

# scaling all results from csv file to get small numbers
def mean_result_with_nan(each_columns):
    #df[each_column.name].replace(np.nan, mean_value, inplace=True)
    all_values = data_file[each_columns].values
    transform_values = np.array(all_values)
    mean_value = np.nanmean(transform_values, dtype='float64')
    data_file[each_columns].replace(np.nan, mean_value, inplace=True)
    return


def check_variance(each_columns):
    all_values = data_file[each_columns].values
    transform_values = np.array(all_values)

    threshold_custom = 0.05
    transform = VarianceThreshold(threshold=threshold_custom)
    X_sel = transform.fit_transform()
    print(X_sel.shape)
    return