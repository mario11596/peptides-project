import configparser
import pandas as pd
import numpy as np
from numpy import unique
from sklearn.preprocessing import StandardScaler
from constants import Constants

config = configparser.ConfigParser()
config.read('config.ini')
config = config['default']

# read output file
filepath_raw = config['output_location']

filter_file = config['output_location-filter']
# remove all columns which results are overflow
overflow_columns = ['MDEC-22', 'MDEC-23', 'MDEC-33', 'MDEO-11', 'MDEC-12', 'MDEC-13', 'MDEN-12', 'MDEN-22', 'MDEC-24', 'VSA_EState9']


# check if each columns contains same values, calculation mean values
def filter_columns_file():
    constat_column = 0
    null_to_mean = 0
    data_file = pd.read_csv(filepath_or_buffer=filepath_raw, delimiter=',')

    for each_columns in data_file.loc[:, ~data_file.columns.isin(['FASTA form', 'SMILE form', 'result'])]:
        data_file[each_columns].replace('None', np.nan, inplace=True)

        if(data_file[each_columns] == data_file[each_columns][0]).all() or data_file[each_columns].isnull().all():
            constat_column += 1
            data_file.drop(each_columns, axis=1, inplace=True)

        elif data_file[each_columns].isnull().any():
            null_to_mean += 1
            mean_result_with_nan(each_columns, data_file)

    drop_overflow_columns(data_file)
    data_file.to_csv(filter_file, index=False, sep=',')
    print("Number of columns with same value or null value: " + str(constat_column))
    print("Number of columns with mean value: " + str(null_to_mean))
    return


# calculate mean for columns where the row is found with non-value (Nan)
def mean_result_with_nan(each_columns, data_file):
    all_values = data_file[each_columns].values
    transform_values = np.array(all_values)
    mean_value = np.nanmean(transform_values, dtype='float64')
    data_file[each_columns].replace(np.nan, mean_value, inplace=True)
    return


# calculate data standardization for filter data
def data_standardization():
    filter_data_file = pd.read_csv(filepath_or_buffer=filter_file, delimiter=',')

    for each_columns in filter_data_file.loc[:, ~filter_data_file.columns.isin(['FASTA form', 'SMILE form', 'result'])]:
        all_values_unscale = filter_data_file[each_columns].values
        scalar = StandardScaler()
        scaled_data = scalar.fit_transform(all_values_unscale.reshape(-1, 1))
        filter_data_file[each_columns].replace(all_values_unscale, scaled_data, inplace=True)
    filter_data_file.to_csv(filter_file, float_format='%.6f', index=False, sep=',')
    return


def drop_overflow_columns(data_file):
    for i in overflow_columns:
        if i in data_file.columns:
            data_file.drop(i, axis=1, inplace=True)
    return


# calculate unique value in each columns for remove unnecessary descriptors
def unique_value():
    low_unique_value = 0
    filter_data_file = pd.read_csv(filepath_or_buffer=filter_file, delimiter=',')

    for each_columns in filter_data_file.loc[:, ~filter_data_file.columns.isin(['FASTA form', 'SMILE form', 'result'])]:
        index = filter_data_file.columns.get_loc(each_columns)
        number_unique_value = len(unique(filter_data_file.iloc[:, index]))

        percentage = (float(number_unique_value) / len(filter_data_file)) * 100

        if(percentage < Constants.LIMIT_UNIQUE):
            low_unique_value += 1
            filter_data_file.drop(each_columns, axis=1, inplace=True)

    same_columns = filter_data_file.duplicated().sum()
    filter_data_file.drop_duplicates(keep='first')
    filter_data_file.to_csv(filter_file, index=False, sep=',')
    print("Number of columns with low unique values: " + str(low_unique_value))
    print("Number of same columns: " + str(same_columns))
    return




