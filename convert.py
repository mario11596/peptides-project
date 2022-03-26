from rdkit import Chem
import configparser
from mordred import Calculator, descriptors
import pandas as pd


config = configparser.ConfigParser()
config.read('config.ini')
config = config['default']

# input original file
filepath_raw = config['peptides_filepath_raw']
header = ['sequence', 'label']
data_file = pd.read_csv(filepath_or_buffer=filepath_raw, header=0, delimiter=',')

# file with peptides in SMILE annotation, prepared all columns
output_raw = config['output_location']


def prepare_columns():
    calc = Calculator(descriptors, ignore_3D=True)
    peptides_name_columns = calc._name_dict.keys()

    headerList = ['FASTA form', 'SMILE form']
    for i in peptides_name_columns:
        headerList.append(i)
    headerList.append("result")

    transform_data_file = pd.DataFrame(columns=headerList)
    transform_data_file.to_csv(output_raw, index=False, sep=',')
    return


def transform_to_smile():
    prepare_columns()

    df_temp = data_file.iloc[:, 0:2]
    with open(output_raw, 'a') as transform_data:
        for peptide, result in df_temp.itertuples(index=False):
            smile = Chem.MolToSmiles(Chem.MolFromFASTA(peptide))
            calculate_descriptors = calculation_all_descriptors(smile)
            transform_data.write(peptide + ',' + smile + ',' + calculate_descriptors + ',' + str(result) + '\n')
    transform_data.close()
    return


def calculation_all_descriptors(smile):
    calc = Calculator(descriptors, ignore_3D=True)
    all_descriptors = calc(Chem.MolFromSmiles(smile)).fill_missing(value=None)

    result = ','.join([str(elem) for elem in all_descriptors.values()])
    return result