import configparser
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
#from mlxtend.feature_selection import SequentialFeatureSelector as Ssfs
from sklearn.model_selection import LeaveOneOut, train_test_split
from sklearn.feature_selection import SequentialFeatureSelector

from constants import Constants

config = configparser.ConfigParser()
config.read('config.ini')
config = config['default']
filter_file = config['output_location-filter']


def train_model_catalytic():
    filter_data_file = pd.read_csv(filepath_or_buffer=filter_file, delimiter=',')

    all_data_feature = filter_data_file.drop(labels=['FASTA form', 'SMILE form', 'result'], axis=1)
    target = filter_data_file["result"]

    forward_selection(all_data_feature, target)


def forward_selection(all_data_feature, target):

    loo_data = LeaveOneOut()
    """
    prediction_results = []
    target_results = []
    probability_target = []
    rez = []

    for train_index, test_index in loo_data.split(all_data_feature):
        print('krenuo sam')
        X_train, X_test = all_data_feature.iloc[train_index, :], all_data_feature.iloc[test_index, :]
        y_train, y_test = target.iloc[train_index], target.iloc[test_index]
        model = RandomForestClassifier(n_estimators=Constants.N_ESTIMATORS, max_features=Constants.MAX_FEATURES,
                                       min_samples_leaf=Constants.MIN_SAMPLES_LEAF)
        sfs = SequentialFeatureSelector(model, direction='forward', scoring='accuracy', n_jobs=2)
        sfs.fit(X_train, y_train)
        rez.append(sfs.get_feature_names_out())
        print(sfs.get_feature_names_out())
    """
    X_train, X_test, y_train, y_test = train_test_split(all_data_feature, target, test_size=0.3, random_state=0)
    model = RandomForestClassifier(n_estimators=Constants.N_ESTIMATORS, max_features=Constants.MAX_FEATURES,
                                   min_samples_leaf=Constants.MIN_SAMPLES_LEAF)
    print('krenuo sam')
    sfs = SequentialFeatureSelector(model, direction='forward', scoring='accuracy', n_jobs=2, cv=loo_data)
    sfs.fit(X_train, y_train)
    print(sfs.get_feature_names_out())





