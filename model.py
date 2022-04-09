import configparser
import pandas as pd
from sklearn.model_selection import LeaveOneOut
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from constants import Constants
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
import joblib

config = configparser.ConfigParser()
config.read('config.ini')
config = config['default']

filter_file = config['output_location-filter']


def train_model_catalytic():
    filter_data_file = pd.read_csv(filepath_or_buffer=filter_file, delimiter=',')

    all_data_feature = filter_data_file.drop(labels=['FASTA form', 'SMILE form', 'result'], axis=1)
    target = filter_data_file["result"]

    model = RandomForestClassifier(n_estimators=Constants.N_ESTIMATORS, max_features=Constants.MAX_FEATURES, min_samples_leaf=Constants.MIN_SAMPLES_LEAF)
    loo_data = LeaveOneOut()

    scores = cross_val_score(model, all_data_feature, target, scoring="accuracy", cv=loo_data).mean()

    print('Accuracy: ' + str(scores))

    """
    tmp_results = []
    target_results = []

    for train_index, test_index in loo_data.split(all_data_feature):
        X_train, X_test = all_data_feature[train_index, :], all_data_feature[test_index, :]
        y_train, y_test = target[train_index], target[test_index]

        model = RandomForestClassifier(n_estimators=100)
        model.fit(X_train, y_train)
        all_prediction = model.predict(X_test)

        tmp_results.append(y_test[0])
        target_results.append(all_prediction)

    acc = accuracy_score(tmp_results, target_results)
    print('Accuracy: %f' %acc)
    """


def train_model_amp():
    filter_data_file = pd.read_csv(filepath_or_buffer=filter_file, delimiter=',')

    all_data_feature = filter_data_file.drop(columns=['FASTA form', 'SMILE form', 'result'], axis=1)
    target = filter_data_file["result"]

    model = RandomForestClassifier(n_estimators=Constants.N_ESTIMATORS, max_features=Constants.MAX_FEATURES,
                                   min_samples_leaf=Constants.MIN_SAMPLES_LEAF)
    ten_fold_cv = KFold(n_splits=10, shuffle=True, random_state=None)

    scores = cross_val_score(model, all_data_feature, target, scoring="accuracy", cv=ten_fold_cv).mean()

    print('Accuracy: ' + str(scores))

    """
    x_train, x_test, y_train, y_test = train_test_split(all_data_feature, target, test_size=0.25, shuffle=True)
    model = RandomForestClassifier(n_estimators=Constants.N_ESTIMATORS, max_features=Constants.MAX_FEATURES,
                                   min_samples_leaf=Constants.MIN_SAMPLES_LEAF)
    model_random_forest = model.fit(x_train, y_train)
    ten_fold = KFold(n_splits=10)

    scores = cross_val_score(model_random_forest, x_train, y_train, scoring="accuracy", cv=ten_fold).mean()
    print(scores)

    scores1 = cross_val_score(model_random_forest, x_test, y_test, scoring="accuracy", cv=ten_fold).mean()
    print(scores1)
    -- 92 % na train
    -- 89 % na test
    """

