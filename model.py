import configparser
import numpy as np
import pandas as pd
from sklearn.model_selection import LeaveOneOut
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from numpy import mean
from numpy import std
import joblib

config = configparser.ConfigParser()
config.read('config.ini')
config = config['default']

filter_file = config['output_location-filter']

def train_model():
    filter_data_file = pd.read_csv(filepath_or_buffer=filter_file, delimiter=',')

    all_data_feature = filter_data_file.drop(labels=['FASTA form', 'SMILE form', 'result'], axis=1)
    all_data_feature = all_data_feature.values
    target = filter_data_file["result"].values

    model = RandomForestClassifier(n_estimators=100)
    loo_data = LeaveOneOut()

    scores = cross_val_score(model, all_data_feature, target, scoring="accuracy", cv=loo_data)

    print('Accuracy: ' + str(mean(scores)))

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