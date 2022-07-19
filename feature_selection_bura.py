from statistics import mean
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.naive_bayes import GaussianNB
from time import process_time, time
from datetime import timedelta
from operator import itemgetter
import multiprocessing
import configparser

config = configparser.ConfigParser()
config.read('config.ini')
config = config['default']
filter_file = config['output_location-filter']

filter_data_file_global = pd.read_csv(filepath_or_buffer=filter_file, delimiter=',')


def catalytic_function():
    filter_data_file = pd.read_csv(filepath_or_buffer=filter_file, delimiter=',')

    all_data_feature = filter_data_file.drop(labels=['FASTA form', 'SMILE form', 'result'], axis=1)
    forward_selection(all_data_feature, 'catalytic_forward')
    return


def forward_selection(all_data_feature, name):
    train_feature_metrics = []
    feature_score = []
    feature_name = []
    feature_subset = []

    tmp_all_data_feature = all_data_feature

    start_time = time()
    start = process_time()

    pool = multiprocessing.Pool(processes=48)

    while len(feature_name) < len(all_data_feature.columns):
        train_feature_metrics.clear()

        train_feature_metrics = pool.starmap(evaluate_model,
                                             [(each_feature, feature_subset)
                                              for each_feature in tmp_all_data_feature.columns])

        best_result = max(train_feature_metrics, key=itemgetter(1))
        feature_name.append(best_result[0])
        feature_score.append(best_result[1])

        feature_subset.append(best_result[0])

        tmp_all_data_feature = tmp_all_data_feature.drop(best_result[0], axis=1)
        print(str(best_result[0]) + ' ' + str(best_result[1]) + ' ' + str(len(feature_name)))

    end = process_time()
    end_time = time()

    print(end - start)
    print(f'Time CPU for forward feature selection: {timedelta(seconds=end - start)}')
    print(f'Time for forward feature selection: {timedelta(seconds=end_time - start_time)}')

    plot_feature_score(feature_score[0:250], name)
    plot_feature_score_all(feature_score, name)
    return


# create graph for feature selection in definition range
def plot_feature_score(feature_score, name):
    plt.figure(figsize=(20, 10), dpi=150)
    plt.ylabel('Mean F1 score')
    plt.xlabel('Number of features')
    plt.title('Feature importance')

    x_range = np.arange(1, len(feature_score) + 1, 1)
    plt.xticks(np.arange(1, len(feature_score), 10))
    plt.plot(x_range, feature_score, color='r', linewidth=1, label='Cross-validation score')
    plt.legend()
    plt.savefig('Mean f1 score-{}.png'.format(name))
    plt.close()
    return


# create graph for feature selection in definition range
def plot_feature_score_all(feature_score, name):
    plt.figure(figsize=(20, 10), dpi=150)
    plt.ylabel('Mean F1 score')
    plt.xlabel('Number of features')
    plt.title('Feature importance')

    x_range = np.arange(1, len(feature_score) + 1, 1)
    plt.xticks(np.arange(1, len(feature_score), 43))
    plt.plot(x_range, feature_score, color='r', linewidth=1, label='Cross-validation score')
    plt.legend()
    plt.savefig('Mean f1 score-all-{}.png'.format(name))
    plt.close()
    return


def evaluate_model(feature, list_feature):
    ten_fold_cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=10)
    f1_average = []

    target = filter_data_file_global["result"]
    selected_feature = list_feature.copy()
    selected_feature.append(feature)
    all_data_feature = filter_data_file_global.loc[:, selected_feature]

    model_feature_selection = GaussianNB()

    for train_index, test_index in ten_fold_cv.split(all_data_feature, target):
        X_train, X_test = all_data_feature.iloc[train_index, :], all_data_feature.iloc[test_index, :]
        y_train, y_test = target.iloc[train_index], target.iloc[test_index]

        model_feature_selection.fit(X_train, y_train)

        all_prediction = model_feature_selection.predict(X_test)
        f1_result = f1_score(y_test, all_prediction)
        f1_average.append(f1_result)

    f1_result_cv = mean(f1_average)
    return feature, f1_result_cv
