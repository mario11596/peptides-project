import configparser
from statistics import mean
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneOut, StratifiedKFold
from sklearn.metrics import confusion_matrix, \
    ConfusionMatrixDisplay, classification_report, roc_curve, roc_auc_score, precision_score, recall_score, f1_score
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from numpy import sqrt
from sklearn.tree import DecisionTreeClassifier
import selectedDataWrapper as dataFile
from constants import Constants
import selectedDataWrapper
from time import time
from datetime import timedelta
import seaborn as sns
from operator import itemgetter
import multiprocessing

config = configparser.ConfigParser()
config.read('config.ini')
config = config['default']
filter_file = config['output_location-filter']

# global dataframe for faster evaluate_model() method
filter_data_file_global = pd.read_csv(filepath_or_buffer=filter_file, delimiter=',')


# feature selections for catalytic peptides
def catalytic_function():
    filter_data_file = pd.read_csv(filepath_or_buffer=filter_file, delimiter=',')

    all_data_feature = filter_data_file.drop(labels=['FASTA form', 'SMILE form', 'result'], axis=1)
    target = filter_data_file["result"]

    # methods for calculate the best features using forward and backward techniques
    forward_selection(all_data_feature, target,  Constants.CATALYTIC_FORWARD_NAME)
    backward_selection(all_data_feature, target, Constants.CATALYTIC_BACKWARD_NAME)

    # previous selected the best features
    new_forward_data_feature = all_data_feature.loc[:, all_data_feature.columns.isin(
        selectedDataWrapper.catalytic_forward_dataset_new)]
    new_backward_data_feature = all_data_feature.loc[:, all_data_feature.columns.isin(
        selectedDataWrapper.catalytic_backward_dataset_new)]

    train_model_catalytic(new_forward_data_feature, target,  Constants.CATALYTIC_FORWARD_NAME)
    train_model_catalytic(new_backward_data_feature, target, Constants.CATALYTIC_BACKWARD_NAME)
    return


# feature selection for AMP (DRAMP 2) peptides
def amp_function():
    filter_data_file = pd.read_csv(filepath_or_buffer=filter_file, delimiter=',')

    all_data_feature = filter_data_file.drop(labels=['FASTA form', 'SMILE form', 'result'], axis=1)
    target = filter_data_file["result"]

    # methods for calculate the best features using forward and backward techniques
    forward_selection(all_data_feature, target, Constants.AMP_FORWARD_NAME)
    backward_selection(all_data_feature, target,  Constants.AMP_BACKWARD_NAME)

    # previous selected the best features
    new_forward_data_feature = all_data_feature.loc[:, all_data_feature.columns.isin(
        selectedDataWrapper.amp_forward_dataset)]
    new_backward_data_feature = all_data_feature.loc[:, all_data_feature.columns.isin(
        selectedDataWrapper.amp_backward_dataset)]

    train_model_amp(new_forward_data_feature, target, Constants.AMP_FORWARD_NAME)
    train_model_amp(new_backward_data_feature, target, Constants.AMP_BACKWARD_NAME)
    return


# forward feature selection
def forward_selection(all_data_feature, target, name):
    train_feature_metrics = []
    feature_score = []
    feature_name = []
    feature_subset = []

    # temporary variable for algorithm optimization
    tmp_all_data_feature = all_data_feature

    start_time = time()

    # number of processes for usage HPC
    pool = multiprocessing.Pool(processes=48)

    # termination condition
    while len(feature_name) < len(all_data_feature.columns):
        train_feature_metrics.clear()

        train_feature_metrics = pool.starmap(evaluate_model_forward,
                                             [(each_feature, feature_subset)
                                              for each_feature in tmp_all_data_feature.columns])

        best_result = max(train_feature_metrics, key=itemgetter(1))
        feature_name.append(best_result[0])
        feature_score.append(best_result[1])

        feature_subset.append(best_result[0])

        tmp_all_data_feature = tmp_all_data_feature.drop(best_result[0], axis=1)
        print(str(best_result[0]) + ' ' + str(best_result[1]) + ' ' + str(len(feature_name)))

    end_time = time()
    print(f'Time for forward feature selection: {timedelta(seconds=end_time - start_time)}')
    new_data_feature = all_data_feature.loc[:, dataFile.amp_forward_dataset]

    # plot F1 scores for first 250 and all features
    plot_feature_score(feature_score[0:250], name)
    plot_feature_score_all(feature_score, name)
    train_model_catalytic(new_data_feature, target, name)
    return


# backward feature selection
def backward_selection(all_data_feature, target, name):
    feature_subset = all_data_feature.columns
    train_feature_metrics = []
    feature_drop = []
    feature_score = []

    # temporary variable for algorithm optimization
    tmp_all_data_feature = all_data_feature

    start_time = time()

    # number of processes for usage HPC
    pool = multiprocessing.Pool(processes=48)

    # termination condition
    count = len(all_data_feature.columns)

    while count > 1:
        train_feature_metrics.clear()
        train_feature_metrics = pool.starmap(evaluate_model_backward,
                                             [(each_feature, feature_subset)
                                              for each_feature in tmp_all_data_feature.columns])

        count -= 1
        best_result = max(train_feature_metrics, key=itemgetter(1))
        feature_subset = feature_subset.drop(best_result[0])
        feature_drop.append(best_result[0])
        feature_score.append(best_result[1])
        tmp_all_data_feature = tmp_all_data_feature.drop(best_result[0], axis=1)
        print(str(best_result[0]) + ' ' + str(best_result[1]) + ' ' + str(len(tmp_all_data_feature.columns)))

    end_time = time()
    print(f'Time for forward feature selection: {timedelta(seconds=end_time - start_time)}')
    new_data_feature = all_data_feature.loc[:, ~all_data_feature.columns.isin(feature_drop)]

    # plot F1 scores for last 250 and all features
    plot_feature_score_all(feature_score, name)

    #last number depends about dataset
    plot_feature_score(feature_score[900:1150], name)
    train_model_catalytic(new_data_feature, target, name)
    return


# machine learning for catalytic peptides
def train_model_catalytic(new_data_feature, target, name):
    all_data_feature = new_data_feature
    print(new_data_feature.shape)

    loo_data = LeaveOneOut()
    prediction_results = []
    target_results = []
    probability_target_positive = []

    model = RandomForestClassifier(n_estimators=Constants.N_ESTIMATORS, max_features=Constants.MAX_FEATURES,
                                   min_samples_leaf=Constants.MIN_SAMPLES_LEAF, random_state=50)

    start_time = time()

    for train_index, test_index in loo_data.split(all_data_feature):
        X_train, X_test = all_data_feature.iloc[train_index, :], all_data_feature.iloc[test_index, :]
        y_train, y_test = target.iloc[train_index], target.iloc[test_index]

        model.fit(X_train, y_train)
        all_prediction = model.predict(X_test)
        prediction_results.extend(all_prediction)
        target_results.extend(y_test)

        # the predicted probabilities for positive
        probability_target_positive.extend(model.predict_proba(X_test)[:, 1])

    confusion_matrix_values = confusion_matrix(target_results, prediction_results)
    accuracy_result = accuracy_score(target_results, prediction_results)
    precision_result = precision_score(target_results, prediction_results)
    recall_result = recall_score(target_results, prediction_results)
    f1_result = f1_score(target_results, prediction_results)
    g_mean_result = sqrt(precision_result * recall_result)

    print('Accuracy: %f' % accuracy_result)
    print('Geometric mean score: %f' % g_mean_result)
    print('Precision score: %f' % precision_result)
    print('Recall score: %f' % recall_result)
    print('F1 score: %f' % f1_result)

    print(classification_report(target_results, prediction_results, labels=[0, 1]))

    end_time = time()

    print(f'Time for for catalytic model: {timedelta(seconds=end_time - start_time)}')

    roc_auc_curve_display(probability_target_positive, target_results, name)
    matrix_display(confusion_matrix_values, name)
    feature_importance(model.feature_importances_, all_data_feature.columns, name)
    return


# machine learning for AMP (DRAMP 2) peptides
def train_model_amp(new_data_feature, target, name):
    all_data_feature = new_data_feature

    ten_fold_cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=10)
    prediction_results = []
    target_results = []
    probability_target_positive = []

    model = RandomForestClassifier(n_estimators=Constants.N_ESTIMATORS, max_features=Constants.MAX_FEATURES,
                                   min_samples_leaf=Constants.MIN_SAMPLES_LEAF, random_state=50)

    start_time = time()

    for train_index, test_index in ten_fold_cv.split(all_data_feature, target):
        X_train, X_test = all_data_feature.iloc[train_index, :], all_data_feature.iloc[test_index, :]
        y_train, y_test = target.iloc[train_index], target.iloc[test_index]

        model.fit(X_train, y_train)

        all_prediction = model.predict(X_test)
        prediction_results.extend(all_prediction)
        target_results.extend(y_test)

        probability_target_positive.extend(model.predict_proba(X_test)[:, 1])

    end_time = time()
    confusion_matrix_values = confusion_matrix(target_results, prediction_results)
    accuracy_result = accuracy_score(target_results, prediction_results)
    precision_result = precision_score(target_results, prediction_results)
    recall_result = recall_score(target_results, prediction_results)
    f1_result = f1_score(target_results, prediction_results)
    g_mean_result = sqrt(precision_result * recall_result)

    print('Geometric mean score: %f' % g_mean_result)
    print('Accuracy: %f' % accuracy_result)
    print('Precision score: %f' % precision_result)
    print('Recall score: %f' % recall_result)
    print('F1 score: %f' % f1_result)
    print(f'Time for for catalytic model: {timedelta(seconds=end_time - start_time)}')
    print(classification_report(target_results, prediction_results, labels=[0, 1]))

    roc_auc_curve_display(probability_target_positive, target_results, name)
    matrix_display(confusion_matrix_values, name)
    feature_importance(model.feature_importances_, all_data_feature.columns, name)
    return


# calculate ROC-AUC score and display curve
def roc_auc_curve_display(probability_target_positive, target_results, name):
    fpr_poz, tpr_poz, thresholds_poz = roc_curve(target_results, probability_target_positive)
    auc_rez_poz = roc_auc_score(target_results, probability_target_positive)

    plt.plot(fpr_poz, tpr_poz, color='orange', label='Positive (ROC-AUC = %f' % auc_rez_poz)
    plt.plot([0, 1], [0, 1], color="blue", lw=2, linestyle="--")

    plt.legend(loc=4)
    plt.grid(color='b', ls='-.', lw=0.25)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.savefig('AUC-ROC_curve-{}.png'.format(name))
    plt.close()
    return


# create confusion matrix
def matrix_display(confusion_matrix_values, name):
    plot_matrix = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix_values, display_labels=['False', 'True'])
    plot_matrix.plot()
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.savefig('Confusion_matrix-{}.png'.format(name))
    plt.close()
    return


# create graphs with features importance in decreasing order
def feature_importance(feature_values_importances, columns_name, name):
    data = {'feature_names': columns_name, 'feature_importance': feature_values_importances}
    data_important_feature = pd.DataFrame(data)

    data_important_feature.sort_values(by=['feature_importance'], ascending=False, inplace=True)
    data_important_feature = data_important_feature.iloc[:40]
    plt.figure(figsize=(10, 8))

    sns.barplot(x=data_important_feature['feature_importance'], y=data_important_feature['feature_names'], color='blue')
    plt.title('Feature importance in Random Forest')
    plt.ylabel('Feature name')
    plt.xlabel('Importance')
    plt.savefig('Feature-importances-{}.png'.format(name))
    plt.close()
    return


# create graph for feature selection in definition range
def plot_feature_score(feature_score, name):
    plt.figure(figsize=(20, 10), dpi=150)
    plt.ylabel('Mean F1 score')
    plt.xlabel('Number of features')
    plt.title('Feature importance')

    x_range = np.arange(1, len(feature_score) + 1, 1)
    plt.xticks(np.arange(1, len(feature_score), 10))
    plt.plot(x_range, feature_score, color='r', linewidth=1, label='Cross validation score')
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
    plt.xticks(np.arange(1, len(feature_score), 46))
    plt.plot(x_range, feature_score, color='r', linewidth=1, label='Cross validation score')
    plt.legend()
    plt.savefig('Mean f1 score-all-{}.png'.format(name))
    plt.close()
    return


# evaluate model using Gaussian Naive Bayes for feature selection after each add feature
def evaluate_model_forward(feature, list_feature):
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


# evaluate model using Decision Tree for feature selection after each drop feature
def evaluate_model_backward(feature, list_feature):
    ten_fold_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=10)
    f1_average = []

    target = filter_data_file_global["result"]
    selected_feature = list_feature.copy()
    selected_feature = selected_feature.drop(feature)
    all_data_feature = filter_data_file_global.loc[:, selected_feature]

    model_feature_selection = DecisionTreeClassifier(max_depth=5, random_state=20)

    for train_index, test_index in ten_fold_cv.split(all_data_feature, target):
        X_train, X_test = all_data_feature.iloc[train_index, :], all_data_feature.iloc[test_index, :]
        y_train, y_test = target.iloc[train_index], target.iloc[test_index]

        model_feature_selection.fit(X_train, y_train)

        all_prediction = model_feature_selection.predict(X_test)
        f1_result = f1_score(y_test, all_prediction)
        f1_average.append(f1_result)

    f1_result_cv = mean(f1_average)
    return feature, f1_result_cv
