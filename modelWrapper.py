import configparser
from statistics import mean

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.ensemble import RandomForestClassifier
from mlxtend.feature_selection import SequentialFeatureSelector
from sklearn.model_selection import LeaveOneOut, train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import confusion_matrix, plot_confusion_matrix, \
    ConfusionMatrixDisplay, classification_report, roc_curve, roc_auc_score, precision_score, recall_score, f1_score
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from numpy import sqrt
from constants import Constants
from time import process_time
from datetime import timedelta
import seaborn as sns
from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs

config = configparser.ConfigParser()
config.read('config.ini')
config = config['default']
filter_file = config['output_location-filter']


def train_model_catalytic():
    filter_data_file = pd.read_csv(filepath_or_buffer=filter_file, delimiter=',')

    all_data_feature = filter_data_file.drop(labels=['FASTA form', 'SMILE form', 'result'], axis=1)
    target = filter_data_file["result"]

    forward_selection(all_data_feature, target, Constants.CATALYTIC_FORWARD_NAME)
    #backward_selection(all_data_feature, target, Constants.CATALYTIC_BACKWARD_NAME)
    return


def train_model_amp():
    filter_data_file = pd.read_csv(filepath_or_buffer=filter_file, delimiter=',')

    all_data_feature = filter_data_file.drop(labels=['FASTA form', 'SMILE form', 'result'], axis=1)
    target = filter_data_file["result"]

    #forward_selection(all_data_feature, target, Constants.AMP_FORWARD_NAME)
    backward_selection(all_data_feature, target, Constants.AMP_BACKWARD_NAME)
    return


def forward_selection(all_data_feature, target, name):
    X_train, X_test, y_train, y_test = train_test_split(all_data_feature, target, test_size=0.3, random_state=20,
                                                        shuffle=True)

    model_feature_selection = DecisionTreeClassifier(max_depth=9, random_state=20)

    model = RandomForestClassifier(n_estimators=Constants.N_ESTIMATORS, max_features=Constants.MAX_FEATURES,
                                   min_samples_leaf=Constants.MIN_SAMPLES_LEAF)

    sfs = SequentialFeatureSelector(estimator=model_feature_selection, k_features=100, forward=True,
                                    floating=False, scoring='accuracy', cv=2, verbose=2, n_jobs=4)

    start = process_time()
    sfs.fit(X_train, y_train)
    end = process_time()

    selected_feature_names = list(sfs.k_feature_names_)
    selected_feature_index = list(sfs.k_feature_idx_)
    X_train = X_train.iloc[:, selected_feature_index]
    X_test = X_test.iloc[:, selected_feature_index]

    print(selected_feature_index)
    print("Number of features: " + str(len(selected_feature_names)))
    print(f'Time : {timedelta(seconds=end - start)}')

    model.fit(X_train, y_train)
    prediction_results = model.predict(X_test)
    probability_target_positive = model.predict_proba(X_test)[:, 1]

    target_results = y_test
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

    roc_auc_curve_display(probability_target_positive, target_results, name)
    matrix_display(confusion_matrix_values, name)
    plot_feature_selection(sfs, len(selected_feature_names), name)
    feature_importance(model.feature_importances_, selected_feature_names, name)
    accuracy_score_display(target_results, probability_target_positive, name)
    return


def backward_selection(all_data_feature, target, name):
    X_train, X_test, y_train, y_test = train_test_split(all_data_feature, target, test_size=0.3, random_state=20,
                                                        shuffle=True)

    model_feature_selection = DecisionTreeClassifier(max_depth=9, random_state=20)
    model = RandomForestClassifier(n_estimators=Constants.N_ESTIMATORS, max_features=Constants.MAX_FEATURES,
                                   min_samples_leaf=Constants.MIN_SAMPLES_LEAF)

    sfs = SequentialFeatureSelector(estimator=model_feature_selection, k_features=250, forward=False, floating=False,
                                    scoring='accuracy', cv=3, verbose=2, n_jobs=4)

    start = process_time()
    sfs.fit(X_train, y_train)
    end = process_time()

    selected_feature_names = list(sfs.k_feature_names_)
    selected_feature_index = list(sfs.k_feature_idx_)
    X_train = X_train.iloc[:, selected_feature_index]
    X_test = X_test.iloc[:, selected_feature_index]
    print(selected_feature_names)
    print(selected_feature_index)
    print(X_train.shape)
    print(X_test.shape)
    print("Number of features: " + str(len(selected_feature_names)))
    print(f'Time : {timedelta(seconds=end - start)}')

    model.fit(X_train, y_train)
    prediction_results = model.predict(X_test)
    probability_target_positive = model.predict_proba(X_test)[:, 1]

    target_results = y_test
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

    roc_auc_curve_display(probability_target_positive, target_results, name)
    matrix_display(confusion_matrix_values, name)
    plot_feature_selection(sfs, len(selected_feature_names), name)
    feature_importance(model.feature_importances_, selected_feature_names, name)
    accuracy_score_display(target_results, probability_target_positive, name)
    return


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


def matrix_display(confusion_matrix_values, name):
    plot_matrix = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix_values, display_labels=['False', 'True'])
    plot_matrix.plot()
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.savefig('Confusion_matrix-{}.png'.format(name))
    plt.close()
    return


def plot_feature_selection(sfs_model, length, name):
    plot_sfs(metric_dict=sfs_model.get_metric_dict(), kind='std_dev', figsize=(6, 4))
    plt.ylim([0.5, 1])
    plt.xticks(np.arange(0, length, 5))
    plt.title('Sequential Forward Selection - standard deviation')
    plt.grid()
    plt.savefig('Sequential-selection-{}.png'.format(name))
    plt.close()
    return


def feature_importance(feature_values_importances, columns_name, name):
    print(feature_values_importances)
    print(columns_name)
    data = {'feature_names': columns_name, 'feature_importance': feature_values_importances}
    data_important_feature = pd.DataFrame(data)

    data_important_feature.sort_values(by=['feature_importance'], ascending=False, inplace=True)
    data_important_feature = data_important_feature.iloc[:40]
    plt.figure(figsize=(10, 8))

    sns.barplot(x=data_important_feature['feature_importance'], y=data_important_feature['feature_names'], color='blue')
    plt.title('Feature importance in Random Forest')
    plt.xlabel('Importance')
    plt.savefig('Feature-importances-{}.png'.format(name))
    plt.close()
    return


def accuracy_score_display(target_results, probability_target_positive, name):
    plt.figure(figsize=(10, 8))
    fop, mpv = calibration_curve(target_results, probability_target_positive, n_bins=30, normalize=True)
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.plot(mpv, fop, marker='.')
    plt.title('Probability calibration Random Forest')
    plt.savefig('Calibration-{}.png'.format(name))
    plt.close()
    return

"""
def histogram_probabilities(probability_target, name):
    plt.figure(1)
    plt.hist(probability_target, bins=76, edgecolor="black")
    plt.xlim(0, 1)
    plt.title('Histogram of predicted probabilities')
    plt.xlabel('Predicted probability of active peptide')
    plt.ylabel('Frequency')
    plt.savefig('Histogram-probabilities-forward-{}.png'.format(name))
    return


feature_set = []
for number_feature in range(100):
    model = RandomForestClassifier(n_estimators=Constants.N_ESTIMATORS, max_features=Constants.MAX_FEATURES,
                                min_samples_leaf=Constants.MIN_SAMPLES_LEAF)

    metric_list = []

    for feature in all_feature:
        if feature not in feature_set:
            f_set = feature_set.copy()
            f_set.append(feature)

            #model.fit(X_train[f_set], y_train)
            metric_list.append((mean(cross_val_score(model, X_train[f_set], y_train, cv=ten_fold_cv, scoring='accuracy')),
                               feature))
            print(metric_list)

    metric_list.sort(key=lambda x: x[0], reverse=True)
    print(metric_list[0][1])
    feature_set.append(metric_list[0][1])
print("ovo je kraj")
print(feature_set)
"""



