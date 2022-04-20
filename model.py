import configparser
from math import sqrt
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, mean_squared_error, confusion_matrix, plot_confusion_matrix, \
    ConfusionMatrixDisplay, classification_report, roc_curve, roc_auc_score, precision_score, recall_score
from sklearn.model_selection import LeaveOneOut, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from constants import Constants


config = configparser.ConfigParser()
config.read('config.ini')
config = config['default']
filter_file = config['output_location-filter']


def train_model_catalytic():
    filter_data_file = pd.read_csv(filepath_or_buffer=filter_file, delimiter=',')

    all_data_feature = filter_data_file.drop(labels=['FASTA form', 'SMILE form', 'result'], axis=1)
    target = filter_data_file["result"]
    """
    model = RandomForestClassifier(n_estimators=Constants.N_ESTIMATORS, max_features=Constants.MAX_FEATURES,
                                   min_samples_leaf=Constants.MIN_SAMPLES_LEAF)
    loo_data = LeaveOneOut()

    scores = cross_val_predict(model, all_data_feature, target, cv=loo_data)
    acc = accuracy_score(target, scores)
    cm = confusion_matrix(target, scores)
    print('Accuracy: ' + str(scores))
    print(acc)

    print(cm)
    """

    loo_data = LeaveOneOut()
    prediction_results = []
    target_results = []
    probability_target = []

    for train_index, test_index in loo_data.split(all_data_feature):
        X_train, X_test = all_data_feature.iloc[train_index, :], all_data_feature.iloc[test_index, :]
        y_train, y_test = target.iloc[train_index], target.iloc[test_index]
        model = RandomForestClassifier(n_estimators=Constants.N_ESTIMATORS, max_features=Constants.MAX_FEATURES,
                                       min_samples_leaf=Constants.MIN_SAMPLES_LEAF)
        model.fit(X_train, y_train)
        all_prediction = model.predict(X_test)
        prediction_results.extend(all_prediction)
        target_results.extend(y_test)

        probability_target.extend(model.predict_proba(X_test)[:, 1])

    confusion_matrix_values = confusion_matrix(target_results, prediction_results)
    accuracy_result = accuracy_score(target_results, prediction_results)
    mean_squad_error_result = mean_squared_error(target_results, prediction_results)
    precision_result = precision_score(target_results, prediction_results)
    recall_result = recall_score(target_results, prediction_results)
    g_mean_result = sqrt(precision_result*recall_result)

    print('Accuracy: %f' % accuracy_result)
    print('Mean squared error: %f' % mean_squad_error_result)
    print('Geometric mean score: %f' % g_mean_result)
    print(classification_report(target_results, prediction_results, labels=[0, 1]))
    print(probability_target)

    roc_auc_curve_display(probability_target, target_results, Constants.CATALYTIC_NAME)
    matrix_display(confusion_matrix_values, Constants.CATALYTIC_NAME)


def train_model_amp():
    filter_data_file = pd.read_csv(filepath_or_buffer=filter_file, delimiter=',')

    all_data_feature = filter_data_file.drop(columns=['FASTA form', 'SMILE form', 'result'], axis=1)

    target = filter_data_file["result"]

    """
    model = RandomForestClassifier(n_estimators=Constants.N_ESTIMATORS, max_features=Constants.MAX_FEATURES,
                                   min_samples_leaf=Constants.MIN_SAMPLES_LEAF)
    ten_fold_cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=None)

    scores = cross_val_score(model, all_data_feature, target, scoring="accuracy", cv=ten_fold_cv).mean()

    print('Accuracy: ' + str(scores))
    """

    ten_fold_cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=None)
    prediction_results = []
    target_results = []
    probability_target = []

    for train_index, test_index in ten_fold_cv.split(all_data_feature, target):
        X_train, X_test = all_data_feature.iloc[train_index, :], all_data_feature.iloc[test_index, :]
        y_train, y_test = target.iloc[train_index], target.iloc[test_index]

        model = RandomForestClassifier(n_estimators=Constants.N_ESTIMATORS, max_features=Constants.MAX_FEATURES,
                                       min_samples_leaf=Constants.MIN_SAMPLES_LEAF)
        model.fit(X_train, y_train)
        all_prediction = model.predict(X_test)
        prediction_results.extend(all_prediction)
        target_results.extend(y_test)

        probability_target.extend(model.predict_proba(X_test)[:, 1])

    confusion_matrix_values = confusion_matrix(target_results, prediction_results)
    accuracy_result = accuracy_score(target_results, prediction_results)
    mean_squad_error_result = mean_squared_error(target_results, prediction_results)
    precision_result = precision_score(target_results, prediction_results)
    recall_result = recall_score(target_results, prediction_results)
    g_mean_result = sqrt(precision_result * recall_result)

    print('Geometric mean score: %f' % g_mean_result)
    print('Accuracy: %f' % accuracy_result)
    print('Mean squared error: %f' % mean_squad_error_result)
    print(classification_report(target_results, prediction_results, labels=[0, 1]))
    print(probability_target)

    roc_auc_curve_display(probability_target, target_results, Constants.AMP_NAME)
    matrix_display(confusion_matrix_values, Constants.AMP_NAME)


def roc_auc_curve_display(probability_target, target_results, name):
    fpr, tpr, thresholds = roc_curve(target_results, probability_target)
    auc_rez = roc_auc_score(target_results, probability_target)

    plt.plot(fpr, tpr, color='orange', label='Random Forest Classifier (auc = %f' % auc_rez)
    plt.plot([0, 1], [0, 1], color="blue", lw=2, linestyle="--")
    plt.legend(loc=4)
    plt.grid(color='b', ls='-.', lw=0.25)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.savefig('AUC-ROC_curve-{}.png'.format(name))
    return


def matrix_display(confusion_matrix_values, name):
    plot_matrix = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix_values, display_labels=['False', 'True'])
    plot_matrix.plot()
    plt.xlabel('Predicted')
    plt.ylabel('Truth')
    plt.savefig('Confusion_matrix-{}.png'.format(name))
    return


def histogram_probabilities(probability_target, name):
    plt.figure(1)
    plt.hist(probability_target, bins=76, edgecolor="black")
    plt.xlim(0, 1)
    plt.title('Histogram of predicted probabilities')
    plt.xlabel('Predicted probability of active peptide')
    plt.ylabel('Frequency')
    plt.savefig('Histogram-probabilities-{}.png'.format(name))
    return




