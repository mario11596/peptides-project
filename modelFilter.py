import configparser
from math import sqrt
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, mean_squared_error, confusion_matrix, f1_score, \
    ConfusionMatrixDisplay, classification_report, roc_curve, roc_auc_score, precision_score, recall_score
from sklearn.model_selection import LeaveOneOut, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from constants import Constants
from time import process_time
from datetime import timedelta
import seaborn as sns


config = configparser.ConfigParser()
config.read('config.ini')
config = config['default']
filter_file = config['output_location-filter']


def train_model_catalytic():
    filter_data_file = pd.read_csv(filepath_or_buffer=filter_file, delimiter=',')
    all_data_feature = filter_data_file.drop(labels=['FASTA form', 'SMILE form', 'result'], axis=1)
    target = filter_data_file["result"]

    loo_data = LeaveOneOut()
    prediction_results = []
    target_results = []
    probability_target_positive = []
    probability_target_negative = []

    model = RandomForestClassifier(n_estimators=Constants.N_ESTIMATORS, max_features=Constants.MAX_FEATURES,
                                   min_samples_leaf=Constants.MIN_SAMPLES_LEAF, random_state=50)
    start = process_time()
    for train_index, test_index in loo_data.split(all_data_feature):
        X_train, X_test = all_data_feature.iloc[train_index, :], all_data_feature.iloc[test_index, :]
        y_train, y_test = target.iloc[train_index], target.iloc[test_index]
        #model = RandomForestClassifier(n_estimators=Constants.N_ESTIMATORS, max_features=Constants.MAX_FEATURES,
                                       #min_samples_leaf=Constants.MIN_SAMPLES_LEAF)
        model.fit(X_train, y_train)
        all_prediction = model.predict(X_test)
        prediction_results.extend(all_prediction)
        target_results.extend(y_test)

        # the predicted probabilities for class 1
        probability_target_positive.extend(model.predict_proba(X_test)[:, 1])
        # the predicted probabilities for class 0
        probability_target_negative.extend(model.predict_proba(X_test)[:, 0])

    end = process_time()
    confusion_matrix_values = confusion_matrix(target_results, prediction_results)
    accuracy_result = accuracy_score(target_results, prediction_results)
    precision_result = precision_score(target_results, prediction_results)
    recall_result = recall_score(target_results, prediction_results)
    f1_result = f1_score(target_results, prediction_results)
    g_mean_result = sqrt(precision_result*recall_result)

    print('Accuracy: %f' % accuracy_result)
    print('Geometric mean score: %f' % g_mean_result)
    print('Precision score: %f' % precision_result)
    print('Recall score: %f' % recall_result)
    print('F1 score: %f' % f1_result)
    print(f'Time : {timedelta(seconds=end - start)}')
    print(classification_report(target_results, prediction_results, labels=[0, 1]))

    roc_auc_curve_display(probability_target_positive, probability_target_negative,
                          target_results, Constants.CATALYTIC_NAME)
    matrix_display(confusion_matrix_values, Constants.CATALYTIC_NAME)
    feature_importance(model.feature_importances_, all_data_feature.columns, Constants.CATALYTIC_NAME)
    return


def train_model_amp():
    filter_data_file = pd.read_csv(filepath_or_buffer=filter_file, delimiter=',')
    all_data_feature = filter_data_file.drop(columns=['FASTA form', 'SMILE form', 'result'], axis=1)
    target = filter_data_file["result"]

    ten_fold_cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=None)
    prediction_results = []
    target_results = []
    probability_target_positive = []
    probability_target_negative = []

    model = RandomForestClassifier(n_estimators=Constants.N_ESTIMATORS, max_features=Constants.MAX_FEATURES,
                                   min_samples_leaf=Constants.MIN_SAMPLES_LEAF, random_state=50)

    start = process_time()
    for train_index, test_index in ten_fold_cv.split(all_data_feature, target):
        X_train, X_test = all_data_feature.iloc[train_index, :], all_data_feature.iloc[test_index, :]
        y_train, y_test = target.iloc[train_index], target.iloc[test_index]

        #model = RandomForestClassifier(n_estimators=Constants.N_ESTIMATORS, max_features=Constants.MAX_FEATURES,
                                       #min_samples_leaf=Constants.MIN_SAMPLES_LEAF, random_state=50)
        model.fit(X_train, y_train)
        all_prediction = model.predict(X_test)
        prediction_results.extend(all_prediction)
        target_results.extend(y_test)

        # the predicted probabilities for class 1
        probability_target_positive.extend(model.predict_proba(X_test)[:, 1])
        # the predicted probabilities for class 0
        probability_target_negative.extend(model.predict_proba(X_test)[:, 0])

    end = process_time()
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
    print(f'Time : {timedelta(seconds=end - start)}')
    print(classification_report(target_results, prediction_results, labels=[0, 1]))

    roc_auc_curve_display(probability_target_positive, probability_target_negative,
                          target_results, Constants.AMP_NAME)
    matrix_display(confusion_matrix_values, Constants.AMP_NAME)
    feature_importance(model.feature_importances_, all_data_feature.columns, Constants.AMP_NAME)
    return


def roc_auc_curve_display(probability_target_positive, probability_target_negative, target_results, name):
    fpr_poz, tpr_poz, thresholds_poz = roc_curve(target_results, probability_target_positive)
    auc_rez_poz = roc_auc_score(target_results, probability_target_positive)

    fpr_neg, tpr_neg, thresholds_neg = roc_curve(target_results, probability_target_negative)
    auc_rez_neg = roc_auc_score(target_results, probability_target_negative)

    plt.plot(fpr_poz, tpr_poz, color='orange', label='Class 1 (ROC-AUC = %f' % auc_rez_poz)
    plt.plot(fpr_neg, tpr_neg, color='red', label='Class 0 (ROC-AUC = %f' % auc_rez_neg)
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
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.savefig('Confusion_matrix-{}.png'.format(name))
    return


def feature_importance(feature_values_importances, columns_name, name):
    data = {'feature_names': columns_name, 'feature_importance': feature_values_importances}
    data_important_feature = pd.DataFrame(data)

    data_important_feature.sort_values(by=['feature_importance'], ascending=False, inplace=True)
    data_important_feature = data_important_feature.iloc[:40]
    plt.figure(figsize=(10, 8))

    sns.barplot(x=data_important_feature['feature_importance'], y=data_important_feature['feature_names'], color='blue')
    plt.title('Feature importance in Random Forest')
    plt.xlabel('Importance')
    plt.savefig('Feature-importances-{}.png'.format(name))
    return
