import configparser
from math import sqrt
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, \
    ConfusionMatrixDisplay, classification_report, roc_curve, roc_auc_score, precision_score, recall_score
from sklearn.model_selection import LeaveOneOut, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from constants import Constants
from time import process_time, time
from datetime import timedelta
import seaborn as sns
from sklearn.calibration import calibration_curve


config = configparser.ConfigParser()
config.read('config.ini')
config = config['default']
filter_file = config['output_location-filter']


# machine learning for catalytic peptides
def train_model_catalytic():
    filter_data_file = pd.read_csv(filepath_or_buffer=filter_file, delimiter=',')
    all_data_feature = filter_data_file.drop(labels=['FASTA form', 'SMILE form', 'result'], axis=1)
    target = filter_data_file["result"]

    loo_data = LeaveOneOut()
    prediction_results = []
    target_results = []
    probability_target_positive = []

    model = RandomForestClassifier(n_estimators=Constants.N_ESTIMATORS, max_features=Constants.MAX_FEATURES,
                                   min_samples_leaf=Constants.MIN_SAMPLES_LEAF, random_state=50)
    start = time()
    for train_index, test_index in loo_data.split(all_data_feature):
        X_train, X_test = all_data_feature.iloc[train_index, :], all_data_feature.iloc[test_index, :]
        y_train, y_test = target.iloc[train_index], target.iloc[test_index]

        model.fit(X_train, y_train)
        all_prediction = model.predict(X_test)
        prediction_results.extend(all_prediction)
        target_results.extend(y_test)

        # the predicted probabilities for positive
        probability_target_positive.extend(model.predict_proba(X_test)[:, 1])

    end = time()
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

    roc_auc_curve_display(probability_target_positive,
                          target_results, Constants.CATALYTIC_NAME)
    matrix_display(confusion_matrix_values, Constants.CATALYTIC_NAME)
    feature_importance(model.feature_importances_, all_data_feature.columns, Constants.CATALYTIC_NAME)
    calibration_score_display(target_results, probability_target_positive, Constants.CATALYTIC_NAME)
    return


# machine learning for AMP (DRAMP 2) peptides
def train_model_amp():
    filter_data_file = pd.read_csv(filepath_or_buffer=filter_file, delimiter=',')
    all_data_feature = filter_data_file.drop(columns=['FASTA form', 'SMILE form', 'result'], axis=1)
    target = filter_data_file["result"]

    ten_fold_cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=10)
    prediction_results = []
    target_results = []
    probability_target_positive = []

    model = RandomForestClassifier(n_estimators=Constants.N_ESTIMATORS, max_features=Constants.MAX_FEATURES,
                                   min_samples_leaf=Constants.MIN_SAMPLES_LEAF, random_state=50)

    start = time()
    for train_index, test_index in ten_fold_cv.split(all_data_feature, target):
        X_train, X_test = all_data_feature.iloc[train_index, :], all_data_feature.iloc[test_index, :]
        y_train, y_test = target.iloc[train_index], target.iloc[test_index]

        model.fit(X_train, y_train)

        all_prediction = model.predict(X_test)
        prediction_results.extend(all_prediction)
        target_results.extend(y_test)

        # the predicted probabilities for positive
        probability_target_positive.extend(model.predict_proba(X_test)[:, 1])

    end = time()
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

    roc_auc_curve_display(probability_target_positive,
                          target_results, Constants.AMP_NAME)
    matrix_display(confusion_matrix_values, Constants.AMP_NAME)
    feature_importance(model.feature_importances_, all_data_feature.columns, Constants.AMP_NAME)
    calibration_score_display(target_results, probability_target_positive, Constants.AMP_NAME)
    return


# create ROC-AUC graphs with value
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


# create calibration curve on graph
def calibration_score_display(target_results, probability_target_positive, name):
    plt.figure()
    fop, mpv = calibration_curve(target_results, probability_target_positive, n_bins=20, normalize=True)
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.plot(mpv, fop, marker='.', label='Random Forest')
    plt.title('Probability calibration Random Forest')
    plt.ylabel('Fraction of positives')
    plt.xlabel('Mean predicted value')
    plt.legend()
    plt.savefig('Calibration-{}.png'.format(name))
    plt.close()
    return

