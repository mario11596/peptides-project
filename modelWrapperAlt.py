import configparser
from statistics import mean
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneOut, StratifiedKFold, cross_val_score
from sklearn.metrics import confusion_matrix, \
    ConfusionMatrixDisplay, classification_report, roc_curve, roc_auc_score, precision_score, recall_score, f1_score
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from numpy import sqrt
from constants import Constants
from time import process_time
from datetime import timedelta
import seaborn as sns

config = configparser.ConfigParser()
config.read('config.ini')
config = config['default']
filter_file = config['output_location-filter']


def catalytic_function():
    filter_data_file = pd.read_csv(filepath_or_buffer=filter_file, delimiter=',')

    all_data_feature = filter_data_file.drop(labels=['FASTA form', 'SMILE form', 'result'], axis=1)
    target = filter_data_file["result"]

    forward_selection(all_data_feature, target, Constants.CATALYTIC_FORWARD_NAME)
    backward_selection(all_data_feature, target, Constants.CATALYTIC_BACKWARD_NAME)
    return


def amp_function():
    filter_data_file = pd.read_csv(filepath_or_buffer=filter_file, delimiter=',')

    all_data_feature = filter_data_file.drop(labels=['FASTA form', 'SMILE form', 'result'], axis=1)
    target = filter_data_file["result"]

    forward_selection(all_data_feature, target, Constants.AMP_FORWARD_NAME)
    backward_selection(all_data_feature, target, Constants.AMP_BACKWARD_NAME)
    return


def forward_selection(all_data_feature, target, name):
    feature_subset = []
    train_feature_subset = []
    train_feature_metrics = []
    feature_score = []

    for i in range(5):
        model_feature_selection = DecisionTreeClassifier(max_depth=9, random_state=20)

        train_feature_metrics.clear()
        train_feature_subset.clear()

        for each_feature in all_data_feature:
            if each_feature not in feature_subset:

                train_feature_subset = feature_subset.copy()
                train_feature_subset.append(each_feature)
                #provjera koliko traje
                print(len(feature_subset))

                score = mean(cross_val_score(model_feature_selection, all_data_feature.loc[:, train_feature_subset],
                                             target, cv=4, scoring='accuracy', n_jobs=4, verbose=3))
                train_feature_metrics.append((score, each_feature))

        train_feature_metrics.sort(key=lambda x: x[0], reverse=True)
        feature_subset.append(train_feature_metrics[0][1])
        feature_score.append(train_feature_metrics[0][0])

    final_subset_feature = feature_subset[0:250]
    print(final_subset_feature)
    new_data_feature = all_data_feature.loc[:, final_subset_feature]

    plot_feature_score(feature_score, name)
    train_model_catalytic(new_data_feature, target, name)
    return


def backward_selection(all_data_feature, target, name):
    feature_subset = all_data_feature
    train_feature_subset = []
    train_feature_metrics = []
    feature_drop = []
    feature_score = []

    count = 5
    while count > 0:
        model_feature_selection = DecisionTreeClassifier(max_depth=7, max_features="sqrt", random_state=20)
        #model_feature_selection = GaussianNB()

        train_feature_metrics.clear()

        for each_feature in all_data_feature:
            if each_feature in feature_subset:

                train_feature_subset = feature_subset.copy()
                train_feature_subset = train_feature_subset.drop(labels=each_feature, axis=1)
                #ovo je za kontrolu
                print(train_feature_subset.shape)

                score = mean(cross_val_score(model_feature_selection, train_feature_subset,
                                             target, cv=3, scoring='accuracy', n_jobs=4, verbose=3))
                train_feature_metrics.append((score, each_feature))

        count -= 1
        train_feature_metrics.sort(key=lambda x: x[0], reverse=False)
        feature_subset = feature_subset.drop(train_feature_metrics[0][1], axis=1)
        feature_drop.append(train_feature_metrics[0][1])
        feature_score.append(train_feature_metrics[-1][0])


    #feature_drop.sort(key=lambda x: x[0], reverse=True)
    final_subset_feature = all_data_feature.loc[:, ~all_data_feature.columns.isin(feature_drop)]
    new_data_feature = final_subset_feature

    plot_feature_score(feature_score, name)

    train_model_amp(new_data_feature, target, name)
    return


def train_model_catalytic(new_data_feature, target, name):
    all_data_feature = new_data_feature
    target = target

    loo_data = LeaveOneOut()
    prediction_results = []
    target_results = []
    probability_target_positive = []

    model = RandomForestClassifier(n_estimators=Constants.N_ESTIMATORS, max_features=Constants.MAX_FEATURES,
                                   min_samples_leaf=Constants.MIN_SAMPLES_LEAF, random_state=50)

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
    g_mean_result = sqrt(precision_result*recall_result)

    print('Accuracy: %f' % accuracy_result)
    print('Geometric mean score: %f' % g_mean_result)
    print('Precision score: %f' % precision_result)
    print('Recall score: %f' % recall_result)
    print('F1 score: %f' % f1_result)

    print(classification_report(target_results, prediction_results, labels=[0, 1]))

    roc_auc_curve_display(probability_target_positive, target_results, name)
    matrix_display(confusion_matrix_values, name)
    feature_importance(model.feature_importances_, all_data_feature.columns, name)
    accuracy_score_display(target_results, probability_target_positive, name)
    return


def train_model_amp(new_data_feature, target, name):
    all_data_feature = new_data_feature
    target = target

    ten_fold_cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=10)
    prediction_results = []
    target_results = []
    probability_target_positive = []

    model = RandomForestClassifier(n_estimators=Constants.N_ESTIMATORS, max_features=Constants.MAX_FEATURES,
                                   min_samples_leaf=Constants.MIN_SAMPLES_LEAF, random_state=50)

    start = process_time()
    for train_index, test_index in ten_fold_cv.split(all_data_feature, target):
        X_train, X_test = all_data_feature.iloc[train_index, :], all_data_feature.iloc[test_index, :]
        y_train, y_test = target.iloc[train_index], target.iloc[test_index]

        model.fit(X_train, y_train)

        all_prediction = model.predict(X_test)
        prediction_results.extend(all_prediction)
        target_results.extend(y_test)

        probability_target_positive.extend(model.predict_proba(X_test)[:, 1])

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

    roc_auc_curve_display(probability_target_positive, target_results, name)
    matrix_display(confusion_matrix_values, name)
    feature_importance(model.feature_importances_, all_data_feature.columns, name)
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


def accuracy_score_display(target_results, probability_target_positive, name):
    plt.figure()
    fop, mpv = calibration_curve(target_results, probability_target_positive, n_bins=30, normalize=True)
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.plot(mpv, fop, marker='.')
    plt.title('Probability calibration Random Forest')
    plt.savefig('Calibration-{}.png'.format(name))
    plt.close()
    return


def plot_feature_score(feature_score, name):
    plt.figure()
    plt.ylabel('Mean accuracy score')
    plt.xlabel('Number of features')
    plt.title('Feature importance')

    x_range = np.arange(0, len(feature_score), 1)
    plt.xticks(np.arange(1, len(feature_score), 10))
    plt.plot(x_range, feature_score, color='r', linewidth=1,  marker='.', label='Cross validation score')
    plt.legend()
    plt.savefig('Mean accuracy score-{}.png'.format(name))
    plt.close()
    return
