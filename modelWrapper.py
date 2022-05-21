import configparser
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from mlxtend.feature_selection import SequentialFeatureSelector
from sklearn.model_selection import LeaveOneOut, train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import confusion_matrix, plot_confusion_matrix, \
    ConfusionMatrixDisplay, classification_report, roc_curve, roc_auc_score, precision_score, recall_score
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from numpy import sqrt
from constants import Constants

config = configparser.ConfigParser()
config.read('config.ini')
config = config['default']
filter_file = config['output_location-filter']


def train_model_catalytic():
    filter_data_file = pd.read_csv(filepath_or_buffer=filter_file, delimiter=',')

    all_data_feature = filter_data_file.drop(labels=['FASTA form', 'SMILE form', 'result'], axis=1)
    target = filter_data_file["result"]

    forward_selection(all_data_feature, target, Constants.CATALYTIC_FORWARD_NAME)
    backward_selection(all_data_feature, target, Constants.CATALYTIC_FORWARD_NAME)
    return


def train_model_amp():
    filter_data_file = pd.read_csv(filepath_or_buffer=filter_file, delimiter=',')

    all_data_feature = filter_data_file.drop(labels=['FASTA form', 'SMILE form', 'result'], axis=1)
    target = filter_data_file["result"]

    forward_selection(all_data_feature, target, Constants.AMP_FORWARD_NAME)
    backward_selection(all_data_feature, target, Constants.AMP_BACKWARD_NAME)


def forward_selection(all_data_feature, target, name):
    X_train, X_test, y_train, y_test = train_test_split(all_data_feature, target, test_size=0.3, random_state=0)

    model_feature_selection = DecisionTreeClassifier()

    model = RandomForestClassifier(n_estimators=Constants.N_ESTIMATORS, max_features=Constants.MAX_FEATURES,
                                   min_samples_leaf=Constants.MIN_SAMPLES_LEAF)

    sfs = SequentialFeatureSelector(model_feature_selection, k_features=(100, 1000), forward=True, floating=False,
                                    scoring='accuracy', cv=3, verbose=2)
    sfs.fit(X_train, y_train)
    selected_feature_names = list(sfs.k_feature_names_)
    selected_feature_index = list(sfs.k_feature_idx_)
    X_train = X_train.iloc[:, selected_feature_index]
    X_test = X_test.iloc[:, selected_feature_index]
    print(selected_feature_names)
    print(selected_feature_index)
    print("Number of features: " + str(len(selected_feature_names)))

    model.fit(X_train, y_train)
    prediction_results = model.predict(X_test)
    probability_target = model.predict_proba(X_test)[:, 1]

    #target_results = np.ravel(y_test)
    target_results = y_test

    confusion_matrix_values = confusion_matrix(target_results, prediction_results)
    accuracy_result = accuracy_score(target_results, prediction_results)
    precision_result = precision_score(target_results, prediction_results)
    recall_result = recall_score(target_results, prediction_results)
    g_mean_result = sqrt(precision_result * recall_result)

    print('Accuracy: %f' % accuracy_result)
    print('Geometric mean score: %f' % g_mean_result)
    print(classification_report(target_results, prediction_results, labels=[0, 1]))

    roc_auc_curve_display(probability_target, target_results, name)
    matrix_display(confusion_matrix_values, name)
    return


def backward_selection(all_data_feature, target, name):
    X_train, X_test, y_train, y_test = train_test_split(all_data_feature, target, test_size=0.3, random_state=0)
    ten_fold_cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=None)

    model_feature_selection = DecisionTreeClassifier()

    model = RandomForestClassifier(n_estimators=Constants.N_ESTIMATORS, max_features=Constants.MAX_FEATURES,
                                   min_samples_leaf=Constants.MIN_SAMPLES_LEAF)

    sfs = SequentialFeatureSelector(model_feature_selection, k_features=(100, 1000), forward=False, floating=False,
                                    scoring='accuracy', cv=3, verbose=2)
    sfs.fit(X_train, y_train)
    selected_feature_names = list(sfs.k_feature_names_)
    selected_feature_index = list(sfs.k_feature_idx_)
    X_train = X_train.iloc[:, selected_feature_index]
    X_test = X_test.iloc[:, selected_feature_index]
    print(selected_feature_names)
    print(selected_feature_index)
    print("Number of features: " + str(len(selected_feature_names)))

    model.fit(X_train, y_train)
    prediction_results = model.predict(X_test)
    probability_target = model.predict_proba(X_test)[:, 1]

    # target_results = np.ravel(y_test)
    target_results = y_test

    confusion_matrix_values = confusion_matrix(target_results, prediction_results)
    accuracy_result = accuracy_score(target_results, prediction_results)
    precision_result = precision_score(target_results, prediction_results)
    recall_result = recall_score(target_results, prediction_results)
    g_mean_result = sqrt(precision_result * recall_result)

    print('Accuracy: %f' % accuracy_result)
    print('Geometric mean score: %f' % g_mean_result)
    print(classification_report(target_results, prediction_results, labels=[0, 1]))

    roc_auc_curve_display(probability_target, target_results, name)
    matrix_display(confusion_matrix_values, name)
    return


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
    plt.savefig('Histogram-probabilities-forward-{}.png'.format(name))
    return

"""
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