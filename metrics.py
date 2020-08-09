import imblearn.metrics
import numpy as np
import sklearn.metrics

from collections import Counter


def metric_decorator(metric_function):
    def metric_wrapper(ground_truth, predictions, minority_class=None):
        if minority_class is None:
            minority_class = Counter(ground_truth).most_common()[-1][0]

        return metric_function(ground_truth, predictions, minority_class)

    return metric_wrapper


@metric_decorator
def precision(ground_truth, predictions, minority_class=None):
    return sklearn.metrics.precision_score(ground_truth, predictions, pos_label=minority_class, zero_division=0)


@metric_decorator
def recall(ground_truth, predictions, minority_class=None):
    return sklearn.metrics.recall_score(ground_truth, predictions, pos_label=minority_class, zero_division=0)


def specificity(ground_truth, predictions, majority_class=None):
    if majority_class is None:
        majority_class = Counter(ground_truth).most_common()[0][0]

    return sklearn.metrics.recall_score(ground_truth, predictions, pos_label=majority_class, zero_division=0)


@metric_decorator
def f_measure(ground_truth, predictions, minority_class=None):
    return sklearn.metrics.f1_score(ground_truth, predictions, pos_label=minority_class, zero_division=0)


def g_mean(ground_truth, predictions):
    return imblearn.metrics.geometric_mean_score(ground_truth, predictions)


def auc(ground_truth, predictions):
    return sklearn.metrics.roc_auc_score(ground_truth, predictions)
