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


@metric_decorator
def f_measure(ground_truth, predictions, minority_class=None):
    return sklearn.metrics.f1_score(ground_truth, predictions, pos_label=minority_class, zero_division=0)


def g_mean(ground_truth, predictions):
    return imblearn.metrics.geometric_mean_score(ground_truth, predictions)


def auc(ground_truth, predictions):
    return sklearn.metrics.roc_auc_score(ground_truth, predictions)


def specificity(ground_truth, predictions, majority_class=None):
    if majority_class is None:
        majority_class = Counter(ground_truth).most_common()[0][0]

    return sklearn.metrics.recall_score(ground_truth, predictions, pos_label=majority_class, zero_division=0)


def confusion_matrix(ground_truth, predictions):
    assert len(ground_truth) == len(predictions)

    if type(ground_truth) is not np.ndarray:
        ground_truth = np.array(ground_truth)

    if type(predictions) is not np.ndarray:
        predictions = np.array(predictions)

    n_classes = len(np.unique(ground_truth))
    result = np.empty((n_classes,) * 2, np.int64)

    for i, c1 in enumerate(np.unique(ground_truth)):
        for j, c2 in enumerate(np.unique(ground_truth)):
            result[i][j] = sum(((ground_truth == c1) & (predictions == c2)))

    return result


def class_accuracies(ground_truth, predictions):
    assert len(ground_truth) == len(predictions)

    if type(ground_truth) is not np.ndarray:
        ground_truth = np.array(ground_truth)

    if type(predictions) is not np.ndarray:
        predictions = np.array(predictions)

    result = {}

    for label in np.unique(ground_truth):
        indices = (ground_truth == label)
        result[label] = sklearn.metrics.accuracy_score(ground_truth[indices], predictions[indices])

    return result


def class_recalls(ground_truth, predictions):
    assert len(ground_truth) == len(predictions)

    if type(ground_truth) is not np.ndarray:
        ground_truth = np.array(ground_truth)

    if type(predictions) is not np.ndarray:
        predictions = np.array(predictions)

    ground_truth = ground_truth.astype(np.int64)
    predictions = predictions.astype(np.int64)

    cm = np.array(confusion_matrix(ground_truth, predictions))
    result = {}

    for label, _ in enumerate(np.unique(ground_truth)):
        result[label] = cm[label][label] / sum(cm[label])

    return result


def average_accuracy(ground_truth, predictions):
    assert len(ground_truth) == len(predictions)

    return np.mean(list(class_accuracies(ground_truth, predictions).values()))


def class_balance_accuracy(ground_truth, predictions):
    assert len(ground_truth) == len(predictions)

    n_classes = len(np.unique(ground_truth))
    cm = confusion_matrix(ground_truth, predictions)
    result = 0

    for i in range(n_classes):
        sum_cm_i_j = sum([cm[i][j] for j in range(n_classes)])
        sum_cm_j_i = sum([cm[j][i] for j in range(n_classes)])

        result += cm[i][i] / max(sum_cm_i_j, sum_cm_j_i)

    result /= n_classes

    return result


def geometric_average_of_recall(ground_truth, predictions):
    assert len(ground_truth) == len(predictions)

    recalls = class_recalls(ground_truth, predictions).values()
    result = 1.0

    for recall in recalls:
        result *= recall

    result = result ** (1 / len(recalls))

    return result
