import argparse
import datasets
import logging
import metrics
import numpy as np
import pandas as pd

from algorithm import CCR
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier


def evaluate_trial(trial):
    RANDOM_STATE = 42
    RESULTS_PATH = Path(__file__).parents[0] / 'results_preliminary_parameters'

    dataset_name, fold, classifier_name, energy, gamma = trial

    trial_name = f'{dataset_name}_{fold}_{classifier_name}_{energy}_{gamma}'

    logging.info(f'Evaluating {trial_name}...')

    dataset = datasets.load(dataset_name)

    (X_train, y_train), (X_test, y_test) = dataset[fold][0], dataset[fold][1]

    classifiers = {
        'cart': DecisionTreeClassifier(random_state=RANDOM_STATE),
        'knn': KNeighborsClassifier(),
        'svm': LinearSVC(random_state=RANDOM_STATE),
        'lr': LogisticRegression(random_state=RANDOM_STATE),
        'nb': GaussianNB(),
        'mlp': MLPClassifier(random_state=RANDOM_STATE)
    }

    classifier = classifiers[classifier_name]

    resampler = CCR(energy=energy, gamma=gamma, random_state=RANDOM_STATE)

    assert len(np.unique(y_train)) == len(np.unique(y_test)) == 2

    if resampler is not None:
        X_train, y_train = resampler.fit_sample(X_train, y_train)

    clf = classifier.fit(X_train, y_train)
    predictions = clf.predict(X_test)

    scoring_functions = {
        'precision': metrics.precision,
        'recall': metrics.recall,
        'specificity': metrics.specificity,
        'auc': metrics.auc,
        'g-mean': metrics.g_mean,
        'f-measure': metrics.f_measure
    }

    rows = []

    for scoring_function_name in scoring_functions.keys():
        score = scoring_functions[scoring_function_name](y_test, predictions)
        row = [dataset_name, fold, classifier_name, energy, gamma, scoring_function_name, score]
        rows.append(row)

    columns = ['Dataset', 'Fold', 'Classifier', 'Energy', 'Gamma', 'Metric', 'Score']

    RESULTS_PATH.mkdir(exist_ok=True, parents=True)

    pd.DataFrame(rows, columns=columns).to_csv(RESULTS_PATH / f'{trial_name}.csv', index=False)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()

    parser.add_argument('-classifier_name', type=str)
    parser.add_argument('-dataset_name', type=str)
    parser.add_argument('-fold', type=int)
    parser.add_argument('-energy', type=float)
    parser.add_argument('-gamma', type=float)

    args = parser.parse_args()

    evaluate_trial((args.dataset_name, args.fold, args.classifier_name, args.energy, args.gamma))
