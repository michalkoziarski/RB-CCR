import datasets
import metrics
import multiprocessing as mp
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
from tqdm import tqdm


RANDOM_STATE = 42
N_PROCESSES = 24
RESULTS_PATH = Path(__file__).parents[0] / 'results_preliminary'


def evaluate_trial(trial):
    dataset_name, fold, classifier_name, energy, gamma = trial

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

    row_block = []

    for scoring_function_name in scoring_functions.keys():
        score = scoring_functions[scoring_function_name](y_test, predictions)
        row = [dataset_name, fold, classifier_name, energy, gamma, scoring_function_name, score]
        row_block.append(row)

    return row_block


if __name__ == '__main__':
    trials = []

    for dataset_name in datasets.names('final'):
        for fold in range(10):
            for classifier_name in ['cart', 'knn', 'svm', 'lr', 'nb', 'mlp']:
                for energy in [0.5, 1.0, 2.5, 5.0, 10.0, 25.0, 50.0, 100.0]:
                    for gamma in [0.5, 1.0, 2.5, 5.0, 10.0]:
                        trials.append((dataset_name, fold, classifier_name, energy, gamma))

    with mp.Pool(N_PROCESSES) as pool:
        row_blocks = list(tqdm(pool.imap(evaluate_trial, trials), total=len(trials)))

    rows = []

    for row_block in row_blocks:
        for row in row_block:
            rows.append(row)

    columns = ['Dataset', 'Fold', 'Classifier', 'Energy', 'Gamma', 'Metric', 'Score']

    RESULTS_PATH.mkdir(exist_ok=True, parents=True)

    pd.DataFrame(rows, columns=columns).to_csv(RESULTS_PATH / 'preliminary_parameters.csv', index=False)
