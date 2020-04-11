import datasets
import metrics
import multiprocessing as mp
import numpy as np
import os
import pandas as pd

from algorithm import CCR
from cv import ResamplingCV
from imblearn.combine import SMOTEENN, SMOTETomek
from imblearn.over_sampling import BorderlineSMOTE, SMOTE
from imblearn.under_sampling import NeighbourhoodCleaningRule
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
RESULTS_PATH = Path(__file__).parents[0] / 'results'


def evaluate_trial(trial):
    dataset_name, fold, minority_training_size, classifier_name, resampler_name = trial

    if minority_training_size == -1:
        minority_training_size = None

    dataset = datasets.load(dataset_name, minority_training_size=minority_training_size)

    (X_train, y_train), (X_test, y_test) = dataset[fold][0], dataset[fold][1]

    energies = [0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 25.0, 50.0, 100.0]
    gammas = [0.01, 0.1, 1.0, 10.0]

    classifiers = {
        'cart': DecisionTreeClassifier(random_state=RANDOM_STATE),
        'knn': KNeighborsClassifier(),
        'svm': LinearSVC(random_state=RANDOM_STATE),
        'lr': LogisticRegression(random_state=RANDOM_STATE),
        'nb': GaussianNB(),
        'mlp': MLPClassifier(random_state=RANDOM_STATE)
    }

    classifier = classifiers[classifier_name]

    resamplers = {
        'none': None,
        'smote':  ResamplingCV(
            SMOTE, classifier,
            k_neighbors=[1, 3, 5, 7, 9],
            random_state=[RANDOM_STATE], seed=RANDOM_STATE
        ),
        'bord': ResamplingCV(
            BorderlineSMOTE, classifier,
            k_neighbors=[1, 3, 5, 7, 9],
            m_neighbors=[5, 10, 15],
            random_state=[RANDOM_STATE], seed=RANDOM_STATE
        ),
        'ncl':  ResamplingCV(
            NeighbourhoodCleaningRule, classifier,
            n_neighbors=[1, 3, 5, 7],
            seed=RANDOM_STATE
        ),
        'smote+tl': ResamplingCV(
            SMOTETomek, classifier,
            smote=[SMOTE(k_neighbors=k) for k in [1, 3, 5, 7, 9]],
            random_state=[RANDOM_STATE], seed=RANDOM_STATE
        ),
        'smote+enn': ResamplingCV(
            SMOTEENN, classifier,
            smote=[SMOTE(k_neighbors=k) for k in [1, 3, 5, 7, 9]],
            random_state=[RANDOM_STATE], seed=RANDOM_STATE
        ),
        'ccr': ResamplingCV(
            CCR, classifier, seed=RANDOM_STATE, energy=energies,
            random_state=[RANDOM_STATE], metrics=(metrics.auc,)
        ),
        'rb-ccr': ResamplingCV(
            CCR, classifier, seed=RANDOM_STATE, energy=energies,
            random_state=[RANDOM_STATE], gamma=gammas, metrics=(metrics.auc,)
        )
    }

    resampler = resamplers[resampler_name]

    assert len(np.unique(y_train)) == len(np.unique(y_test)) == 2

    if resampler is not None:
        X_train, y_train = resampler.fit_sample(X_train, y_train)

    clf = classifier.fit(X_train, y_train)
    predictions = clf.predict(X_test)

    scoring_functions = {
        'precision': metrics.precision,
        'recall': metrics.recall,
        'auc': metrics.auc,
        'g-mean': metrics.g_mean,
        'f-measure': metrics.f_measure
    }

    row_block = []

    for scoring_function_name in scoring_functions.keys():
        score = scoring_functions[scoring_function_name](y_test, predictions)
        row = [dataset_name, fold, minority_training_size, classifier_name, resampler_name, scoring_function_name, score]
        row_block.append(row)

    return row_block


if __name__ == '__main__':
    results_path = os.path.join(os.path.dirname(__file__), 'results')

    trials = []

    for dataset_name in datasets.names('final'):
        for fold in range(10):
            for minority_training_size in [-1, 5, 10, 15, 20, 30]:
                for classifier_name in ['cart', 'knn', 'svm', 'lr', 'nb', 'mlp']:
                    for resampler_name in ['none', 'smote', 'bord', 'ncl', 'smote+tl', 'smote+enn', 'ccr', 'rb-ccr']:
                        trials.append((dataset_name, fold, minority_training_size, classifier_name, resampler_name))

    with mp.Pool(N_PROCESSES) as pool:
        row_blocks = list(tqdm(pool.imap(evaluate_trial, trials), total=len(trials)))

    rows = []

    for row_block in row_blocks:
        for row in row_block:
            rows.append(row)

    columns = ['Dataset', 'Fold', 'Minority Size', 'Classifier', 'Resampler', 'Metric', 'Score']

    RESULTS_PATH.mkdir(exist_ok=True, parents=True)

    pd.DataFrame(rows, columns=columns).to_csv(RESULTS_PATH / 'final.csv', index=False)
