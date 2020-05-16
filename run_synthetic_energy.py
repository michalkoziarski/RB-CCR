import argparse
import logging
import metrics
import numpy as np
import pandas as pd

from algorithm import CCR
from pathlib import Path
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier


def evaluate_trial(trial):
    for n_samples in range(200, 2100, 100):
        for n_features in range(5, 21):
            RANDOM_STATE = 42
            RESULTS_PATH = Path(__file__).parents[0] / 'results_synthetic_energy'

            fold, imbalance_ratio, energy = trial

            trial_name = f'{n_samples}_{n_features}_{imbalance_ratio}_{energy}_{fold}'

            logging.info(f'Evaluating {trial_name}...')

            n_informative = int(np.ceil(n_features * 0.25))
            n_redundant = int(np.ceil(n_features * 0.25))

            weights = np.array([1.0, 1.0 / imbalance_ratio])
            weights /= np.sum(weights)

            X, y = make_classification(
                n_samples, n_features, n_informative, n_redundant,
                class_sep=0.3, flip_y=0., weights=weights,
                random_state=RANDOM_STATE + fold
            )

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.5, stratify=y,
                random_state=RANDOM_STATE + fold
            )

            scaler = StandardScaler().fit(X_train)

            X_train = scaler.transform(X_train)
            X_test = scaler.transform(X_test)

            resampler = CCR(energy=energy, random_state=RANDOM_STATE)

            assert len(np.unique(y_train)) == len(np.unique(y_test)) == 2

            if resampler is not None:
                X_train, y_train = resampler.fit_sample(X_train, y_train)

            rows = []

            classifiers = {
                'cart': DecisionTreeClassifier(random_state=RANDOM_STATE),
                'knn': KNeighborsClassifier(),
                'svm': LinearSVC(random_state=RANDOM_STATE),
                'lr': LogisticRegression(random_state=RANDOM_STATE),
                'nb': GaussianNB(),
                'mlp': MLPClassifier(random_state=RANDOM_STATE)
            }

            for classifier_name in classifiers.keys():
                classifier = classifiers[classifier_name]

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

                for scoring_function_name in scoring_functions.keys():
                    score = scoring_functions[scoring_function_name](y_test, predictions)
                    row = [n_samples, n_features, imbalance_ratio, fold, classifier_name,
                           energy, scoring_function_name, score]
                    rows.append(row)

            columns = ['Samples', 'Features', 'IR', 'Fold', 'Classifier', 'Energy', 'Metric', 'Score']

            RESULTS_PATH.mkdir(exist_ok=True, parents=True)

            pd.DataFrame(rows, columns=columns).to_csv(RESULTS_PATH / f'{trial_name}.csv', index=False)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()

    parser.add_argument('-fold', type=int)
    parser.add_argument('-imbalance_ratio', type=float)
    parser.add_argument('-energy', type=float)

    args = parser.parse_args()

    evaluate_trial((args.fold, args.imbalance_ratio, args.energy))
