import argparse
import datasets
import logging
import metrics
import numpy as np
import pandas as pd

from algorithm import RBCCR
from cv import ResamplingCV
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.tree import DecisionTreeClassifier


def evaluate_trial(classifier_name, fold, energy):
    for dataset_name in datasets.names():
        RESULTS_PATH = Path(__file__).parents[0] / 'results_preliminary_energy'
        RANDOM_STATE = 42

        trial_name = f'{dataset_name}_{fold}_{classifier_name}_{energy}'
        trial_path = RESULTS_PATH / f'{trial_name}.csv'

        if trial_path.exists():
            continue

        logging.info(f'Evaluating {trial_name}...')

        dataset = datasets.load(dataset_name)

        (X_train, y_train), (X_test, y_test) = dataset[fold][0], dataset[fold][1]

        gammas = [0.5, 1.0, 2.5, 5.0, 10.0]

        classifiers = {
            'CART': DecisionTreeClassifier(random_state=RANDOM_STATE),
            'KNN': KNeighborsClassifier(),
            'L-SVM': LinearSVC(random_state=RANDOM_STATE),
            'R-SVM': SVC(random_state=RANDOM_STATE, kernel='rbf'),
            'P-SVM': SVC(random_state=RANDOM_STATE, kernel='poly'),
            'LR': LogisticRegression(random_state=RANDOM_STATE),
            'NB': GaussianNB(),
            'R-MLP': MLPClassifier(random_state=RANDOM_STATE),
            'L-MLP': MLPClassifier(random_state=RANDOM_STATE, activation='identity')
        }

        classifier = classifiers[classifier_name]

        resampler = ResamplingCV(
            RBCCR, classifier, seed=RANDOM_STATE, energy=[energy],
            random_state=[RANDOM_STATE], gamma=gammas, regions=['L', 'E', 'H', 'LEH']
        )

        assert len(np.unique(y_train)) == len(np.unique(y_test)) == 2

        if resampler is not None:
            X_train, y_train = resampler.fit_sample(X_train, y_train)

        clf = classifier.fit(X_train, y_train)
        predictions = clf.predict(X_test)

        scoring_functions = {
            'Precision': metrics.precision,
            'Recall': metrics.recall,
            'Specificity': metrics.specificity,
            'AUC': metrics.auc,
            'G-mean': metrics.g_mean,
            'F-measure': metrics.f_measure
        }

        rows = []

        for scoring_function_name in scoring_functions.keys():
            score = scoring_functions[scoring_function_name](y_test, predictions)
            row = [dataset_name, fold, classifier_name, energy, scoring_function_name, score]
            rows.append(row)

        columns = ['Dataset', 'Fold', 'Classifier', 'Energy', 'Metric', 'Score']

        RESULTS_PATH.mkdir(exist_ok=True, parents=True)

        pd.DataFrame(rows, columns=columns).to_csv(trial_path, index=False)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()

    parser.add_argument('-classifier_name', type=str)
    parser.add_argument('-fold', type=int)
    parser.add_argument('-energy', type=float)

    args = parser.parse_args()

    evaluate_trial(args.classifier_name, args.fold, args.energy)
