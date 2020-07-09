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


def evaluate_trial(classifier_name, fold, resampler_name):
    for dataset_name in datasets.names():
        RESULTS_PATH = Path(__file__).parents[0] / 'results_preliminary_scoring'
        RANDOM_STATE = 42

        trial_name = f'{dataset_name}_{fold}_{classifier_name}_{resampler_name}'
        trial_path = RESULTS_PATH / f'{trial_name}.csv'

        if trial_path.exists():
            continue

        logging.info(f'Evaluating {trial_name}...')

        dataset = datasets.load(dataset_name)

        (X_train, y_train), (X_test, y_test) = dataset[fold][0], dataset[fold][1]

        energies = [0.5, 1.0, 2.5, 5.0, 10.0, 25.0, 50.0, 100.0]
        gammas = [0.5, 1.0, 2.5, 5.0, 10.0]

        classifiers = {
            'cart': DecisionTreeClassifier(random_state=RANDOM_STATE),
            'knn': KNeighborsClassifier(),
            'svm': LinearSVC(random_state=RANDOM_STATE),
            'rsvm': SVC(random_state=RANDOM_STATE, kernel='rbf'),
            'psvm': SVC(random_state=RANDOM_STATE, kernel='poly'),
            'lr': LogisticRegression(random_state=RANDOM_STATE),
            'nb': GaussianNB(),
            'mlp': MLPClassifier(random_state=RANDOM_STATE),
            'lmlp': MLPClassifier(random_state=RANDOM_STATE, activation='identity')
        }

        classifier = classifiers[classifier_name]

        resamplers = {
            'rb-ccr-minority': ResamplingCV(
                RBCCR, classifier, seed=RANDOM_STATE, energy=energies,
                random_state=[RANDOM_STATE], gamma=gammas, metrics=(metrics.auc,),
                regions=['L', 'E', 'H'], scoring=['minority']
            ),
            'rb-ccr-majority': ResamplingCV(
                RBCCR, classifier, seed=RANDOM_STATE, energy=energies,
                random_state=[RANDOM_STATE], gamma=gammas, metrics=(metrics.auc,),
                regions=['L', 'E', 'H'], scoring=['majority']
            ),
            'rb-ccr-relative': ResamplingCV(
                RBCCR, classifier, seed=RANDOM_STATE, energy=energies,
                random_state=[RANDOM_STATE], gamma=gammas, metrics=(metrics.auc,),
                regions=['L', 'E', 'H'], scoring=['relative']
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
            'specificity': metrics.specificity,
            'auc': metrics.auc,
            'g-mean': metrics.g_mean,
            'f-measure': metrics.f_measure
        }

        rows = []

        for scoring_function_name in scoring_functions.keys():
            score = scoring_functions[scoring_function_name](y_test, predictions)
            row = [dataset_name, fold, classifier_name, resampler_name, scoring_function_name, score]
            rows.append(row)

        columns = ['Dataset', 'Fold', 'Classifier', 'Resampler', 'Metric', 'Score']

        RESULTS_PATH.mkdir(exist_ok=True, parents=True)

        pd.DataFrame(rows, columns=columns).to_csv(trial_path, index=False)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()

    parser.add_argument('-classifier_name', type=str)
    parser.add_argument('-fold', type=int)
    parser.add_argument('-resampler_name', type=str)

    args = parser.parse_args()

    evaluate_trial(args.classifier_name, args.fold, args.resampler_name)
