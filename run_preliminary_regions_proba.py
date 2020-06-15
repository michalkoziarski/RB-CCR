import argparse
import datasets
import logging
import metrics
import numpy as np
import pandas as pd

from algorithm import CCRv6
from cv import ResamplingCV
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC


def evaluate_trial(trial):
    for minority_training_size in [-1, 10]:
        for dataset_name in datasets.names(small=True):
            RANDOM_STATE = 42
            RESULTS_PATH = Path(__file__).parents[0] / f'results_preliminary_regions_proba_{minority_training_size}'

            fold, classifier_name, regions = trial

            trial_name = f'{dataset_name}_{fold}_{classifier_name}_{regions}'
            trial_path = RESULTS_PATH / f'{trial_name}.csv'

            if trial_path.exists():
                continue

            logging.info(f'Evaluating {trial_name}...')

            dataset = datasets.load(dataset_name, minority_training_size=minority_training_size)

            (X_train, y_train), (X_test, y_test) = dataset[fold][0], dataset[fold][1]

            energies = [0.5, 1.0, 2.5, 5.0, 10.0, 25.0, 50.0, 100.0]

            classifiers = {
                'svm': SVC(random_state=RANDOM_STATE, probability=True),
                'lr': LogisticRegression(random_state=RANDOM_STATE),
                'mlp': MLPClassifier(random_state=RANDOM_STATE)
            }

            classifier = classifiers[classifier_name]

            resampler = ResamplingCV(
                CCRv6, classifier, seed=RANDOM_STATE, energy=energies,
                random_state=[RANDOM_STATE], clf=[classifier],
                regions=[regions], metrics=(metrics.auc,)
            )

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
                row = [dataset_name, fold, classifier_name, regions, scoring_function_name, score]
                rows.append(row)

            columns = ['Dataset', 'Fold', 'Classifier', 'Regions', 'Metric', 'Score']

            RESULTS_PATH.mkdir(exist_ok=True, parents=True)

            pd.DataFrame(rows, columns=columns).to_csv(trial_path, index=False)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()

    parser.add_argument('-classifier_name', type=str)
    parser.add_argument('-fold', type=int)
    parser.add_argument('-regions', type=str)

    args = parser.parse_args()

    evaluate_trial((args.fold, args.classifier_name, args.regions))
