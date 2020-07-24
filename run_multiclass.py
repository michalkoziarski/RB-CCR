import argparse
import datasets
import logging
import metrics
import numpy as np
import pandas as pd

from algorithm import MultiClassRBCCR
from collections import Counter
from cv import ResamplingCV
from imblearn.over_sampling import SMOTE
from pathlib import Path
from reference_methods import GlobalCS, MDO, SOUP, StaticSMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.tree import DecisionTreeClassifier


def evaluate_trial(classifier_name, fold):
    for dataset_name in datasets.names(multiclass=True):
        for resampler_name in ['None', 'SMOTE-OVA', 'S-SMOTE', 'Global-CS', 'MDO', 'SOUP',
                               'CCR', 'RB-CCR-H', 'RB-CCR-E', 'RB-CCR-L', 'RB-CCR-CV']:
            RESULTS_PATH = Path(__file__).parents[0] / 'results_multiclass'
            RANDOM_STATE = 42

            np.random.seed(RANDOM_STATE)

            trial_name = f'{dataset_name}_{fold}_{classifier_name}_{resampler_name}'
            trial_path = RESULTS_PATH / f'{trial_name}.csv'

            if trial_path.exists():
                continue

            logging.info(f'Evaluating {trial_name}...')

            dataset = datasets.load(dataset_name)

            (X_train, y_train), (X_test, y_test) = dataset[fold][0], dataset[fold][1]

            class_counts = Counter(y_train)

            for k, v in class_counts.items():
                if v == 1:
                    X_train = np.concatenate([X_train, X_train[y_train == k]])
                    y_train = np.concatenate([y_train, y_train[y_train == k]])

            energies = [0.5, 1.0, 2.5, 5.0, 10.0, 25.0, 50.0, 100.0]
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

            resamplers = {
                'None': None,
                'SMOTE-OVA': ResamplingCV(
                    SMOTE, classifier,
                    k_neighbors=[1, 3, 5, 7, 9],
                    random_state=[RANDOM_STATE], seed=RANDOM_STATE,
                    metrics=(metrics.geometric_average_of_recall,)
                ),
                'S-SMOTE': ResamplingCV(
                    StaticSMOTE, classifier,
                    k=[1, 3, 5, 7, 9], seed=RANDOM_STATE,
                    metrics=(metrics.geometric_average_of_recall,)
                ),
                'Global-CS': GlobalCS(),
                'MDO': ResamplingCV(
                    MDO, classifier,
                    k=[2, 4, 6, 8, 10], seed=RANDOM_STATE,
                    metrics=(metrics.geometric_average_of_recall,)
                ),
                'SOUP': SOUP(),
                'CCR': ResamplingCV(
                    MultiClassRBCCR, classifier, seed=RANDOM_STATE, energy=energies,
                    random_state=[RANDOM_STATE], gamma=[None],
                    metrics=(metrics.geometric_average_of_recall,)
                ),
                'RB-CCR-H': ResamplingCV(
                    MultiClassRBCCR, classifier, seed=RANDOM_STATE, energy=energies,
                    random_state=[RANDOM_STATE], gamma=gammas, regions=['H'],
                    metrics=(metrics.geometric_average_of_recall,)
                ),
                'RB-CCR-E': ResamplingCV(
                    MultiClassRBCCR, classifier, seed=RANDOM_STATE, energy=energies,
                    random_state=[RANDOM_STATE], gamma=gammas, regions=['E'],
                    metrics=(metrics.geometric_average_of_recall,)
                ),
                'RB-CCR-L': ResamplingCV(
                    MultiClassRBCCR, classifier, seed=RANDOM_STATE, energy=energies,
                    random_state=[RANDOM_STATE], gamma=gammas, regions=['L'],
                    metrics=(metrics.geometric_average_of_recall,)
                ),
                'RB-CCR-CV': ResamplingCV(
                    MultiClassRBCCR, classifier, seed=RANDOM_STATE, energy=energies,
                    random_state=[RANDOM_STATE], gamma=gammas, regions=['L', 'E', 'H'],
                    metrics=(metrics.geometric_average_of_recall,)
                )
            }

            resampler = resamplers[resampler_name]

            if resampler is not None:
                X_train, y_train = resampler.fit_sample(X_train, y_train)

            clf = classifier.fit(X_train, y_train)
            predictions = clf.predict(X_test)

            scoring_functions = {
                'AvAcc': metrics.average_accuracy,
                'CBA': metrics.class_balance_accuracy,
                'mGM': metrics.geometric_average_of_recall
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

    args = parser.parse_args()

    evaluate_trial(args.classifier_name, args.fold)
