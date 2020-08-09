import argparse
import datasets
import logging
import metrics
import numpy as np
import pandas as pd

from algorithm import RBCCR
from cv import ResamplingCV
from imblearn.combine import SMOTEENN, SMOTETomek
from imblearn.over_sampling import BorderlineSMOTE, SMOTE
from imblearn.under_sampling import NeighbourhoodCleaningRule
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.tree import DecisionTreeClassifier


def evaluate_trial(classifier_name, fold):
    for dataset_name in datasets.names():
        for resampler_name in ['None', 'SMOTE', 'Bord', 'NCL', 'SMOTE+TL', 'SMOTE+EN', 'CCR',
                               'RB-CCR-H', 'RB-CCR-E', 'RB-CCR-L', 'RB-CCR-CV']:
            RESULTS_PATH = Path(__file__).parents[0] / 'results_final'
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
                'SMOTE': ResamplingCV(
                    SMOTE, classifier,
                    k_neighbors=[1, 3, 5, 7, 9],
                    random_state=[RANDOM_STATE], seed=RANDOM_STATE
                ),
                'Bord': ResamplingCV(
                    BorderlineSMOTE, classifier,
                    k_neighbors=[1, 3, 5, 7, 9],
                    m_neighbors=[5, 10, 15],
                    random_state=[RANDOM_STATE], seed=RANDOM_STATE
                ),
                'NCL': ResamplingCV(
                    NeighbourhoodCleaningRule, classifier,
                    n_neighbors=[1, 3, 5, 7],
                    seed=RANDOM_STATE
                ),
                'SMOTE+TL': ResamplingCV(
                    SMOTETomek, classifier,
                    smote=[SMOTE(k_neighbors=k) for k in [1, 3, 5, 7, 9]],
                    random_state=[RANDOM_STATE], seed=RANDOM_STATE
                ),
                'SMOTE+EN': ResamplingCV(
                    SMOTEENN, classifier,
                    smote=[SMOTE(k_neighbors=k) for k in [1, 3, 5, 7, 9]],
                    random_state=[RANDOM_STATE], seed=RANDOM_STATE
                ),
                'CCR': ResamplingCV(
                    RBCCR, classifier, seed=RANDOM_STATE, energy=energies,
                    random_state=[RANDOM_STATE], gamma=[None]
                ),
                'RB-CCR-H': ResamplingCV(
                    RBCCR, classifier, seed=RANDOM_STATE, energy=energies,
                    random_state=[RANDOM_STATE], gamma=gammas, regions=['H']
                ),
                'RB-CCR-E': ResamplingCV(
                    RBCCR, classifier, seed=RANDOM_STATE, energy=energies,
                    random_state=[RANDOM_STATE], gamma=gammas, regions=['E']
                ),
                'RB-CCR-L': ResamplingCV(
                    RBCCR, classifier, seed=RANDOM_STATE, energy=energies,
                    random_state=[RANDOM_STATE], gamma=gammas, regions=['L']
                ),
                'RB-CCR-CV': ResamplingCV(
                    RBCCR, classifier, seed=RANDOM_STATE, energy=energies,
                    random_state=[RANDOM_STATE], gamma=gammas, regions=['L', 'E', 'H', 'LEH']
                )
            }

            resampler = resamplers[resampler_name]

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
