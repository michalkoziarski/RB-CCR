import numpy as np

from itertools import product
from metrics import auc
from sklearn.model_selection import StratifiedKFold


class ResamplingCV:
    def __init__(self, algorithm, classifier, metrics=(auc,), n=3, seed=None, **kwargs):
        self.algorithm = algorithm
        self.classifier = classifier
        self.metrics = metrics
        self.n = n
        self.seed = seed
        self.kwargs = kwargs

    def fit_sample(self, X, y):
        best_score = -np.inf
        best_parameters = None

        parameter_combinations = list((dict(zip(self.kwargs, x)) for x in product(*self.kwargs.values())))

        if len(parameter_combinations) == 1:
            return self.algorithm(**parameter_combinations[0]).fit_sample(X, y)

        for parameters in parameter_combinations:
            scores = []

            for i in range(self.n):
                skf = StratifiedKFold(n_splits=2, shuffle=True, random_state=self.seed + i)

                for train_idx, test_idx in skf.split(X, y):
                    try:
                        X_train, y_train = self.algorithm(**parameters).fit_sample(X[train_idx], y[train_idx])
                    except (ValueError, RuntimeError) as e:
                        scores.append(-np.inf)

                        break
                    else:
                        if len(np.unique(y_train)) < 2:
                            scores.append(-np.inf)

                            break

                        classifier = self.classifier.fit(X_train, y_train)
                        predictions = classifier.predict(X[test_idx])

                        scores.append(np.mean([metric(y[test_idx], predictions) for metric in self.metrics]))

            score = np.mean(scores)

            if score > best_score:
                best_score = score
                best_parameters = parameters

        if best_parameters is None:
            best_parameters = parameter_combinations[0]

        return self.algorithm(**best_parameters).fit_sample(X, y)
