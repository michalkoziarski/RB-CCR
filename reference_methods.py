import numpy as np

from collections import Counter
from imblearn.over_sampling import SMOTE
from multi_imbalance.resampling.global_cs import GlobalCS as _GlobalCS
from multi_imbalance.resampling.soup import SOUP as _SOUP
from smote_variants import MDO as _MDO, MulticlassOversampling


class StaticSMOTE:
    """
    Adapted from https://github.com/damian-horna/multi-imbalance/blob/master/multi_imbalance/resampling/static_smote.py

    Static SMOTE implementation:

    Reference:
    Fernández-Navarro, F., Hervás-Martínez, C., Gutiérrez, P.A.: A dynamic over-sampling
    procedure based on sensitivity for multi-class problems. Pattern Recognit. 44, 1821–1833
    (2011)
    """
    def __init__(self, k):
        self.k = k

    def fit_sample(self, X, y):
        cnt = Counter(y)
        min_class = min(cnt, key=cnt.get)
        X_original, y_original = X.copy(), y.copy()
        X_resampled, y_resampled = X.copy(), y.copy()

        M = len(list(cnt.keys()))

        for _ in range(M):
            sm = SMOTE(k_neighbors=self.k, sampling_strategy={min_class: cnt[min_class] * 2})
            X_smote, y_smote = sm.fit_resample(X_original, y_original)
            X_added_examples = X_smote[y_smote == min_class][cnt[min_class]:, :]
            X_resampled = np.vstack([X_resampled, X_added_examples])
            y_resampled = np.hstack([y_resampled, y_smote[y_smote == min_class][cnt[min_class]:]])
            cnt = Counter(y_resampled)
            min_class = min(cnt, key=cnt.get)

        return X_resampled, y_resampled


class GlobalCS(_GlobalCS):
    def fit_sample(self, X, y):
        return self.fit_transform(X, y)


class MDO:
    def __init__(self, k=5):
        self.mdo = MulticlassOversampling(_MDO(K2=k))

    def fit_sample(self, X, y):
        return self.mdo.sample(X, y)


class SOUP(_SOUP):
    def fit_sample(self, X, y):
        return self.fit_transform(X, y)
