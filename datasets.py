import os
import zipfile
import numpy as np
import pandas as pd
import pickle

from sklearn import preprocessing
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from urllib.request import urlretrieve


DATA_PATH = os.path.join(os.path.dirname(__file__), 'data')
FOLDS_PATH = os.path.join(os.path.dirname(__file__), 'folds')

RANDOM_SEED = 42


def download(url):
    name = url.split('/')[-1]
    download_path = os.path.join(DATA_PATH, name)

    if not os.path.exists(DATA_PATH):
        os.makedirs(DATA_PATH)

    if not os.path.exists(download_path):
        urlretrieve(url, download_path)

    if not os.path.exists(download_path.replace('.zip', '.dat')):
        if name.endswith('.zip'):
            with zipfile.ZipFile(download_path) as f:
                f.extractall(DATA_PATH)
        else:
            raise Exception('Unrecognized file type.')


def encode(X, y, encode_features=True):
    y = preprocessing.LabelEncoder().fit(y).transform(y)

    if encode_features:
        encoded = []

        for i in range(X.shape[1]):
            try:
                float(X[0, i])
                encoded.append(X[:, i])
            except ValueError:
                encoded.append(preprocessing.LabelEncoder().fit_transform(X[:, i]))

        X = np.transpose(encoded)

    return X.astype(np.float32), y.astype(np.float32)


def partition(X, y):
    partitions = []

    for i in range(5):
        folds = []
        skf = StratifiedKFold(n_splits=2, shuffle=True, random_state=RANDOM_SEED + i)

        for train_idx, test_idx in skf.split(X, y):
            folds.append([train_idx, test_idx])

        partitions.append(folds)

    return partitions


def load(name, url=None, encode_features=True, remove_metadata=True, scale=True):
    file_name = '%s.dat' % name

    if url is not None:
        download(url)

    skiprows = 0

    if remove_metadata:
        with open(os.path.join(DATA_PATH, file_name)) as f:
            for line in f:
                if line.startswith('@'):
                    skiprows += 1
                else:
                    break

    df = pd.read_csv(os.path.join(DATA_PATH, file_name), header=None, skiprows=skiprows, skipinitialspace=True,
                     sep=' *, *', na_values='?', engine='python')

    matrix = df.dropna().values

    X, y = matrix[:, :-1], matrix[:, -1]
    X, y = encode(X, y, encode_features)

    partitions_path = os.path.join(FOLDS_PATH, file_name.replace('.dat', '.folds.pickle'))

    if not os.path.exists(FOLDS_PATH):
        os.mkdir(FOLDS_PATH)

    if os.path.exists(partitions_path):
        partitions = pickle.load(open(partitions_path, 'rb'))
    else:
        partitions = partition(X, y)
        pickle.dump(partitions, open(partitions_path, 'wb'))

    folds = []

    for i in range(5):
        for j in range(2):
            train_idx, test_idx = partitions[i][j]

            train_set = [X[train_idx], y[train_idx]]
            test_set = [X[test_idx], y[test_idx]]

            if scale:
                scaler = StandardScaler().fit(train_set[0])

                train_set[0] = scaler.transform(train_set[0])
                test_set[0] = scaler.transform(test_set[0])

            folds.append([train_set, test_set])

    return folds


def urls():
    result = []

    with open(os.path.join(os.path.dirname(__file__), 'urls.txt')) as file:
        for line in file.readlines():
            result.append(line.rstrip())

    return result


def names():
    return [url.split('/')[-1].replace('.zip', '') for url in urls()]


def load_all():
    datasets = {}

    for url, name in zip(urls(), names()):
        datasets[name] = load(name, url)

    return datasets


if __name__ == '__main__':
    load_all()
