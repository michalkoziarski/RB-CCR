import numpy as np
import pandas as pd

from collections import Counter
from datasets import load_all


if __name__ == '__main__':
    rows = []
    columns = ['Name', 'IR', 'Samples', 'Features']

    for name, dataset in load_all().items():
        (X_train, y_train), (X_test, y_test) = dataset[0][0], dataset[0][1]

        X = np.concatenate([X_train, X_test])
        y = np.concatenate([y_train, y_test])

        n_samples = X.shape[0]
        n_features = X.shape[1]

        majority_class = Counter(y).most_common()[0][0]

        n_majority_samples = Counter(y).most_common()[0][1]
        n_minority_samples = Counter(y).most_common()[1][1]

        imbalance_ratio = np.round(n_majority_samples / n_minority_samples, 2)

        rows.append([name.replace('_', '').replace('-', ''), imbalance_ratio, n_samples, n_features])

    df = pd.DataFrame(rows, columns=columns).sort_values('IR')
    df['IR'] = df['IR'].map(lambda x: f'{x:.2f}')

    for i in range(30):
        row = [str(df.iloc[i][c]) for c in columns]

        if i + 30 < len(df):
            row += [str(df.iloc[i + 30][c]) for c in columns]
        else:
            row += ['' for _ in columns]

        print(' & '.join(row) + ' \\\\')
