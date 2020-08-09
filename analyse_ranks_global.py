import datasets
import numpy as np
import pandas as pd

from merge import RESULTS_PATH


def get_data(metric):
    df = pd.read_csv(RESULTS_PATH / 'results_final.csv')
    df = df[~df['Metric'].isin(['Precision', 'Recall', 'Specificity'])]
    df = df[~df['Resampler'].isin(['CCR', 'RB-CCR-L', 'RB-CCR-E', 'RB-CCR-H'])]
    df['Resampler'] = df['Resampler'].replace({'RB-CCR-CV': 'RB-CCR'})
    df['Method'] = df.apply(lambda x: f'({x["Resampler"]}, {x["Classifier"]})', axis=1)
    df = df.drop(['Classifier', 'Resampler'], axis=1)

    methods = list(df['Method'].unique())
    columns = ['Dataset'] + methods
    rows = []

    for dataset in datasets.names():
        row = [dataset]

        for method in methods:
            ds = df[(df['Method'] == method) & (df['Dataset'] == dataset) & (df['Metric'] == metric)]['Score']

            assert len(ds) == 10

            row.append(np.round(list(ds)[0], 4))

        rows.append(row)

    return pd.DataFrame(rows, columns=columns)


if __name__ == '__main__':
    for metric in ['AUC', 'F-measure', 'G-mean']:
        print(metric)

        ranks = get_data(metric).rank(axis=1, ascending=False).mean(axis=0).sort_values()

        l = [(a, np.round(b, 2)) for a, b in ranks.items()]

        for i in range(21):
            line = f'{l[i][0]} & {l[i][1]:.2f} & {l[i + 21][0]} & {l[i + 21][1]:.2f} & {l[i + 42][0]} & {l[i + 42][1]:.2f} \\\\'

            print(line)

        print('---')
