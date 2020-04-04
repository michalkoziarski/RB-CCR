import datasets
import metrics
import numpy as np
import os
import pandas as pd

from tqdm import tqdm


def evaluate(method, classifier, output_file, eval_type=None):
    names = []
    partitions = []
    precisions = []
    recalls = []
    f_measures = []
    aucs = []
    g_means = []

    for name, folds in tqdm(datasets.load_all(eval_type).items(), desc=output_file):
        for i in range(len(folds)):
            (X_train, y_train), (X_test, y_test) = folds[i]

            assert len(np.unique(y_train)) == len(np.unique(y_test)) == 2

            if method is not None:
                X_train, y_train = method.fit_sample(X_train, y_train)

            clf = classifier.fit(X_train, y_train)
            predictions = clf.predict(X_test)

            names.append(name)
            partitions.append(i)
            precisions.append(metrics.precision(y_test, predictions))
            recalls.append(metrics.recall(y_test, predictions))
            f_measures.append(metrics.f_measure(y_test, predictions))
            g_means.append(metrics.g_mean(y_test, predictions))
            aucs.append(metrics.auc(y_test, predictions))

    results_path = os.path.join(os.path.dirname(__file__), 'results')

    if not os.path.exists(results_path):
        os.mkdir(results_path)

    output_path = os.path.join(os.path.dirname(__file__), 'results', output_file)
    df = pd.DataFrame({'dataset': names, 'partition': partitions, 'precision': precisions,
                       'recall': recalls, 'f-measure': f_measures, 'g-mean': g_means, 'auc': aucs})
    df = df[['dataset', 'partition', 'precision', 'recall', 'f-measure', 'g-mean', 'auc']]
    df.to_csv(output_path, index=False)


def compare(output_files):
    dfs = {}
    results = {}
    summary = {}
    tables = {}

    for f in output_files:
        path = os.path.join(os.path.dirname(__file__), 'results', f)
        dfs[f] = pd.read_csv(path)

    dataset_names = list(dfs.values())[0]['dataset'].unique()
    measures = ['precision', 'recall', 'f-measure', 'g-mean', 'auc']

    for measure in measures:
        results[measure] = {}
        summary[measure] = {}
        tables[measure] = []

        for dataset in dataset_names:
            results[measure][dataset] = {}
            row = [dataset]

            for method in output_files:
                df = dfs[method]
                result = df[df['dataset'] == dataset][measure].mean()
                results[measure][dataset][method] = result
                row.append(result)

            tables[measure].append(row)

        for method in output_files:
            summary[measure][method] = 0

        tables[measure] = pd.DataFrame(tables[measure], columns=['dataset'] + output_files)

    for measure in measures:
        for dataset in dataset_names:
            best_method = max(results[measure][dataset], key=results[measure][dataset].get)
            summary[measure][best_method] += 1

    return summary, tables
