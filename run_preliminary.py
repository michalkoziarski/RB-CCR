import os
import pandas as pd
import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt
import seaborn as sns
import metrics

from utils import evaluate, compare
from sklearn.tree import DecisionTreeClassifier
from algorithm import CCR
from cv import ResamplingCV


if __name__ == '__main__':
    results_path = os.path.join(os.path.dirname(__file__), 'results')

    if not os.path.exists(results_path):
        os.mkdir(results_path)

    comparable = []
    energies = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0]
    gammas = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0]

    for gamma in gammas:
        file_name = 'preliminary_cart_gamma_%s.csv' % gamma
        classifier = DecisionTreeClassifier()
        evaluate(ResamplingCV(CCR, classifier, energy=energies, gamma=[gamma], metrics=(metrics.auc,)),
                 classifier, file_name, eval_type='preliminary')
        comparable.append(file_name)

    summary, tables = compare(comparable)

    for measure in ['auc', 'g-mean', 'f-measure']:
        table = tables[measure]
        data = []

        for dataset in table['dataset'].unique():
            for gamma in gammas:
                value = float(table[table['dataset'] == dataset]['preliminary_cart_gamma_%s.csv' % gamma])
                data.append([dataset.replace('-', '').replace('_', ''), gamma, value])

        df = pd.DataFrame(data, columns=['dataset', 'gamma', 'value'])

        grid = sns.FacetGrid(df, col='dataset', col_wrap=5)
        grid.set(ylim=(0.0, 1.0), xticks=range(len(gammas)))
        grid.set_xticklabels(gammas, rotation=90)
        grid.map(plt.plot, 'value')
        grid.savefig(os.path.join(results_path, 'preliminary_%s.pdf' % measure))
