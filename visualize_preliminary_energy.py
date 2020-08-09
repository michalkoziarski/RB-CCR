import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from merge import RESULTS_PATH
from pathlib import Path


VISUALIZATIONS_PATH = Path(__file__).parent / 'visualizations'


if __name__ == '__main__':
    VISUALIZATIONS_PATH.mkdir(exist_ok=True, parents=True)

    df = pd.read_csv(RESULTS_PATH / 'results_preliminary_energy.csv')

    g = sns.catplot(
        data=df, x='Energy', y='Score', row='Classifier', col='Metric',
        sharey='col', height=1.5, aspect=1.5, kind='point', ci=None,
        row_order=['CART', 'KNN', 'L-SVM', 'R-SVM', 'P-SVM', 'LR', 'NB', 'L-MLP', 'R-MLP'],
        col_order=['Precision', 'Recall', 'Specificity', 'AUC', 'F-measure', 'G-mean'],
        hue='Metric'
    )
    g.set_titles('{row_name}, {col_name}')
    g.set_xticklabels(rotation=60)
    g.fig.tight_layout()

    plt.savefig(VISUALIZATIONS_PATH / 'preliminary_energy.pdf', bbox_inches='tight')
