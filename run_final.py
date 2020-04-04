import metrics
import os

from algorithm import CCR
from cv import ResamplingCV
from imblearn.combine import SMOTEENN, SMOTETomek
from imblearn.over_sampling import ADASYN, BorderlineSMOTE, SMOTE
from imblearn.under_sampling import NeighbourhoodCleaningRule
from sklearn.ensemble import BaggingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from utils import evaluate, compare


if __name__ == '__main__':
    classifiers = {
        'cart': DecisionTreeClassifier(),
        'knn': KNeighborsClassifier(),
        'svm': LinearSVC(),
        'lr': LogisticRegression(),
        'nb': GaussianNB(),
        'mlp': MLPClassifier(),
        'cart-bag': BaggingClassifier(DecisionTreeClassifier()),
        'knn-bag': BaggingClassifier(KNeighborsClassifier())
    }

    energies = [0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 25.0, 50.0, 100.0]

    results_path = os.path.join(os.path.dirname(__file__), 'results')

    if not os.path.exists(results_path):
        os.mkdir(results_path)

    for name, classifier in classifiers.items():
        evaluate(None, classifier, '%s_base.csv' % name, eval_type='final')
        evaluate(ADASYN(), classifier, '%s_adasyn.csv' % name, eval_type='final')
        evaluate(SMOTE(), classifier, '%s_smote.csv' % name, eval_type='final')
        evaluate(BorderlineSMOTE(), classifier, '%s_borderline.csv' % name, eval_type='final')
        evaluate(NeighbourhoodCleaningRule(), classifier, '%s_ncr.csv' % name, eval_type='final')
        evaluate(SMOTETomek(), classifier, '%s_t-link.csv' % name, eval_type='final')
        evaluate(SMOTEENN(), classifier, '%s_enn.csv' % name, eval_type='final')
        evaluate(ResamplingCV(CCR, classifier, energy=energies, metrics=(metrics.auc,)),
                 classifier, '%s_ccr.csv' % name, eval_type='final')

        summary, tables = compare(['%s_base.csv' % name, '%s_adasyn.csv' % name, '%s_smote.csv' % name,
                                   '%s_borderline.csv' % name, '%s_ncr.csv' % name, '%s_t-link.csv' % name,
                                   '%s_enn.csv' % name, '%s_ccr.csv' % name])

        print(summary)

        for measure, table in tables.items():
            table.to_csv(os.path.join(results_path, 'table_%s_%s.csv' % (measure, name)), index=False)
