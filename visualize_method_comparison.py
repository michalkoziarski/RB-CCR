from algorithm import RBCCR
from imblearn.over_sampling import BorderlineSMOTE, SMOTE
from rbo import RBO
from visualization_tools import *


if __name__ == '__main__':
    dataset_name = 'vehicle3'
    energy = 0.25
    gamma = 0.1

    methods = {
        'none': None,
        'smote': SMOTE(random_state=42),
        'bord': BorderlineSMOTE(random_state=42),
        'rbo': RBO(gamma=gamma, n_steps=100, random_state=42),
        'ccr': RBCCR(energy=energy, gamma=gamma, regions='LEH', random_state=42),
        'rb-ccr': RBCCR(energy=energy, gamma=gamma, regions='E', random_state=42)
    }

    X, y = prepare_data(dataset_name, n_minority_samples=25)

    for method_name, method in methods.items():
        if method is not None:
            X_, y_ = method.fit_sample(X, y)
            appended = X_[X.shape[0]:]
            X_ = X_[:X.shape[0]]
            y_ = y_[:y.shape[0]]
        else:
            X_, y_ = X, y
            appended = None

        visualize(
            X_, y_,
            appended=appended,
            file_name=f'method_comparison_{method_name}.pdf',
            lim=(-0.05, 1.05)
        )
