from visualization_tools import *


if __name__ == '__main__':
    dataset_name = 'pima'

    X, y = prepare_data(dataset_name, n_minority_samples=15)

    for energy in [0.1, 0.25, 0.5, 1.0]:
        rbccr = RBCCR(
            energy=energy, p_norm=2, regions='LEH',
            gamma=None, random_state=42,
            keep_appended=True, keep_radii=True
        )
        X_, y_ = rbccr.fit_sample(X, y)

        visualize(
            X_[:X.shape[0]], y_[:y.shape[0]],
            appended=rbccr.appended,
            radii=rbccr.radii,
            file_name=f'energy_sphere_radius_{energy}.pdf',
            lim=(-0.05, 1.05)
        )
