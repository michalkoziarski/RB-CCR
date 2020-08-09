from visualization_tools import *


if __name__ == '__main__':
    energy = 1.2
    gamma = 0.3
    dataset_name = 'vehicle3'

    X, y = prepare_data(dataset_name)

    rbccr = RBCCR(
        energy=energy, p_norm=2, regions='LEH',
        gamma=gamma, random_state=42,
        keep_appended=True, keep_radii=True
    )
    rbccr.fit_sample(X, y)

    minority_class = Counter(y).most_common()[1][0]
    minority_points = X[y == minority_class]

    visualize(
        X, y, gamma=gamma,
        regions_center=minority_points[1],
        regions_radius=rbccr.radii[1],
        file_name='regions_sphere.pdf'
    )
