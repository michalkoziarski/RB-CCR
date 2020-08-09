from visualization_tools import *


if __name__ == '__main__':
    energy = 0.5
    gamma = 0.25
    dataset_name = 'pima'

    X, y = prepare_data(dataset_name)

    for regions in ['L', 'E', 'H', 'LEH']:
        rbccr = RBCCR(
            energy=energy, p_norm=2, regions=regions,
            gamma=gamma, random_state=42,
            keep_appended=True, keep_radii=True
        )
        rbccr.fit_sample(X, y)

        visualize(
            X, y, gamma=gamma,
            appended=rbccr.appended,
            radii=rbccr.radii,
            file_name=f'regions_placement_{regions}.pdf'
        )
