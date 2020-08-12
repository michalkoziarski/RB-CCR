from visualization_tools import *


if __name__ == '__main__':
    dataset_name = 'pima'

    X, y = prepare_data(dataset_name, scaler='Standard', n_minority_samples=1000)

    for gamma in [0.1, 0.25, 0.5, 1.0]:
        visualize(
            X, y, gamma=gamma,
            file_name=f'potential_{gamma}.pdf'
        )
