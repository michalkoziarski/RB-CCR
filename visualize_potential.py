from visualization_tools import *


if __name__ == '__main__':
    dataset_name = 'yeast3'

    X, y = prepare_data(dataset_name, scaler='Standard')

    for gamma in [0.5, 0.75, 1.0, 1.25]:
        visualize(
            X, y, gamma=gamma,
            file_name=f'potential_{gamma}.pdf'
        )
