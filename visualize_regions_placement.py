from visualize_regions_sphere import *


if __name__ == '__main__':
    energy = 0.5
    gamma = 0.25
    dataset_name = 'pima'

    dataset = datasets.load(dataset_name)
    (X_train, y_train), (X_test, y_test) = dataset[0][0], dataset[0][1]
    X, y = np.concatenate([X_train, X_test]), np.concatenate([y_train, y_test])

    minority_class = Counter(y).most_common()[1][0]
    majority_class = Counter(y).most_common()[0][0]

    n_minority = Counter(y).most_common()[1][1]
    n_majority = Counter(y).most_common()[0][1]

    X, y = RandomUnderSampler(
        sampling_strategy={
            minority_class: np.min([n_minority, 20]),
            majority_class: n_majority
        },
        random_state=42,
    ).fit_sample(X, y)

    X = TSNE(n_components=2, random_state=42).fit_transform(X)
    X = MinMaxScaler().fit_transform(X)

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
