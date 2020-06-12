import numpy as np

from sklearn.neighbors import NearestNeighbors


def distance(x, y, p_norm=2):
    return np.sum(np.abs(x - y) ** p_norm) ** (1 / p_norm)


def sample_inside_sphere(dimensionality, radius, p_norm=2):
    direction_unit_vector = (2 * np.random.rand(dimensionality) - 1)
    direction_unit_vector = direction_unit_vector / distance(direction_unit_vector, np.zeros(dimensionality), p_norm)

    return direction_unit_vector * np.random.rand() * radius


def rbf(d, gamma):
    if gamma == 0.0:
        return 0.0
    else:
        return np.exp(-(d / gamma) ** 2)


def rbf_score(point, majority_points, gamma, p_norm=2):
    result = 0.0

    for majority_point in majority_points:
        result += rbf(distance(point, majority_point, p_norm), gamma)

    return result


class CCR:
    def __init__(self, energy, gamma=None, max_attempts=10, p_norm=2, n=None, random_state=None):
        self.energy = energy
        self.gamma = gamma
        self.max_attempts = max_attempts
        self.p_norm = p_norm
        self.n = n
        self.random_state = random_state

    def fit_sample(self, X, y):
        np.random.seed(self.random_state)

        classes = np.unique(y)
        sizes = [sum(y == c) for c in classes]
        minority_class = classes[np.argmin(sizes)]

        minority_points = X[y == minority_class].copy()
        majority_points = X[y != minority_class].copy()
        minority_labels = y[y == minority_class].copy()
        majority_labels = y[y != minority_class].copy()

        if self.n is None:
            n = len(majority_points) - len(minority_points)
        else:
            n = self.n

        distances = np.zeros((len(minority_points), len(majority_points)))

        for i in range(len(minority_points)):
            for j in range(len(majority_points)):
                distances[i][j] = distance(minority_points[i], majority_points[j], self.p_norm)

        radii = np.zeros(len(minority_points))

        translations = np.zeros(majority_points.shape)

        for i in range(len(minority_points)):
            minority_point = minority_points[i]
            remaining_energy = self.energy
            radius = 0.0
            sorted_distances = np.argsort(distances[i])
            n_majority_points_within_radius = 0

            while True:
                if n_majority_points_within_radius == len(majority_points):
                    if n_majority_points_within_radius == 0:
                        radius_change = remaining_energy / (n_majority_points_within_radius + 1)
                    else:
                        radius_change = remaining_energy / n_majority_points_within_radius

                    radius += radius_change

                    break

                radius_change = remaining_energy / (n_majority_points_within_radius + 1)

                if distances[i, sorted_distances[n_majority_points_within_radius]] >= radius + radius_change:
                    radius += radius_change

                    break
                else:
                    if n_majority_points_within_radius == 0:
                        last_distance = 0.0
                    else:
                        last_distance = distances[i, sorted_distances[n_majority_points_within_radius - 1]]

                    radius_change = distances[i, sorted_distances[n_majority_points_within_radius]] - last_distance
                    radius += radius_change
                    remaining_energy -= radius_change * (n_majority_points_within_radius + 1)
                    n_majority_points_within_radius += 1

            radii[i] = radius

            for j in range(n_majority_points_within_radius):
                majority_point = majority_points[sorted_distances[j]]
                d = distances[i, sorted_distances[j]]

                while d < 1e-20:
                    majority_point += (1e-6 * np.random.rand(len(majority_point)) + 1e-6) * \
                                      np.random.choice([-1.0, 1.0], len(majority_point))
                    d = distance(minority_point, majority_point)

                translation = (radius - d) / d * (majority_point - minority_point)
                translations[sorted_distances[j]] += translation

        appended = []

        for i in range(len(minority_points)):
            minority_point = minority_points[i]
            n_synthetic_samples = int(np.round(1.0 / (radii[i] * np.sum(1.0 / radii)) * n))
            r = radii[i]

            if self.gamma is None:
                for _ in range(n_synthetic_samples):
                    appended.append(minority_point + sample_inside_sphere(len(minority_point), r, self.p_norm))
            else:
                seed_score = rbf_score(minority_point, majority_points, self.gamma)

                for _ in range(n_synthetic_samples):
                    attempt = 0

                    while attempt < self.max_attempts:
                        sample = minority_point + sample_inside_sphere(len(minority_point), r, self.p_norm)
                        sample_score = rbf_score(sample, majority_points, self.gamma)

                        if sample_score <= seed_score:
                            appended.append(sample)

                            break

                        attempt += 1
                    else:
                        appended.append(minority_point)

        majority_points += translations

        if len(appended) > 0:
            points = np.concatenate([majority_points, minority_points, appended])
            labels = np.concatenate([majority_labels, minority_labels, np.tile([minority_class], len(appended))])
        else:
            points = np.concatenate([majority_points, minority_points])
            labels = np.concatenate([majority_labels, minority_labels])

        return points, labels


class CCRv2:
    def __init__(self, energy, gamma=None, n_samples=250, threshold=0.33,
                 regions='LE', p_norm=2, n=None, random_state=None):
        self.energy = energy
        self.gamma = gamma
        self.n_samples = n_samples
        self.threshold = threshold
        self.regions = regions
        self.p_norm = p_norm
        self.n = n
        self.random_state = random_state

    def fit_sample(self, X, y):
        np.random.seed(self.random_state)

        classes = np.unique(y)
        sizes = [sum(y == c) for c in classes]
        minority_class = classes[np.argmin(sizes)]

        minority_points = X[y == minority_class].copy()
        majority_points = X[y != minority_class].copy()
        minority_labels = y[y == minority_class].copy()
        majority_labels = y[y != minority_class].copy()

        if self.n is None:
            n = len(majority_points) - len(minority_points)
        else:
            n = self.n

        distances = np.zeros((len(minority_points), len(majority_points)))

        for i in range(len(minority_points)):
            for j in range(len(majority_points)):
                distances[i][j] = distance(minority_points[i], majority_points[j], self.p_norm)

        radii = np.zeros(len(minority_points))

        translations = np.zeros(majority_points.shape)

        for i in range(len(minority_points)):
            minority_point = minority_points[i]
            remaining_energy = self.energy
            radius = 0.0
            sorted_distances = np.argsort(distances[i])
            n_majority_points_within_radius = 0

            while True:
                if n_majority_points_within_radius == len(majority_points):
                    if n_majority_points_within_radius == 0:
                        radius_change = remaining_energy / (n_majority_points_within_radius + 1)
                    else:
                        radius_change = remaining_energy / n_majority_points_within_radius

                    radius += radius_change

                    break

                radius_change = remaining_energy / (n_majority_points_within_radius + 1)

                if distances[i, sorted_distances[n_majority_points_within_radius]] >= radius + radius_change:
                    radius += radius_change

                    break
                else:
                    if n_majority_points_within_radius == 0:
                        last_distance = 0.0
                    else:
                        last_distance = distances[i, sorted_distances[n_majority_points_within_radius - 1]]

                    radius_change = distances[i, sorted_distances[n_majority_points_within_radius]] - last_distance
                    radius += radius_change
                    remaining_energy -= radius_change * (n_majority_points_within_radius + 1)
                    n_majority_points_within_radius += 1

            radii[i] = radius

            for j in range(n_majority_points_within_radius):
                majority_point = majority_points[sorted_distances[j]]
                d = distances[i, sorted_distances[j]]

                while d < 1e-20:
                    majority_point += (1e-6 * np.random.rand(len(majority_point)) + 1e-6) * \
                                      np.random.choice([-1.0, 1.0], len(majority_point))
                    d = distance(minority_point, majority_point)

                translation = (radius - d) / d * (majority_point - minority_point)
                translations[sorted_distances[j]] += translation

        appended = []

        for i in range(len(minority_points)):
            minority_point = minority_points[i]
            n_synthetic_samples = int(np.round(1.0 / (radii[i] * np.sum(1.0 / radii)) * n))
            r = radii[i]

            samples = []
            scores = []

            for _ in range(self.n_samples):
                sample = minority_point + sample_inside_sphere(len(minority_point), r, self.p_norm)
                score = rbf_score(sample, majority_points, self.gamma)

                samples.append(sample)
                scores.append(score)

            if self.gamma is None or ('L' in self.regions and 'E' in self.regions and 'H' in self.regions):
                for _ in range(n_synthetic_samples):
                    appended.append(minority_point + sample_inside_sphere(len(minority_point), r, self.p_norm))
            else:
                seed_score = rbf_score(minority_point, majority_points, self.gamma)

                lower_threshold = seed_score - self.threshold * (seed_score - np.min(scores + [seed_score]))
                higher_threshold = seed_score + self.threshold * (np.max(scores + [seed_score]) - seed_score)

                suitable_samples = [minority_point]

                for sample, score in zip(samples, scores):
                    if score <= lower_threshold:
                        case = 'L'
                    elif score >= higher_threshold:
                        case = 'H'
                    else:
                        case = 'E'

                    if case in self.regions:
                        suitable_samples.append(sample)

                suitable_samples = np.array(suitable_samples)

                if n_synthetic_samples <= len(suitable_samples):
                    replace = False
                else:
                    replace = True

                selected_samples = suitable_samples[
                    np.random.choice(len(suitable_samples), n_synthetic_samples, replace=replace)
                ]

                for sample in selected_samples:
                    appended.append(sample)

        majority_points += translations

        if len(appended) > 0:
            points = np.concatenate([majority_points, minority_points, appended])
            labels = np.concatenate([majority_labels, minority_labels, np.tile([minority_class], len(appended))])
        else:
            points = np.concatenate([majority_points, minority_points])
            labels = np.concatenate([majority_labels, minority_labels])

        return points, labels


class CCRv3:
    def __init__(self, energy, regions, gamma=None, n_samples=250, threshold=0.33,
                 p_norm=2, n=None, random_state=None):
        self.energy = energy
        self.gamma = gamma
        self.n_samples = n_samples
        self.threshold = threshold
        self.regions = regions
        self.p_norm = p_norm
        self.n = n
        self.random_state = random_state

    def fit_sample(self, X, y):
        np.random.seed(self.random_state)

        classes = np.unique(y)
        sizes = [sum(y == c) for c in classes]
        minority_class = classes[np.argmin(sizes)]

        minority_points = X[y == minority_class].copy()
        majority_points = X[y != minority_class].copy()
        minority_labels = y[y == minority_class].copy()
        majority_labels = y[y != minority_class].copy()

        if self.n is None:
            n = len(majority_points) - len(minority_points)
        else:
            n = self.n

        distances = np.zeros((len(minority_points), len(majority_points)))

        for i in range(len(minority_points)):
            for j in range(len(majority_points)):
                distances[i][j] = distance(minority_points[i], majority_points[j], self.p_norm)

        radii = np.zeros(len(minority_points))

        translations = np.zeros(majority_points.shape)

        for i in range(len(minority_points)):
            minority_point = minority_points[i]
            remaining_energy = self.energy
            radius = 0.0
            sorted_distances = np.argsort(distances[i])
            n_majority_points_within_radius = 0

            while True:
                if n_majority_points_within_radius == len(majority_points):
                    if n_majority_points_within_radius == 0:
                        radius_change = remaining_energy / (n_majority_points_within_radius + 1)
                    else:
                        radius_change = remaining_energy / n_majority_points_within_radius

                    radius += radius_change

                    break

                radius_change = remaining_energy / (n_majority_points_within_radius + 1)

                if distances[i, sorted_distances[n_majority_points_within_radius]] >= radius + radius_change:
                    radius += radius_change

                    break
                else:
                    if n_majority_points_within_radius == 0:
                        last_distance = 0.0
                    else:
                        last_distance = distances[i, sorted_distances[n_majority_points_within_radius - 1]]

                    radius_change = distances[i, sorted_distances[n_majority_points_within_radius]] - last_distance
                    radius += radius_change
                    remaining_energy -= radius_change * (n_majority_points_within_radius + 1)
                    n_majority_points_within_radius += 1

            radii[i] = radius

            for j in range(n_majority_points_within_radius):
                majority_point = majority_points[sorted_distances[j]]
                d = distances[i, sorted_distances[j]]

                while d < 1e-20:
                    majority_point += (1e-6 * np.random.rand(len(majority_point)) + 1e-6) * \
                                      np.random.choice([-1.0, 1.0], len(majority_point))
                    d = distance(minority_point, majority_point)

                translation = (radius - d) / d * (majority_point - minority_point)
                translations[sorted_distances[j]] += translation

        appended = []

        knn = NearestNeighbors(6).fit(X)

        for i in range(len(minority_points)):
            minority_point = minority_points[i]
            n_synthetic_samples = int(np.round(1.0 / (radii[i] * np.sum(1.0 / radii)) * n))
            r = radii[i]

            samples = []
            scores = []

            for _ in range(self.n_samples):
                sample = minority_point + sample_inside_sphere(len(minority_point), r, self.p_norm)
                score = rbf_score(sample, majority_points, self.gamma)

                samples.append(sample)
                scores.append(score)

            if self.gamma is None:
                for _ in range(n_synthetic_samples):
                    appended.append(minority_point + sample_inside_sphere(len(minority_point), r, self.p_norm))
            else:
                indices = knn.kneighbors([minority_point], return_distance=False)[0, 1:]
                n_majority_neighbors = sum(y[indices] != minority_class)

                if n_majority_neighbors in [0, 1]:
                    observation_type = 'safe'
                elif n_majority_neighbors in [2, 3]:
                    observation_type = 'borderline'
                elif n_majority_neighbors == 4:
                    observation_type = 'rare'
                elif n_majority_neighbors == 5:
                    observation_type = 'outlier'
                else:
                    raise ValueError

                observation_regions = self.regions[observation_type]

                if 'L' in observation_regions and 'E' in observation_regions and 'H' in observation_regions:
                    for _ in range(n_synthetic_samples):
                        appended.append(minority_point + sample_inside_sphere(len(minority_point), r, self.p_norm))
                else:
                    seed_score = rbf_score(minority_point, majority_points, self.gamma)

                    lower_threshold = seed_score - self.threshold * (seed_score - np.min(scores + [seed_score]))
                    higher_threshold = seed_score + self.threshold * (np.max(scores + [seed_score]) - seed_score)

                    suitable_samples = [minority_point]

                    for sample, score in zip(samples, scores):
                        if score <= lower_threshold:
                            case = 'L'
                        elif score >= higher_threshold:
                            case = 'H'
                        else:
                            case = 'E'

                        if case in observation_regions:
                            suitable_samples.append(sample)

                    suitable_samples = np.array(suitable_samples)

                    if n_synthetic_samples <= len(suitable_samples):
                        replace = False
                    else:
                        replace = True

                    selected_samples = suitable_samples[
                        np.random.choice(len(suitable_samples), n_synthetic_samples, replace=replace)
                    ]

                    for sample in selected_samples:
                        appended.append(sample)

        majority_points += translations

        if len(appended) > 0:
            points = np.concatenate([majority_points, minority_points, appended])
            labels = np.concatenate([majority_labels, minority_labels, np.tile([minority_class], len(appended))])
        else:
            points = np.concatenate([majority_points, minority_points])
            labels = np.concatenate([majority_labels, minority_labels])

        return points, labels


class CCRv4:
    def __init__(self, energy, gamma=None, n_samples=250, threshold=0.33,
                 regions='LEH', types=('safe', 'borderline', 'rare', 'outlier'),
                 p_norm=2, n=None, random_state=None):
        self.energy = energy
        self.gamma = gamma
        self.n_samples = n_samples
        self.threshold = threshold
        self.regions = regions
        self.types = types
        self.p_norm = p_norm
        self.n = n
        self.random_state = random_state

    def fit_sample(self, X, y):
        np.random.seed(self.random_state)

        classes = np.unique(y)
        sizes = [sum(y == c) for c in classes]
        minority_class = classes[np.argmin(sizes)]

        minority_points = X[y == minority_class].copy()
        majority_points = X[y != minority_class].copy()
        minority_labels = y[y == minority_class].copy()
        majority_labels = y[y != minority_class].copy()

        if self.n is None:
            n = len(majority_points) - len(minority_points)
        else:
            n = self.n

        distances = np.zeros((len(minority_points), len(majority_points)))

        for i in range(len(minority_points)):
            for j in range(len(majority_points)):
                distances[i][j] = distance(minority_points[i], majority_points[j], self.p_norm)

        radii = np.zeros(len(minority_points))

        translations = np.zeros(majority_points.shape)

        for i, minority_point in enumerate(minority_points):
            remaining_energy = self.energy
            radius = 0.0
            sorted_distances = np.argsort(distances[i])
            n_majority_points_within_radius = 0

            while True:
                if n_majority_points_within_radius == len(majority_points):
                    if n_majority_points_within_radius == 0:
                        radius_change = remaining_energy / (n_majority_points_within_radius + 1)
                    else:
                        radius_change = remaining_energy / n_majority_points_within_radius

                    radius += radius_change

                    break

                radius_change = remaining_energy / (n_majority_points_within_radius + 1)

                if distances[i, sorted_distances[n_majority_points_within_radius]] >= radius + radius_change:
                    radius += radius_change

                    break
                else:
                    if n_majority_points_within_radius == 0:
                        last_distance = 0.0
                    else:
                        last_distance = distances[i, sorted_distances[n_majority_points_within_radius - 1]]

                    radius_change = distances[i, sorted_distances[n_majority_points_within_radius]] - last_distance
                    radius += radius_change
                    remaining_energy -= radius_change * (n_majority_points_within_radius + 1)
                    n_majority_points_within_radius += 1

            radii[i] = radius

            for j in range(n_majority_points_within_radius):
                majority_point = majority_points[sorted_distances[j]]
                d = distances[i, sorted_distances[j]]

                while d < 1e-20:
                    majority_point += (1e-6 * np.random.rand(len(majority_point)) + 1e-6) * \
                                      np.random.choice([-1.0, 1.0], len(majority_point))
                    d = distance(minority_point, majority_point)

                translation = (radius - d) / d * (majority_point - minority_point)
                translations[sorted_distances[j]] += translation

        appended = []

        inverse_radius_sum = 0.0
        observation_types = []

        knn = NearestNeighbors(6).fit(X)

        for i, minority_point in enumerate(minority_points):
            indices = knn.kneighbors([minority_point], return_distance=False)[0, 1:]
            n_majority_neighbors = sum(y[indices] != minority_class)

            if n_majority_neighbors in [0, 1]:
                observation_type = 'safe'
            elif n_majority_neighbors in [2, 3]:
                observation_type = 'borderline'
            elif n_majority_neighbors == 4:
                observation_type = 'rare'
            elif n_majority_neighbors == 5:
                observation_type = 'outlier'
            else:
                raise ValueError

            observation_types.append(observation_type)

            if observation_type in self.types:
                inverse_radius_sum += radii[i] ** -1

        for i, minority_point in enumerate(minority_points):
            n_synthetic_samples = int(np.round((radii[i] ** -1) / inverse_radius_sum * n))
            r = radii[i]

            samples = []
            scores = []

            for _ in range(self.n_samples):
                sample = minority_point + sample_inside_sphere(len(minority_point), r, self.p_norm)
                score = rbf_score(sample, majority_points, self.gamma)

                samples.append(sample)
                scores.append(score)

            if self.gamma is None or ('L' in self.regions and 'E' in self.regions and 'H' in self.regions):
                for _ in range(n_synthetic_samples):
                    appended.append(minority_point + sample_inside_sphere(len(minority_point), r, self.p_norm))
            else:
                seed_score = rbf_score(minority_point, majority_points, self.gamma)

                lower_threshold = seed_score - self.threshold * (seed_score - np.min(scores + [seed_score]))
                higher_threshold = seed_score + self.threshold * (np.max(scores + [seed_score]) - seed_score)

                suitable_samples = [minority_point]

                for sample, score in zip(samples, scores):
                    if score <= lower_threshold:
                        case = 'L'
                    elif score >= higher_threshold:
                        case = 'H'
                    else:
                        case = 'E'

                    if case in self.regions:
                        suitable_samples.append(sample)

                suitable_samples = np.array(suitable_samples)

                if n_synthetic_samples <= len(suitable_samples):
                    replace = False
                else:
                    replace = True

                selected_samples = suitable_samples[
                    np.random.choice(len(suitable_samples), n_synthetic_samples, replace=replace)
                ]

                for sample in selected_samples:
                    appended.append(sample)

        majority_points += translations

        if len(appended) > 0:
            points = np.concatenate([majority_points, minority_points, appended])
            labels = np.concatenate([majority_labels, minority_labels, np.tile([minority_class], len(appended))])
        else:
            points = np.concatenate([majority_points, minority_points])
            labels = np.concatenate([majority_labels, minority_labels])

        return points, labels


class CCRv5:
    def __init__(self, energy, gamma=None, n_samples=100, threshold=0.33,
                 regions='E', p_norm=2, n=None, random_state=None):
        self.energy = energy
        self.gamma = gamma
        self.n_samples = n_samples
        self.threshold = threshold
        self.regions = regions
        self.p_norm = p_norm
        self.n = n
        self.random_state = random_state

    def fit_sample(self, X, y):
        np.random.seed(self.random_state)

        classes = np.unique(y)
        sizes = [sum(y == c) for c in classes]
        minority_class = classes[np.argmin(sizes)]

        minority_points = X[y == minority_class].copy()
        majority_points = X[y != minority_class].copy()
        minority_labels = y[y == minority_class].copy()
        majority_labels = y[y != minority_class].copy()

        if self.n is None:
            n = len(majority_points) - len(minority_points)
        else:
            n = self.n

        if self.gamma == 'auto':
            gamma = 1.06 * np.std(majority_points) * len(majority_points) ** (-1 / 5)
        else:
            gamma = self.gamma

        distances = np.zeros((len(minority_points), len(majority_points)))

        for i in range(len(minority_points)):
            for j in range(len(majority_points)):
                distances[i][j] = distance(minority_points[i], majority_points[j], self.p_norm)

        radii = np.zeros(len(minority_points))

        translations = np.zeros(majority_points.shape)

        for i in range(len(minority_points)):
            minority_point = minority_points[i]
            remaining_energy = self.energy
            radius = 0.0
            sorted_distances = np.argsort(distances[i])
            n_majority_points_within_radius = 0

            while True:
                if n_majority_points_within_radius == len(majority_points):
                    if n_majority_points_within_radius == 0:
                        radius_change = remaining_energy / (n_majority_points_within_radius + 1)
                    else:
                        radius_change = remaining_energy / n_majority_points_within_radius

                    radius += radius_change

                    break

                radius_change = remaining_energy / (n_majority_points_within_radius + 1)

                if distances[i, sorted_distances[n_majority_points_within_radius]] >= radius + radius_change:
                    radius += radius_change

                    break
                else:
                    if n_majority_points_within_radius == 0:
                        last_distance = 0.0
                    else:
                        last_distance = distances[i, sorted_distances[n_majority_points_within_radius - 1]]

                    radius_change = distances[i, sorted_distances[n_majority_points_within_radius]] - last_distance
                    radius += radius_change
                    remaining_energy -= radius_change * (n_majority_points_within_radius + 1)
                    n_majority_points_within_radius += 1

            radii[i] = radius

            for j in range(n_majority_points_within_radius):
                majority_point = majority_points[sorted_distances[j]]
                d = distances[i, sorted_distances[j]]

                while d < 1e-20:
                    majority_point += (1e-6 * np.random.rand(len(majority_point)) + 1e-6) * \
                                      np.random.choice([-1.0, 1.0], len(majority_point))
                    d = distance(minority_point, majority_point)

                translation = (radius - d) / d * (majority_point - minority_point)
                translations[sorted_distances[j]] += translation

        appended = []

        if gamma is None or ('L' in self.regions and 'E' in self.regions and 'H' in self.regions):
            for i in range(len(minority_points)):
                minority_point = minority_points[i]
                n_synthetic_samples = int(np.round(1.0 / (radii[i] * np.sum(1.0 / radii)) * n))
                r = radii[i]

                for _ in range(n_synthetic_samples):
                    appended.append(minority_point + sample_inside_sphere(len(minority_point), r, self.p_norm))
        else:
            samples = []
            scores = []

            for i in range(len(minority_points)):
                minority_point = minority_points[i]
                n_synthetic_samples = int(np.round(1.0 / (radii[i] * np.sum(1.0 / radii)) * n))
                r = radii[i]

                for _ in range(self.n_samples):
                    sample = minority_point + sample_inside_sphere(len(minority_point), r, self.p_norm)
                    score = rbf_score(sample, majority_points, gamma)

                    samples.append(sample)
                    scores.append(score)

            lower_threshold = np.mean(scores) - self.threshold * (np.mean(scores) - np.min(scores))
            higher_threshold = np.mean(scores) + self.threshold * (np.max(scores) - np.mean(scores))

            suitable_samples = []

            for sample, score in zip(samples, scores):
                if score <= lower_threshold:
                    case = 'L'
                elif score >= higher_threshold:
                    case = 'H'
                else:
                    case = 'E'

                if case in self.regions:
                    suitable_samples.append(sample)

            suitable_samples = np.array(suitable_samples)

            if n <= len(suitable_samples):
                replace = False
            else:
                replace = True

            selected_samples = suitable_samples[
                np.random.choice(len(suitable_samples), n, replace=replace)
            ]

            for sample in selected_samples:
                appended.append(sample)

        majority_points += translations

        if len(appended) > 0:
            points = np.concatenate([majority_points, minority_points, appended])
            labels = np.concatenate([majority_labels, minority_labels, np.tile([minority_class], len(appended))])
        else:
            points = np.concatenate([majority_points, minority_points])
            labels = np.concatenate([majority_labels, minority_labels])

        return points, labels
