import numpy as np


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


def rbf_score(point, minority_points, majority_points, gamma, p_norm=2, scoring='minority'):
    assert scoring in ['minority', 'majority', 'relative']

    result = 0.0

    if scoring == 'minority':
        for reference_point in minority_points:
            result += rbf(distance(point, reference_point, p_norm), gamma)
    elif scoring == 'majority':
        for reference_point in majority_points:
            result += rbf(distance(point, reference_point, p_norm), gamma)
    elif scoring == 'relative':
        for reference_point in majority_points:
            result += rbf(distance(point, reference_point, p_norm), gamma)

        for reference_point in minority_points:
            result -= rbf(distance(point, reference_point, p_norm), gamma)
    else:
        raise NotImplementedError

    return result


class RBCCR:
    def __init__(self, energy, gamma=None, n_samples=100, threshold=0.33,
                 regions='E', scoring='minority', p_norm=2,
                 minority_class=None, n=None, random_state=None):
        self.energy = energy
        self.gamma = gamma
        self.n_samples = n_samples
        self.threshold = threshold
        self.regions = regions
        self.scoring = scoring
        self.p_norm = p_norm
        self.minority_class = minority_class
        self.n = n
        self.random_state = random_state

    def fit_sample(self, X, y):
        np.random.seed(self.random_state)

        if self.minority_class is None:
            classes = np.unique(y)
            sizes = [sum(y == c) for c in classes]
            minority_class = classes[np.argmin(sizes)]
        else:
            minority_class = self.minority_class

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

        sample_ratios = np.array([1.0 / (radii[i] * np.sum(1.0 / radii)) for i in range(len(minority_points))])
        n_synthetic_samples = np.round(sample_ratios * n).astype(np.int64)

        if np.sum(n_synthetic_samples) < n:
            for i in np.argsort(sample_ratios)[::-1]:
                n_synthetic_samples[i] += 1

                if np.sum(n_synthetic_samples) >= n:
                    break
        elif np.sum(n_synthetic_samples) > n:
            for i in np.argsort(sample_ratios):
                if n_synthetic_samples[i] > 0:
                    n_synthetic_samples[i] -= 1

                if np.sum(n_synthetic_samples) <= n:
                    break

        for i in range(len(minority_points)):
            minority_point = minority_points[i]
            r = radii[i]

            if self.gamma is None or ('L' in self.regions and 'E' in self.regions and 'H' in self.regions):
                for _ in range(n_synthetic_samples[i]):
                    appended.append(minority_point + sample_inside_sphere(len(minority_point), r, self.p_norm))
            else:
                samples = []
                scores = []

                for _ in range(self.n_samples):
                    sample = minority_point + sample_inside_sphere(len(minority_point), r, self.p_norm)
                    score = rbf_score(sample, minority_points, majority_points, self.gamma, self.p_norm, self.scoring)

                    samples.append(sample)
                    scores.append(score)

                seed_score = rbf_score(minority_point, minority_points, majority_points, self.gamma, self.p_norm, self.scoring)

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

                if n_synthetic_samples[i] <= len(suitable_samples):
                    replace = False
                else:
                    replace = True

                selected_samples = suitable_samples[
                    np.random.choice(len(suitable_samples), n_synthetic_samples[i], replace=replace)
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


class MultiClassRBCCR:
    def __init__(self, energy, gamma=None, n_samples=100, threshold=0.33,
                 regions='E', scoring='minority', p_norm=2,
                 random_state=None, method='sampling'):
        assert method in ['sampling', 'complete']

        self.energy = energy
        self.gamma = gamma
        self.n_samples = n_samples
        self.threshold = threshold
        self.regions = regions
        self.scoring = scoring
        self.p_norm = p_norm
        self.random_state = random_state
        self.method = method

    def fit_sample(self, X, y):
        np.random.seed(self.random_state)

        classes = np.unique(y)
        sizes = np.array([sum(y == c) for c in classes])
        indices = np.argsort(sizes)[::-1]
        classes = classes[indices]
        observations = {c: X[y == c] for c in classes}
        n_max = max(sizes)

        if self.method == 'sampling':
            for i in range(1, len(classes)):
                current_class = classes[i]
                n = n_max - len(observations[current_class])

                used_observations = {}
                unused_observations = {}

                for j in range(0, i):
                    all_indices = list(range(len(observations[classes[j]])))
                    used_indices = np.random.choice(all_indices, int(n_max / i), replace=False)

                    used_observations[classes[j]] = [
                        observations[classes[j]][idx] for idx in all_indices if idx in used_indices
                    ]
                    unused_observations[classes[j]] = [
                        observations[classes[j]][idx] for idx in all_indices if idx not in used_indices
                    ]

                used_observations[current_class] = observations[current_class]
                unused_observations[current_class] = []

                for j in range(i + 1, len(classes)):
                    used_observations[classes[j]] = []
                    unused_observations[classes[j]] = observations[classes[j]]

                unpacked_points, unpacked_labels = MultiClassRBCCR._unpack_observations(used_observations)

                ccr = RBCCR(
                    energy=self.energy, gamma=self.gamma, n_samples=self.n_samples,
                    threshold=self.threshold, regions=self.regions, scoring=self.scoring,
                    p_norm=self.p_norm, minority_class=current_class, n=n
                )

                oversampled_points, oversampled_labels = ccr.fit_sample(unpacked_points, unpacked_labels)

                observations = {}

                for cls in classes:
                    class_oversampled_points = oversampled_points[oversampled_labels == cls]
                    class_unused_points = unused_observations[cls]

                    if len(class_oversampled_points) == 0 and len(class_unused_points) == 0:
                        observations[cls] = np.array([])
                    elif len(class_oversampled_points) == 0:
                        observations[cls] = class_unused_points
                    elif len(class_unused_points) == 0:
                        observations[cls] = class_oversampled_points
                    else:
                        observations[cls] = np.concatenate([class_oversampled_points, class_unused_points])
        else:
            for i in range(1, len(classes)):
                current_class = classes[i]
                n = n_max - len(observations[current_class])

                unpacked_points, unpacked_labels = MultiClassRBCCR._unpack_observations(observations)

                ccr = RBCCR(
                    energy=self.energy, gamma=self.gamma, n_samples=self.n_samples,
                    threshold=self.threshold, regions=self.regions, scoring=self.scoring,
                    p_norm=self.p_norm, minority_class=current_class, n=n
                )

                oversampled_points, oversampled_labels = ccr.fit_sample(unpacked_points, unpacked_labels)

                observations = {cls: oversampled_points[oversampled_labels == cls] for cls in classes}

        unpacked_points, unpacked_labels = MultiClassRBCCR._unpack_observations(observations)

        return unpacked_points, unpacked_labels

    @staticmethod
    def _unpack_observations(observations):
        unpacked_points = []
        unpacked_labels = []

        for cls in observations.keys():
            if len(observations[cls]) > 0:
                unpacked_points.append(observations[cls])
                unpacked_labels.append(np.tile([cls], len(observations[cls])))

        unpacked_points = np.concatenate(unpacked_points)
        unpacked_labels = np.concatenate(unpacked_labels)

        return unpacked_points, unpacked_labels
