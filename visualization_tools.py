import datasets
import matplotlib.pyplot as plt
import numpy as np

from algorithm import distance, rbf_score
from collections import Counter
from imblearn.under_sampling import RandomUnderSampler
from matplotlib.patches import Polygon
from scipy.spatial import ConvexHull
from sklearn.manifold import TSNE
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from visualize_preliminary_energy import VISUALIZATIONS_PATH


ALPHA = 0.9
BACKGROUND_COLOR = '#EEEEEE'
BORDER_COLOR = '#161921'
COLOR_MAJORITY = '#C44E52'
COLOR_MINORITY = '#4C72B0'
COLOR_NEUTRAL = '#F2F2F2'
FIGURE_SIZE = (6, 6)
LINE_WIDTH = 1.0
MARGIN = 0.05
MARKER_SIZE = 75
MARKER_SYMBOL = 'o'
ORIGINAL_EDGE_COLOR = '#F2F2F2'
OVERSAMPLED_EDGE_COLOR = '#262223'
POTENTIAL_GRID_N = 150
REGIONS_COLORS = ['red', 'yellow', 'green']
REGIONS_STEPS = 1000
REGIONS_THRESHOLD = 0.33


def visualize(X, y, appended=None, gamma=None, radii=None,
              regions_center=None, regions_radius=None,
              p_norm=2, file_name=None, lim=None):
    assert len(np.unique(y)) == 2
    assert X.shape[1] == 2

    if appended is not None:
        assert appended.shape[1] == 2

    plt.style.use('ggplot')

    classes = np.unique(y)
    sizes = [sum(y == c) for c in classes]
    minority_class = classes[np.argmin(sizes)]

    minority_points = X[y == minority_class].copy()
    majority_points = X[y != minority_class].copy()

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111)

    ax.set_xticks([])
    ax.set_yticks([])

    for key in ax.spines.keys():
        ax.spines[key].set_color(BORDER_COLOR)

    ax.tick_params(colors=BORDER_COLOR)
    ax.set_facecolor(BACKGROUND_COLOR)

    x_limits = [np.min(X[:, 0]), np.max(X[:, 0])]
    y_limits = [np.min(X[:, 1]), np.max(X[:, 1])]

    x_spread = np.abs(x_limits[1] - x_limits[0])
    y_spread = np.abs(y_limits[1] - y_limits[0])

    x_limits[0] = x_limits[0] - MARGIN * x_spread
    y_limits[0] = y_limits[0] - MARGIN * y_spread
    x_limits[1] = x_limits[1] + MARGIN * x_spread
    y_limits[1] = y_limits[1] + MARGIN * y_spread

    if lim is None:
        plt.xlim(x_limits)
        plt.ylim(y_limits)
    else:
        plt.xlim(lim)
        plt.ylim(lim)

    if regions_center is not None and regions_radius is not None and gamma is not None:
        seed_score = rbf_score(regions_center, minority_points, gamma, p_norm)

        l_points = []
        e_points = []
        h_points = []

        pp = []
        scores = []

        for x_i in np.linspace(regions_center[0] - regions_radius, regions_center[0] + regions_radius, REGIONS_STEPS):
            for y_i in np.linspace(regions_center[1] - regions_radius, regions_center[1] + regions_radius, REGIONS_STEPS):
                p_i = np.array([x_i, y_i])

                if distance(p_i, regions_center) <= regions_radius:
                    pp.append(p_i)
                    scores.append(rbf_score(p_i, minority_points, gamma, p_norm))

        lower_threshold = seed_score - REGIONS_THRESHOLD * (seed_score - np.min(scores + [seed_score]))
        higher_threshold = seed_score + REGIONS_THRESHOLD * (np.max(scores + [seed_score]) - seed_score)

        for sample, score in zip(pp, scores):
            if score <= lower_threshold:
                l_points.append(sample)
            elif score >= higher_threshold:
                h_points.append(sample)
            else:
                e_points.append(sample)

        for region_points, color in zip([l_points, e_points, h_points], REGIONS_COLORS):
            pp = np.array(region_points)
            hull = ConvexHull(pp)

            cent = np.mean(pp, 0)
            pts = []

            for pt in pp[hull.simplices]:
                pts.append(pt[0].tolist())
                pts.append(pt[1].tolist())

            pts.sort(key=lambda p: np.arctan2(p[1] - cent[1], p[0] - cent[0]))
            pts = pts[0::2]
            pts.insert(len(pts), pts[0])

            poly = Polygon((np.array(pts) - cent) + cent, facecolor=color, alpha=1.0, zorder=-20)
            poly.set_capstyle('round')

            ax.add_patch(poly)

        circle = plt.Circle(
            regions_center, regions_radius, color=BORDER_COLOR,
            alpha=1.0, linewidth=LINE_WIDTH, fill=False, zorder=-10
        )

        ax.add_artist(circle)

    if radii is not None:
        for point, radius in zip(minority_points, radii):
            circle = plt.Circle(point, radius, color=BORDER_COLOR, alpha=0.2, linewidth=LINE_WIDTH, fill=True)

            ax.add_artist(circle)

    plt.scatter(
        majority_points[:, 0], majority_points[:, 1],
        s=MARKER_SIZE, c=COLOR_MAJORITY, linewidths=LINE_WIDTH,
        alpha=ALPHA, marker=MARKER_SYMBOL, edgecolors=ORIGINAL_EDGE_COLOR
    )

    plt.scatter(
        minority_points[:, 0], minority_points[:, 1],
        s=MARKER_SIZE, c=COLOR_MINORITY, linewidths=LINE_WIDTH,
        alpha=ALPHA, marker=MARKER_SYMBOL, edgecolors=ORIGINAL_EDGE_COLOR
    )

    if appended is not None:
        plt.scatter(
            appended[:, 0], appended[:, 1],
            s=MARKER_SIZE, c=COLOR_MINORITY, linewidths=LINE_WIDTH,
            alpha=ALPHA, marker=MARKER_SYMBOL, edgecolors=OVERSAMPLED_EDGE_COLOR
        )

    if gamma is not None:
        x_cont = np.linspace(x_limits[0], x_limits[1], POTENTIAL_GRID_N + 1)
        y_cont = np.linspace(y_limits[0], y_limits[1], POTENTIAL_GRID_N + 1)

        X_cont, Y_cont = np.meshgrid(x_cont, y_cont)

        Z = np.zeros(X_cont.shape)

        for i, x1 in enumerate(x_cont):
            for j, x2 in enumerate(y_cont):
                Z[j][i] = rbf_score(
                    np.array([x1, x2]),
                    minority_points,
                    gamma,
                    p_norm
                )

        plt.contour(X_cont, Y_cont, Z)

    if file_name is not None:
        VISUALIZATIONS_PATH.mkdir(exist_ok=True, parents=True)

        plt.savefig(VISUALIZATIONS_PATH / f'{file_name}.pdf', bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()


def prepare_data(dataset_name, n_minority_samples=20, scaler='MinMax'):
    dataset = datasets.load(dataset_name)
    (X_train, y_train), (X_test, y_test) = dataset[0][0], dataset[0][1]
    X, y = np.concatenate([X_train, X_test]), np.concatenate([y_train, y_test])

    minority_class = Counter(y).most_common()[1][0]
    majority_class = Counter(y).most_common()[0][0]

    n_minority = Counter(y).most_common()[1][1]
    n_majority = Counter(y).most_common()[0][1]

    X, y = RandomUnderSampler(
        sampling_strategy={
            minority_class: np.min([n_minority, n_minority_samples]),
            majority_class: n_majority
        },
        random_state=42,
    ).fit_sample(X, y)

    X = TSNE(n_components=2, random_state=42).fit_transform(X)

    if scaler == 'MinMax':
        X = MinMaxScaler().fit_transform(X)
    elif scaler == 'Standard':
        X = StandardScaler().fit_transform(X)
    else:
        raise NotImplementedError

    return X, y
