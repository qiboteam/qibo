from itertools import product

import matplotlib.pyplot as plt
import numpy as np


def create_dataset(name, grid=None, samples=1000, seed=0):
    """Function to create training and test sets for classifying.

    Args:
        name (str): Name of the problem to create the dataset, to choose between
            ['circle', '3 circles', 'square', '4 squares', 'crown', 'tricrown', 'wavy lines'].
        grid (int): Number of points in one direction defining the grid of points.
            If not specified, the dataset does not follow a regular grid.
        samples (int): Number of points in the set, randomly located.
            This argument is ignored if grid is specified.
        seed (int): Random seed

    Returns:
        Dataset for the given problem (x, y)
    """
    if grid is None:
        np.random.seed(seed)
        points = 1 - 2 * np.random.rand(samples, 2)
    else:
        x = np.linspace(-1, 1, grid)
        points = np.array(list(product(x, x)))
    creator = globals()[f"_{name}"]

    x, y = creator(points)
    return x, y


def create_target(name):
    """Function to create target states for classification.

    Args:
        name (str): Name of the problem to create the target states, to choose between
            ['circle', '3 circles', 'square', '4 squares', 'crown', 'tricrown', 'wavy lines']

    Returns:
        List of numpy arrays encoding target states that depend only on the number of classes of the given problem
    """
    if name in ["circle", "square", "crown"]:
        targets = [np.array([1, 0], dtype="complex"), np.array([0, 1], dtype="complex")]
    elif name in ["tricrown"]:
        targets = [
            np.array([1, 0], dtype="complex"),
            np.array([np.cos(np.pi / 3), np.sin(np.pi / 3)], dtype="complex"),
            np.array([np.cos(np.pi / 3), -np.sin(np.pi / 3)], dtype="complex"),
        ]
    elif name in ["4_squares", "wavy_lines", "3_circles"]:
        targets = [
            np.array([1, 0], dtype=complex),
            np.array([1 / np.sqrt(3), np.sqrt(2 / 3)], dtype=complex),
            np.array(
                [1 / np.sqrt(3), np.exp(1j * 2 * np.pi / 3) * np.sqrt(2 / 3)],
                dtype=complex,
            ),
            np.array(
                [1 / np.sqrt(3), np.exp(-1j * 2 * np.pi / 3) * np.sqrt(2 / 3)],
                dtype=complex,
            ),
        ]

    else:
        raise NotImplementedError("This dataset is not implemented")

    return targets


def fig_template(name):
    """Function to create templates for plotting results of classification.

    Args:
        name (str): Name of the problem to create the figure template, to choose between
            ['circle', '3 circles', 'square', '4 squares', 'crown', 'tricrown', 'wavy lines']

    Returns:
        matplotlib.figure, matplotlib.axis with the templates for plotting results.

    """
    fig, axs = plt.subplots(ncols=2, figsize=(9, 4))
    if name == "circle":
        for ax in axs:
            circle = plt.Circle(
                (0, 0), np.sqrt(2 / np.pi), color="black", fill=False, zorder=10
            )
            ax.add_artist(circle)

    elif name == "3_circles":
        centers = np.array([[-1, 1], [1, 0], [-0.5, -0.5]])
        radii = np.array([1, np.sqrt(6 / np.pi - 1), 1 / 2])
        for c, r in zip(centers, radii):
            for ax in axs:
                circle = plt.Circle(c, r, color="black", fill=False, zorder=10)
                ax.add_artist(circle)

    elif name == "square":
        p = 0.5 * np.sqrt(2)
        for ax in axs:
            ax.plot([-p, p, p, -p, -p], [-p, -p, p, p, -p], color="black", zorder=10)

    elif name == "4_squares":
        for ax in axs:
            ax.plot([0, 0], [-1, 1], color="black", zorder=10)
            ax.plot([-1, 1], [0, 0], color="black", zorder=10)

    elif name == "crown" or name == "tricrown":
        centers = [[0, 0], [0, 0]]
        radii = [np.sqrt(0.8), np.sqrt(0.8 - 2 / np.pi)]
        for c, r in zip(centers, radii):
            for ax in axs:
                circle = plt.Circle(c, r, color="black", fill=False, zorder=10)
                ax.add_artist(circle)

    elif name == "wavy_lines":
        freq = 1

        def fun1(s):
            return s + np.sin(freq * np.pi * s)

        def fun2(s):
            return -s + np.sin(freq * np.pi * s)

        x = np.linspace(-1, 1)
        for ax in axs:
            ax.plot(x, np.clip(fun1(x), -1, 1), color="black", zorder=10)
            ax.plot(x, np.clip(fun2(x), -1, 1), color="black", zorder=10)

    axs[0].set(xlabel=r"$x_0$", ylabel=r"$x_1$", xlim=[-1, 1], ylim=[-1, 1])
    axs[0].axis("equal")
    axs[1].set(xlabel=r"$x_0$", xlim=[-1, 1], ylim=[-1, 1])
    axs[1].axis("equal")

    return fig, axs


def world_map_template():
    """Function to create templates for plotting the Bloch Sphere after classification.

    Returns:
        matplotlib.figure, matplotlib.axis with the templates for plotting results
    """
    fig, ax = plt.subplots(figsize=(20, 10))
    ax.plot(
        laea_x(np.pi, np.linspace(-np.pi / 2, np.pi / 2)),
        laea_y(np.pi, np.linspace(-np.pi / 2, np.pi / 2)),
        color="k",
        zorder=10,
    )
    ax.plot(
        laea_x(-np.pi, np.linspace(-np.pi / 2, np.pi / 2)),
        laea_y(-np.pi, np.linspace(-np.pi / 2, np.pi / 2)),
        color="k",
        zorder=10,
    )
    ax.plot(
        laea_x(np.pi / 3, np.linspace(-np.pi / 2, np.pi / 2)),
        laea_y(np.pi / 3, np.linspace(-np.pi / 2, np.pi / 2)),
        color="k",
        zorder=10,
    )
    ax.plot(
        laea_x(-np.pi / 3, np.linspace(-np.pi / 2, np.pi / 2)),
        laea_y(-np.pi / 3, np.linspace(-np.pi / 2, np.pi / 2)),
        color="k",
        zorder=10,
    )
    ax.plot(
        laea_x(2 * np.pi / 3, np.linspace(-np.pi / 2, np.pi / 2)),
        laea_y(2 * np.pi / 3, np.linspace(-np.pi / 2, np.pi / 2)),
        color="k",
        zorder=10,
    )
    ax.plot(
        laea_x(-2 * np.pi / 3, np.linspace(-np.pi / 2, np.pi / 2)),
        laea_y(-2 * np.pi / 3, np.linspace(-np.pi / 2, np.pi / 2)),
        color="k",
        zorder=10,
    )
    ax.plot(
        laea_x(0, np.linspace(-np.pi / 2, np.pi / 2)),
        laea_y(0, np.linspace(-np.pi / 2, np.pi / 2)),
        color="k",
        zorder=10,
    )
    ax.plot(
        laea_x(np.linspace(-np.pi, np.pi), 0),
        laea_y(np.linspace(-np.pi, np.pi), 0),
        color="k",
        zorder=10,
    )
    ax.plot(
        laea_x(np.linspace(-np.pi, np.pi), np.pi / 6),
        laea_y(np.linspace(-np.pi, np.pi), np.pi / 6),
        color="k",
        zorder=10,
    )
    ax.plot(
        laea_x(np.linspace(-np.pi, np.pi), -np.pi / 6),
        laea_y(np.linspace(-np.pi, np.pi), -np.pi / 6),
        color="k",
        zorder=10,
    )
    ax.plot(
        laea_x(np.linspace(-np.pi, np.pi), np.pi / 3),
        laea_y(np.linspace(-np.pi, np.pi), np.pi / 3),
        color="k",
        zorder=10,
    )
    ax.plot(
        laea_x(np.linspace(-np.pi, np.pi), -np.pi / 3),
        laea_y(np.linspace(-np.pi, np.pi), -np.pi / 3),
        color="k",
        zorder=10,
    )
    ax.text(0, 1.47, r"$|0\rangle$", fontsize=20)
    ax.text(0, -1.53, r"$|1\rangle$", fontsize=20)
    ax.text(0.05, 0.05, r"$|+\rangle$", fontsize=20)
    ax.text(2.9, 0, r"$|-\rangle$", fontsize=20)
    ax.text(-3.2, 0, r"$|-\rangle$", fontsize=20)

    return fig, ax


def laea_x(lamb, phi):
    """Auxiliary function to represent spheres in a 2D map - x axis. Inspired in the Hammer projection.

    Args:
        lamb (float): longitude.
        phi (float): latitude.

    Returns:
        x-axis in Hammer projection.
    """
    return (
        2
        * np.sqrt(2)
        * np.cos(phi)
        * np.sin(lamb / 2)
        / np.sqrt(1 + np.cos(phi) * np.cos(lamb / 2))
    )


def laea_y(lamb, phi):
    """Auxiliary function to represent spheres in a 2D map - y axis. Inspired in the Hammer projection.

    Args:
        lamb (float): longitude.
        phi (float): latitude.

    Returns:
        y-axis in Hammer projection.
    """
    return np.sqrt(2) * np.sin(phi) / np.sqrt(1 + np.cos(phi) * np.cos(lamb / 2))


def _circle(points):
    labels = np.zeros(len(points), dtype=np.int32)
    ids = np.where(np.linalg.norm(points, axis=1) > np.sqrt(2 / np.pi))
    labels[ids] = 1

    return points, labels


def _3_circles(points):
    centers = np.array([[-1, 1], [1, 0], [-0.5, -0.5]])
    radii = np.array([1, np.sqrt(6 / np.pi - 1), 1 / 2])
    labels = np.zeros(len(points), dtype=np.int32)
    for j, (c, r) in enumerate(zip(centers, radii)):
        ids = np.where(np.linalg.norm(points - c, axis=1) < r)
        labels[ids] = 1 + j

    return points, labels


def _square(points):
    labels = np.zeros(len(points), dtype=np.int32)
    ids = np.where(np.max(np.abs(points), axis=1) > 0.5 * np.sqrt(2))
    labels[ids] = 1

    return points, labels


def _4_squares(points):
    labels = np.zeros(len(points), dtype=np.int32)
    ids = np.where(np.logical_and(points[:, 0] < 0, points[:, 1] > 0))
    labels[ids] = 1
    ids = np.where(np.logical_and(points[:, 0] > 0, points[:, 1] < 0))
    labels[ids] = 2
    ids = np.where(np.logical_and(points[:, 0] > 0, points[:, 1] > 0))
    labels[ids] = 3

    return points, labels


def _crown(points):
    c = [[0, 0], [0, 0]]
    r = [np.sqrt(0.8), np.sqrt(0.8 - 2 / np.pi)]
    labels = np.zeros(len(points), dtype=np.int32)
    ids = np.where(
        np.logical_and(
            np.linalg.norm(points - [c[0]], axis=1) < r[0],
            np.linalg.norm(points - [c[1]], axis=1) > r[1],
        )
    )
    labels[ids] = 1

    return points, labels


def _tricrown(points):
    c = [[0, 0], [0, 0]]
    r = [np.sqrt(0.8), np.sqrt(0.8 - 2 / np.pi)]
    labels = np.zeros(len(points), dtype=np.int32)
    ids = np.where(np.linalg.norm(points - [c[0]], axis=1) > r[0])
    labels[ids] = 2
    ids = np.where(
        np.logical_and(
            np.linalg.norm(points - [c[0]], axis=1) < r[0],
            np.linalg.norm(points - [c[1]], axis=1) > r[1],
        )
    )
    labels[ids] = 1

    return points, labels


def _wavy_lines(points):
    freq = 1

    def fun1(s):
        return s + np.sin(freq * np.pi * s)

    def fun2(s):
        return -s + np.sin(freq * np.pi * s)

    labels = np.zeros(len(points), dtype=np.int32)
    ids = np.where(
        np.logical_and(
            points[:, 1] < fun1(points[:, 0]), points[:, 1] > fun2(points[:, 0])
        )
    )
    labels[ids] = 1
    ids = np.where(
        np.logical_and(
            points[:, 1] > fun1(points[:, 0]), points[:, 1] < fun2(points[:, 0])
        )
    )
    labels[ids] = 2
    ids = np.where(
        np.logical_and(
            points[:, 1] > fun1(points[:, 0]), points[:, 1] > fun2(points[:, 0])
        )
    )
    labels[ids] = 3

    return points, labels
