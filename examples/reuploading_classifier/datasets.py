import numpy as np
from itertools import product
import matplotlib.pyplot as plt


def create_dataset(name, grid = None, samples = 1000, seed=0):
    if grid == None:
        np.random.seed(seed)
        points = 1 - 2 * np.random.rand(samples, 2)
    else:
        x = np.linspace(-1, 1, grid)
        points = np.array(list(product(x, x)))
    if name == 'circle':
        creator = _circle
    if name == '3 circles':
        creator = _3circles
    if name == 'square':
        creator = _square
    if name == '4 squares':
        creator = _4squares
    if name == 'crown':
        creator = _crown
    if name == 'tricrown':
        creator = _tricrown
    if name == 'wavy lines':
        creator = _wavy_lines

    x, y = creator(points)
    return x, y


def create_target(name):
    if name in ['circle', 'square', 'crown']:
        targets = [np.array([1, 0], dtype='complex'), np.array([0, 1], dtype='complex')]
    elif name in ['tricrown']:
        targets = [np.array([1, 0], dtype='complex'), np.array([np.cos(np.pi / 3), np.sin(np.pi / 3)], dtype='complex'),
                   np.array([np.cos(np.pi / 3), -np.sin(np.pi / 3)], dtype='complex')]
    elif name in ['4 squares', 'wavy lines', '3 circles']:
        targets = [np.array([1, 0], dtype=complex),
                   np.array([1 / np.sqrt(3), np.sqrt(2 / 3)], dtype=complex),
                   np.array([1 / np.sqrt(3), np.exp(1j * 2 * np.pi / 3) * np.sqrt(2 / 3)], dtype=complex),
                   np.array([1 / np.sqrt(3), np.exp(-1j * 2 * np.pi / 3) * np.sqrt(2 / 3)], dtype=complex)]

    return targets

def fig_template(name):
    fig, axs = plt.subplots(ncols=2, figsize=(9, 4))
    if name == 'circle':
        for ax in axs:
            circle = plt.Circle((0, 0), np.sqrt(2 / np.pi), color='black', fill=False, zorder=10)
            ax.add_artist(circle)

    elif name == '3 circles':
        centers = np.array([[-1, 1], [1, 0], [-.5, -.5]])
        radii = np.array([1, np.sqrt(6 / np.pi - 1), 1 / 2])
        for (c, r) in zip(centers, radii):
            for ax in axs:
                circle = plt.Circle(c, r, color='black', fill=False, zorder=10)
                ax.add_artist(circle)

    elif name == 'square':
        p = .5 * np.sqrt(2)
        for ax in axs:
            ax.plot([-p, p, p, -p], [-p, -p, p, p], color='black', zorder=10)

    elif name == '4 squares':
        for ax in axs:
            ax.plot([0, 0], [-1, 1], color='black', zorder=10)
            ax.plot([-1, 1], [0, 0], color='black', zorder=10)

    elif name == 'crown' or name == 'tricrown':
        centers = [[0, 0], [0, 0]]
        radii = [np.sqrt(.8), np.sqrt(.8 - 2 / np.pi)]
        for (c, r) in zip(centers, radii):
            for ax in axs:
                circle = plt.Circle(c, r, color='black', fill=False, zorder=10)
                ax.add_artist(circle)

    elif name == 'wavy lines':
        freq = 1

        def fun1(s):
            return s + np.sin(freq * np.pi * s)

        def fun2(s):
            return -s + np.sin(freq * np.pi * s)
        x = np.linspace(-1, 1)
        for ax in axs:
            ax.plot(x, fun1(x), color='black', zorder=10)
            ax.plot(x, fun2(x), color='black', zorder=10)

    axs[0].set(xlabel=r'$x_0$', ylabel=r'$x_1$', xlim=[-1, 1], ylim=[-1, 1])
    axs[0].axis('equal')
    axs[1].set(xlabel=r'$x_0$', xlim=[-1, 1], ylim=[-1, 1])
    axs[1].axis('equal')

    return fig, axs

def world_map_template():
    fig, ax = plt.subplots(figsize = (20, 10))
    ax.plot(laea_x(np.pi, np.linspace(-np.pi / 2, np.pi / 2)), laea_y(np.pi, np.linspace(-np.pi / 2, np.pi / 2)), color='k', zorder=10)
    ax.plot(laea_x(-np.pi, np.linspace(-np.pi / 2, np.pi / 2)), laea_y(-np.pi, np.linspace(-np.pi / 2, np.pi / 2)),
            color='k', zorder=10)
    ax.plot(laea_x(np.pi / 3, np.linspace(-np.pi / 2, np.pi / 2)), laea_y(np.pi / 3, np.linspace(-np.pi / 2, np.pi / 2)),
           color='k', zorder=10)
    ax.plot(laea_x(-np.pi / 3, np.linspace(-np.pi / 2, np.pi / 2)),
            laea_y(-np.pi / 3, np.linspace(-np.pi / 2, np.pi / 2)),
            color='k', zorder=10)
    ax.plot(laea_x(2*np.pi / 3, np.linspace(-np.pi / 2, np.pi / 2)),
            laea_y(2*np.pi / 3, np.linspace(-np.pi / 2, np.pi / 2)),
            color='k', zorder=10)
    ax.plot(laea_x(-2*np.pi / 3, np.linspace(-np.pi / 2, np.pi / 2)),
            laea_y(-2*np.pi / 3, np.linspace(-np.pi / 2, np.pi / 2)),
            color='k', zorder=10)
    ax.plot(laea_x(0, np.linspace(-np.pi / 2, np.pi / 2)),
            laea_y(0, np.linspace(-np.pi / 2, np.pi / 2)),
            color='k', zorder=10)
    ax.plot(laea_x(np.linspace(-np.pi, np.pi), 0),
            laea_y(np.linspace(-np.pi, np.pi), 0),
            color='k', zorder=10)
    ax.plot(laea_x(np.linspace(-np.pi, np.pi), np.pi / 6),
            laea_y(np.linspace(-np.pi, np.pi), np.pi / 6),
            color='k', zorder=10)
    ax.plot(laea_x(np.linspace(-np.pi, np.pi), -np.pi / 6),
            laea_y(np.linspace(-np.pi, np.pi), -np.pi / 6),
            color='k', zorder=10)
    ax.plot(laea_x(np.linspace(-np.pi, np.pi), np.pi / 3),
            laea_y(np.linspace(-np.pi, np.pi), np.pi / 3),
            color='k', zorder=10)
    ax.plot(laea_x(np.linspace(-np.pi, np.pi), -np.pi / 3),
            laea_y(np.linspace(-np.pi, np.pi), -np.pi / 3),
            color='k', zorder=10)
    ax.text(0, 1.47, r'$|0\rangle$', fontsize=20)
    ax.text(0, -1.53, r'$|1\rangle$', fontsize=20)
    ax.text(0.05, 0.05, r'$|+\rangle$', fontsize=20)
    ax.text(2.9, 0, r'$|-\rangle$', fontsize=20)
    ax.text(-3.2, 0, r'$|-\rangle$', fontsize=20)
    #ax.plot(laea_x(-np.pi, np.linspace(0, -np.pi)), laea_y(-np.pi, np.linspace(0, -np.pi)), color='k', zorder=10)
    #ax.plot(laea_x(np.pi/3, np.linspace(0, np.pi)), laea_y(np.pi/3, np.linspace(0, np.pi)), color='k', zorder=10)
    #ax.plot(laea_x(-np.pi/3, np.linspace(0, -np.pi)), laea_y(-np.pi/3, np.linspace(0, -np.pi)), color='k', zorder=10)
    #ax.plot(laea_x(2*np.pi / 3, np.linspace(0, np.pi)), laea_y(2*np.pi / 3, np.linspace(0, np.pi)), color='k', zorder=10)
    #ax.plot(laea_x(-2*np.pi / 3, np.linspace(0, -np.pi)), laea_y(-2*np.pi / 3, np.linspace(0, -np.pi)), color='k',
            #zorder=10)

    return fig, ax


def laea_x(lamb, phi):
    return 2 * np.sqrt(2) * np.cos(phi) * np.sin(lamb / 2) / np.sqrt(1 + np.cos(phi) * np.cos(lamb / 2))

def laea_y(lamb, phi):
    return np.sqrt(2) * np.sin(phi) / np.sqrt(1 + np.cos(phi) * np.cos(lamb / 2))

def _circle(points):
    labels = [0] * len(points)
    for i, p in enumerate(points):
        if np.linalg.norm(p) > np.sqrt(2 / np.pi):
            labels[i] = 1

    return points, labels


def _3circles(points):
    centers = np.array([[-1, 1], [1, 0], [-.5, -.5]])
    radii = np.array([1, np.sqrt(6 / np.pi - 1), 1 / 2])
    labels = [0] * len(points)
    for i, p in enumerate(points):
        for j, (c, r) in enumerate(zip(centers, radii)):
            if np.linalg.norm(p - c) < r:
                labels[i] = j + 1
                break

    return points, labels

def _square(points):
    labels = [0] * len(points)
    for i, p in enumerate(points):
        if np.max(np.abs(p)) > .5 * np.sqrt(2):
            labels[i] = 1

    return points, labels


def _4squares(points):
    labels = [0] * len(points)
    for i, p in enumerate(points):
        if p[0] < 0 and p[1] > 0: labels[i] = 1
        elif p[0] > 0 and p[1] < 0: labels[i] = 2
        elif p[0] > 0 and p[1] > 0: labels[i] = 3

    return points, labels

def _crown(points):
    c = [[0,0],[0,0]]
    r = [np.sqrt(.8), np.sqrt(.8 - 2/np.pi)]
    labels = [0]*points
    points = 1 - 2 * np.random.random((points, 2))
    for i, p in enumerate(points):
        if np.linalg.norm(p - c[0]) < r[0] and np.linalg.norm(p - c[1]) > r[1]:
            labels[i] = 1

    return points, labels

def _tricrown(points):
    c = [[0, 0], [0, 0]]
    r = [np.sqrt(.8), np.sqrt(.8 - 2 / np.pi)]
    labels = [0] * len(points)
    for i, p in enumerate(points):
        if np.linalg.norm(p - c[0]) > r[0] and np.linalg.norm(p - c[1]) > r[1]:
            labels[i] = 2
        elif np.linalg.norm(p - c[1]) > r[1]:
            labels[i] = 1

    return points, labels


def _wavy_lines(points):
    freq = 1
    def fun1(s):
        return s + np.sin(freq * np.pi * s)

    def fun2(s):
        return -s + np.sin(freq * np.pi * s)
    labels = [0] * len(points)
    for i, p in enumerate(points):
        if p[1] < fun1(p[0]) and p[1] > fun2(p[0]): labels[i] = 1
        elif p[1] > fun1(p[0]) and p[1] < fun2(p[0]): labels[i] = 2
        elif p[1] > fun1(p[0]) and p[1] > fun2(p[0]): labels[i] = 3

    return points, labels
