import numpy as np
from itertools import product


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

def _circle(points):
    labels = [0] * len(points)
    for i, p in enumerate(points):
        if np.linalg.norm(p) > np.sqrt(2 / np.pi):
            labels[i] = 1

    return points, labels


def _3circles(points):
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
        elif p[0] > 0 and p[1] > 0: labels[i] = 2

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
