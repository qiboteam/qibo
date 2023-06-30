import distance_calc as distc
import minimization as m
import numpy as np


def initialize_centroids(points, k):
    """Randomly initialize centroids of data points.

    Parameters
    ----------
    points : :class:`numpy.ndarray`
        Points represented as an array of shape ``(N, X)``, where `N` = number of samples, `X` = dimension of latent space.
    k : int
        Number of clusters.

    Returns
    -------
    `numpy.ndarray`
        `k` number of centroids.
    """
    indexes = np.random.randint(points.shape[0], size=k)
    return points[indexes]


def find_centroids(points, cluster_labels, clusters=2):
    """Find cluster centroids .

    Parameters
    ----------
    points : :class:`numpy.ndarray`
        Points represented as an array of shape ``(N, X)``, where `N` = number of samples, `X` = dimension of latent space.
    cluster_labels : :class:`numpy.ndarray`
        Cluster labels assigned to each data point - shape `(N,)`.
    clusters : int
        Number of clusters.

    Returns
    -------
    :class:`numpy.ndarray`
        Centroids
    """

    centroids = np.zeros([clusters, points.shape[1]])
    k = points.shape[1]
    for j in range(clusters):
        points_class_i = points[cluster_labels == j]
        median = np.median(points_class_i, axis=0)
        centroids[j, :] = median
    return np.array(centroids)


def find_nearest_neighbour(points, centroids, mintype="classic", nshots=10000):
    """Find cluster assignments for points.

    Parameters
    -----------
    points : :class:`numpy.ndarray`
        Points represented as an array of shape ``(N, X)``, where `N` = number of samples, `X` = dimension of latent space.
    centroids : :class:`numpy.ndarray`
        Centroids of shape ``(k, X)``.
    mintype : str
        Minimization type for cluster assignment.
    nshots : int
        Number of shots for executing a quantum circuit - to get frequencies.

    Returns
    -------
    :class:`numpy.ndarray`
        Cluster labels : array of shape `(N,)` specifying to which cluster each point is assigned.
    :class:`numpy.ndarray`
        Distances: array of shape `(N,)` specifying distances to nearest cluster for each point.
    """

    n = points.shape[0]
    num_features = points.shape[1]
    k = centroids.shape[0]  # number of centroids
    cluster_label = []
    distances = []

    for i in range(n):  # through all training samples
        dist = []
        for j in range(k):  # distance of each training example to each centroid
            temp_dist, _ = distc.DistCalc(
                points[i, :], centroids[j, :], nshots=nshots
            )  # returning back one number for all latent dimensions!
            dist.append(temp_dist)
        # assign cluster
        if mintype == "classic":
            cluster_index = np.argmin(dist)  # classical minimization
        else:
            cluster_index = m.duerr_hoyer_algo(dist)  # quantum minimization
        cluster_label.append(cluster_index)
        distances.append(dist)
    return np.asarray(cluster_label), np.asarray(distances)
