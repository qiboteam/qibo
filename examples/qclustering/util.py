import os

import h5py
import matplotlib.pyplot as plt
import numpy as np
import qkmedians as qkmed
from sklearn.metrics import auc, roc_curve


def combine_loss_min(loss):
    """Find minimum loss from loss arrays obtained from dijet events.

    Parameters
    ----------
    loss : :class:`numpy.ndarray`
        Loss array of shape (num_of_samples*2, ), where num_of_samples = train_size or test_size.

    Returns
    -------
    :class:`numpy.ndarray`
        Final loss array of shape (num_of_samples, ).

    """
    loss_j1, loss_j2 = np.split(loss, 2)
    return np.minimum(loss_j1, loss_j2)


def load_clustering_test_data(
    data_qcd_file, data_signal_file, test_size=10000, k=2, read_dir=None
):
    """Load test dataset.

    Parameters
    ----------
    data_qcd_file : str
        Name of the file for test QCD dataset.
    data_signal_file : str
        Name of the file for test signal dataset.
    test_size : int
        Number of test samples.
    k : int
        Number of classes in quantum k-medians.
    read_dir : str
        Path to file with test datasets.

    Returns
    -------
    (:class:`numpy.ndarray`, :class:`numpy.ndarray`)
        (Test data for QCD events, Test data for signal events).
    """

    if not read_dir:
        raise ValueError("Need to specify directory for datasets.")

    # read QCD latent space data
    with h5py.File(f"{read_dir}/{data_qcd_file}", "r") as file:
        data = file["latent_space"]
        l1 = data[:, 0, :]
        l2 = data[:, 1, :]

        data_test_qcd = np.vstack([l1[:test_size], l2[:test_size]])

    # read SIGNAL predicted data
    with h5py.File(f"{read_dir}/{data_signal_file}", "r") as file:
        data = file["latent_space"]
        l1 = data[:, 0, :]
        l2 = data[:, 1, :]

        data_test_sig = np.vstack([l1[:test_size], l2[:test_size]])

    return data_test_qcd, data_test_sig


def AD_score(cluster_assignments, distances, method="sum_all"):
    """Calculate anomaly detection score.

    Parameters
    ----------
    cluster_assignments : :class:`numpy.ndarray`
        Cluster assignments for each point in space.
    distances : :class:`numpy.ndarray`
        Distances to cluster centroids. Shape = (N_events, k) where k = number of clusters.
    method : str
        Method how to calculate anomaly detection score. `'Sum_all'` = sum all distances to different clusters.

    Returns
    -------
    :class:`numpy.ndarray`
        Anomaly score for each event.
    """
    if method == "sum_all":
        return np.sqrt(np.sum(distances**2, axis=1))
    else:
        return np.sqrt(distances[range(len(distances)), cluster_assignments] ** 2)


def AD_scores(test_qcd, test_sig, centroids):
    """Helper function to calculate anomaly scores for QCD and signal events.

    Parameters
    ----------
    test_qcd : :class:`numpy.ndarray`
        Test QCD dataset.
    test_sig : :class:`numpy.ndarray`
        Test signal dataset.
    centroids : :class:`numpy.ndarray`
        Coordinates for class centroids.

    Returns
    -------
    (:class:`numpy.ndarray`, :class:`numpy.ndarray`)
        (Anomaly score for QCD data, Anomaly score for signal data).
    """

    # find cluster assignments + distance to centroids for test data
    cluster_assign, distances = qkmed.find_nearest_neighbour(test_qcd, centroids)
    cluster_assign_s, distances_s = qkmed.find_nearest_neighbour(test_sig, centroids)

    # calc AD scores
    score_qcd = AD_score(cluster_assign, distances)
    score_sig = AD_score(cluster_assign_s, distances_s)

    # calculate loss from 2 jets
    loss_qcd = combine_loss_min(score_qcd)
    loss_sig = combine_loss_min(score_sig)

    return [loss_qcd, loss_sig]


def calc_AD_scores(
    centroids_file,
    data_qcd_file,
    data_signal_file,
    k=2,
    test_size=10000,
    results_dir=None,
    data_dir=None,
):
    """Calculate anomaly scores for QCD and signal events.

    Parameters
    ----------
    data_qcd_file : str
        Name of the file for test QCD dataset.
    data_signal_file : str
        Name of the file for test signal dataset.
    k : int
        Number of classes in quantum k-medians.
    test_size : int
        Number of test samples.
    results_dir : str
        Path to file with saved centroids.
    data_dir : str
        Path to file with test datasets.

    Returns
    -------
    (:class:`numpy.ndarray`, :class:`numpy.ndarray`)
        loss = (Anomaly score for QCD data, Anomaly score for signal data).
    """

    if not results_dir:
        raise ValueError("Need to specify directory for loading the centroids.")

    # load centroids
    centroids = np.load(f"{results_dir}/{centroids_file}")

    test_qcd, test_sig = load_clustering_test_data(
        data_qcd_file, data_signal_file, test_size=test_size, k=k, read_dir=data_dir
    )

    loss = AD_scores(test_qcd, test_sig, centroids)

    return loss


def get_roc_data(qcd, signal):
    """Calculate ROC curve data - tpr, fpr, auc

    Parameters
    ----------
    qcd : :class:`numpy.ndarray`
        Anomaly score for QCD dataset.
    signal : :class:`numpy.ndarray`
        Anomaly score for signal dataset.

    Returns
    -------
    (:class:`numpy.ndarray`, :class:`numpy.ndarray`, :class:`numpy.ndarray`)
        False positive rate (fpr), True positive rate (tpr), Area under the ROC curve (auc)
    """

    true_val = np.concatenate((np.ones(signal.shape[0]), np.zeros(qcd.shape[0])))
    pred_val = np.nan_to_num(np.concatenate((signal, qcd)))
    fpr, tpr, _ = roc_curve(true_val, pred_val)
    auc_data = auc(fpr, tpr)
    return fpr, tpr, auc_data


def plot_ROCs_compare(loss, title, xlabel="TPR", ylabel="1/FPR", save_dir=None):
    """Plot ROC curve.

    Parameters
    ----------
    loss : (:class:`numpy.ndarray`, :class:`numpy.ndarray`)
        (Anomaly score for QCD data, Anomaly score for signal data).
    title : str
        Title of ROC curve plot.
    xlabel : str
        Name of x-axis in ROC plot.
    ylabel : str
        Name of y-axis in ROC plot.
    save_dir : str
        Path to directory for saving ROC plot.
    """

    fig = plt.figure(figsize=(8, 8))

    loss_qcd, loss_sig = loss

    # roc data
    data = get_roc_data(loss_qcd, loss_sig)
    tpr = data[1]
    fpr = data[0]

    plt.plot(tpr, 1.0 / fpr, label="(auc = %.2f)" % (data[2] * 100.0), linewidth=1.5)

    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.yscale("log")
    plt.title(title)
    plt.legend(
        fancybox=True, frameon=True, prop={"size": 10}, bbox_to_anchor=(1.0, 1.0)
    )
    plt.grid(True)

    if save_dir:
        if os.path.exists(save_dir) is False:
            os.system(f"mkdir {save_dir}")
        plt.savefig(f"{save_dir}/roc_curve.pdf", dpi=fig.dpi, bbox_inches="tight")
    else:
        plt.show()
