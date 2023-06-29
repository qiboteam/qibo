import math
import numpy as np
from sklearn.metrics import roc_curve, auc, confusion_matrix
import h5py
import matplotlib.pyplot as plt
import qkmedians as qkmed


def euclidean_dist(a, b):
    return np.linalg.norm(a - b)


def calc_norm(a, b):
    return math.sqrt(np.sum(a**2) + np.sum(b**2))


def combine_loss_min(loss):
    """Returns minimum loss for 2 jet data."""
    loss_j1, loss_j2 = np.split(loss, 2)
    return np.minimum(loss_j1, loss_j2)

def load_clustering_test_data(data_qcd_file, data_signal_file, test_size=10000, k=2, read_dir=None):
    
    if not read_dir:
        raise ValueError('Need to specify directory for datasets.')

    # read QCD latent space data
    with h5py.File(f'{read_dir}/{data_qcd_file}', 'r') as file:
        data = file['latent_space']
        l1 = data[:,0,:]
        l2 = data[:,1,:]
        
        data_test_qcd = np.vstack([l1[:test_size], l2[:test_size]])
    
    # read SIGNAL predicted data
    with h5py.File(f'{read_dir}/{data_signal_file}', 'r') as file:
        data = file['latent_space']
        l1 = data[:,0,:]
        l2 = data[:,1,:]
        
        data_test_sig = np.vstack([l1[:test_size], l2[:test_size]])
    
    return data_test_qcd, data_test_sig

def AD_score(cluster_assignments, distances, method='sum_all'):
    if method=='sum_all':
        return np.sqrt(np.sum(distances**2, axis=1))
    else:
        return np.sqrt(distances[range(len(distances)), cluster_assignments]**2)

def AD_scores(test_qcd, test_sig, centroids):
        
    # find cluster assignments + distance to centroids for test data
    cluster_assign, distances = qkmed.find_nearest_neighbour(test_qcd, centroids)
    #plot_latent_representations(test_qcd, q_cluster_assign)
    cluster_assign_s, distances_s = qkmed.find_nearest_neighbour(test_sig, centroids)

    # calc AD scores
    score_qcd = AD_score(cluster_assign, distances)
    score_sig = AD_score(cluster_assign_s, distances_s)

    # calculate loss from 2 jets
    loss_qcd = combine_loss_min(score_qcd)
    loss_sig = combine_loss_min(score_sig)
    
    return [loss_qcd, loss_sig]

def calc_AD_scores(centroids_file, data_qcd_file, data_signal_file, k=2, test_size=10000, results_dir=None, data_dir=None):
    
    if not results_dir:
        raise ValueError('Need to specify directory for loading the centroids.')
    
    # load centroids
    centroids = np.load(f'{results_dir}/{centroids_file}')

    test_qcd, test_sig = load_clustering_test_data(data_qcd_file, data_signal_file, test_size=test_size, k=k, read_dir=data_dir)

    loss = AD_scores(test_qcd, test_sig, centroids)
    
    return loss


def get_roc_data(qcd, bsm):
    """Calculates FPR, TPR and AUC"""
    true_val = np.concatenate((np.ones(bsm.shape[0]), np.zeros(qcd.shape[0])))
    pred_val = np.nan_to_num(np.concatenate((bsm, qcd)))
    fpr_loss, tpr_loss, threshold_loss = roc_curve(true_val, pred_val)
    auc_data = auc(fpr_loss, tpr_loss)
    return fpr_loss, tpr_loss, auc_data

def plot_ROCs_compare(loss, title, xlabel='TPR', ylabel='1/FPR', legend_loc='best', save_dir=None):
    
    fig = plt.figure(figsize=(8,8))
    
    loss_qcd, loss_sig = loss

    # roc data
    data = get_roc_data(loss_qcd, loss_sig)
    tpr = data[1]; fpr = data[0]


    plt.plot(tpr, 1./fpr, label='(auc = %.2f)'% (data[2]*100.), linewidth=1.5)
        
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.yscale('log')
    plt.title(title)
    plt.legend(fancybox=True, frameon=True, prop={"size":10}, bbox_to_anchor =(1.0, 1.0))
    plt.grid(True)

    if save_dir:
        plt.savefig(f'{save_dir}/roc_curve.pdf', dpi = fig.dpi, bbox_inches='tight')
    else:
        plt.show()
