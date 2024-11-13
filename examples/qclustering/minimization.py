import math

import numpy as np
from grover import grover_qc
from oracle import create_oracle_circ

import qibo
from qibo import gates
from qibo.models import Circuit


def duerr_hoyer_algo(distances):
    """Perform Duerr-Hoyer algorithm.

    Parameters
    ----------
    distance : :class:`numpy.ndarray`
        Array of distances a point to each cluster.
        Shape=(1, k) where k is number of cluster centers.

    Returns
    -------
    int
        New cluster assigned for that point.
    """
    qibo.set_backend(backend="qiboml", platform="tensorflow")
    tf = qibo.get_backend().tf

    k = len(distances)
    n = int(math.floor(math.log2(k)) + 1)

    # choose random threshold
    index_rand = np.random.choice(k)
    threshold = distances[index_rand]
    max_iters = int(math.ceil(np.sqrt(2**n)))

    for _ in range(max_iters):
        qc = Circuit(n)

        for i in range(n):
            qc.add(gates.H(i))

        # create oracle
        qc_oracle, n_indices_marked = create_oracle_circ(distances, threshold, n)

        # grover circuit
        qc = grover_qc(qc, n, qc_oracle, n_indices_marked)

        counts = qc.execute(nshots=1000).frequencies(binary=True)

        # measure highest probability
        probs = counts.items()
        sorted_probs = dict(sorted(probs, key=lambda item: item[1], reverse=True))
        sorted_probs_keys = list(sorted_probs.keys())
        new_ix = [
            int(sorted_probs_keys[i], 2)
            for i, _ in enumerate(sorted_probs_keys)
            if int(sorted_probs_keys[i], 2) < k
        ]
        new_ix = new_ix[0]
        threshold = distances[new_ix]
    return new_ix
