from typing import Any, Dict, Tuple

import numpy as np
from qubo_utils import binary2spin, spin2QiboHamiltonian


def qubo_qap(g: Tuple[np.ndarray, np.ndarray], penalty: float = None):
    """Given adjacency matrix of flow and distance, formulate the QUBO of
    Quadratic Assignment Problem (QAP)

    Args:
        g: the adjacency matrices of flow and distance that describe the QAP
        penalty: penalty weight for the constraints, if not given, it
            is inferred from the adjacency matrices

    Returns:
        linear (Dict[Any, float]): linear terms
        quadratic (Dict[(Any, Any), float]): quadratic terms
        offset (float): the constant term
    """

    flow, distance = g
    n = len(flow)
    q = np.einsum("ij,kl->ikjl", flow, distance)

    if penalty is None:
        penalty = qubo_qap_penalty(g)

    i = range(len(q))
    q[i, :, i, :] += 1 * penalty
    q[i, :, i, :] -= 2 * penalty * np.eye(n)
    q[:, i, :, i] += 1 * penalty
    q[:, i, :, i] -= 2 * penalty * np.eye(n)

    q = q.reshape(n**2, n**2).astype(np.float32)

    offset = penalty * 2 * n
    linear = {i: q[i, i] for i in range(q.shape[0])}
    quadratic = {
        (i, j): q[i, j] for j in range(q.shape[1]) for i in range(q.shape[0]) if i > j
    }

    return linear, quadratic, offset


def qubo_qap_penalty(g: Tuple[np.ndarray, np.ndarray]):
    """Find appropriate penalty weight to ensure feasibility

    Args:
        g: the adjacency matrices of flow and distance that describe the QAP

    Returns:
        penalty (float): penalty weight for the constraints
    """
    F, D = g
    q = np.einsum("ij,kl->ikjl", F, D)
    return F.shape[0] * np.abs(q).max()


def qubo_qap_feasibility(g: Tuple[np.ndarray, np.ndarray], solution: Dict[Any, int]):
    """Check if the solution violates the constraints of the problem

    Args:
        g: the adjacency matrices of flow and distance that describe the QAP
        solution (Dict[Any, int]): the binary solution

    Returns:
        feasibility (bool): whether the solution meet the constraints
    """
    F, D = g

    solution = [[k, v] for k, v in solution.items()]
    solution.sort(key=lambda x: x[0])
    _, vals = zip(*solution)
    vals = np.asarray(vals).reshape(F.shape)

    assert np.all(
        np.logical_or(vals == 0, vals == 1)
    ), "Decision variable must be 0 or 1"
    return np.all(vals.sum(axis=0) == 1) and np.all(vals.sum(axis=1) == 1)


def qubo_qap_energy(g: Tuple[np.ndarray, np.ndarray], solution: Dict[Any, int]):
    """Calculate the energy of the solution on a QAP problem
    The result is based on the assumption that soltuion is feasible.

    Args:
        g: the adjacency matrices of flow and distance that describe the QAP
        solution (Dict[Any, int]): the solution

    Returns:
        energy (float): the energy of the solution
    """

    F, D = g

    solution = [[k, v] for k, v in solution.items()]
    solution.sort(key=lambda x: x[0])
    _, vals = zip(*solution)
    vals = np.asarray(vals).reshape(F.shape)

    state = np.vstack(np.where(vals == 1)).T
    state = np.array(state).astype(np.int32)
    energy = 0
    for i in range(len(state)):
        energy += np.einsum(
            "i,i->", F[state[i, 0], state[:, 0]], D[state[i, 1], state[:, 1]]
        )
    return energy


def hamiltonian_qap(
    g: Tuple[np.ndarray, np.ndarray], penalty: float = None, dense: bool = False
):
    """Given a flow and distance matrices, return the hamiltonian of Quadratic
    Assignment Problem (QAP).

    If penalty is not given, it is inferred from g.

    Args:
        g (Tuple[np.ndarray, np.ndarray]): flow and distance matrices
        penalty (float): penalty weight of the constraints
        dense (bool): sparse or dense hamiltonian

    Returns:
        ham: qibo hamiltonian

    """

    linear, quadratic, offset = qubo_qap(g, penalty)

    h, J, _ = binary2spin(linear, quadratic)
    h = {k: -v for k, v in h.items()}
    J = {k: -v for k, v in J.items()}

    ham = spin2QiboHamiltonian(h, J, dense=dense)

    return ham
