import numpy as np
import functions
from qibo import matrices, hamiltonians
import matplotlib.pyplot as plt
import collections


def times(qubits, clauses):
    """Count the times each qubit appears in a clause to normalize H0.
    Args:
        qubits (int): # of total qubits in the instance.
        clauses (list): clauses of the Exact Cover instance.

    Returns:
        times (list): number of times a qubit apears in all clauses.
    """
    times = np.zeros(qubits)
    for clause in clauses:
        for num in clause:
            times[num-1] += 1
    return times


def h(qubits, n, matrix):
    """Apply a matrix to a single qubit in the register.
    Args:
        qubits (int): # of total qubits in the instance.
        n (int): qubit position to apply matrix.
        matrix (np.array): 2x2 matrix to apply to a qubit.

    Returns:
        h (np.array): 2**qubits x 2**qubits hamiltonian.
    """
    a = np.eye(2 ** n, dtype=matrix.dtype)
    b = np.eye(2 ** (qubits - n - 1), dtype=matrix.dtype)
    return np.kron(np.kron(a, matrix), b)


def z(qubits, n):
    """Apply matrix [[0, 0], [0, 1]] to qubit n.
    Args:
        qubits (int): # of total qubits in the instance.
        n (int): qubit position to apply matrix.
        matrix (np.array): 2x2 matrix to apply to a qubit.

    Returns:
        h (np.array): 2**qubits x 2**qubits hamiltonian.
    """
    matrix = 0.5*(matrices.I-matrices.Z)
    return h(qubits, n, matrix)


def x(qubits, n):
    """Apply matrix [[0, 1], [1, 0]] to qubit n.
    Args:
        qubits (int): # of total qubits in the instance.
        n (int): qubit position to apply matrix.
        matrix (np.array): 2x2 matrix to apply to a qubit.

    Returns:
        h (np.array): 2**qubits x 2**qubits hamiltonian..
    """
    matrix = matrices.X
    return h(qubits, n, matrix)


def h_c(qubits, clause):
    """Hamiltonian with ground state that satisfies an Exact Cover clause.
    Args:
        qubits (int): # of total qubits in the instance.
        clause (list): Exact Cover clause.

    Returns:
        h_c (np.array): 2**qubits x 2**qubits hamiltonian that satisfies clause.
    """
    h_c = 0
    for i in clause:
        h_c += z(qubits, i-1)
    h_c -= np.eye(2**qubits)
    return h_c**2


def h_p(qubits, clauses):
    """Hamiltonian that satisfies all Exact Cover clauses.
    Args:
        qubits (int): # of total qubits in the instance.
        clauses (list): clauses for an Exact Cover instance.

    Return:
        h_p (np.array): 2**qubits x 2**qubits problem Hamiltonian.
    """
    h_p = 0
    for clause in clauses:
        h_p += h_c(qubits, clause)
    return h_p


def h0(qubits, times):
    """Initial hamiltonian for adiabatic evolution.
    Args:
        qubits (int): # of total qubits in the instance.
        times (list): number of times a qubit apears in all clauses.

    Return:
        h0 (np.array): 2**qubits x 2**qubits initial Hamiltonian.
    """
    h0 = 0
    for i in range(qubits):
        h0 += times[i]*0.5*(np.eye(2**qubits)-x(qubits, i))
    return h0


atol = 1e-15
instance = 1
for nqubits in [4, 8, 10, 12]:
    control, solution, clauses = functions.read_file(nqubits, instance)
    nqubits = int(control[0])
    # Define "easy" and "problem" Hamiltonians
    times = functions.times(nqubits, clauses)

    h0_target = h0(nqubits, times)
    h1_target = h_p(nqubits, clauses)

    sh0, smap0 = functions.h_initial(nqubits, times)
    sh1, smap1 = functions.h_problem(nqubits, clauses)

    H0_trotter = hamiltonians.TrotterHamiltonian.from_symbolic(sh0, smap0)
    H1_trotter = hamiltonians.TrotterHamiltonian.from_symbolic(sh1, smap1)
    cH0_trotter = H1_trotter.make_compatible(H0_trotter)

    print(f"Testing nqubits={nqubits}")
    np.testing.assert_allclose(H0_trotter.dense.matrix, h0_target, atol=atol)
    np.testing.assert_allclose(cH0_trotter.matrix, h0_target, atol=atol)
    np.testing.assert_allclose(H1_trotter.dense.matrix, h1_target, atol=atol)
