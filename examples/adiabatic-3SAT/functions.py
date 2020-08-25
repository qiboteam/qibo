import numpy as np
from qibo import matrices, hamiltonians
import matplotlib.pyplot as plt
import collections


def read_file(file_name):
    """Collect data from .txt file that characterizes the problem instance.
    Args:
        file_name (str): name of the file that contains the instance information. Available
            in \data are examples for instances with 4, 8, 10, 12 and 16 qubits.
        
    Returns:
        control (list): important parameters of the instance. 
            [number of qubits, number of clauses, number of ones in the solution]
        solution (list): list of the correct outputs of the instance for testing.
        clauses (list): list of all clauses, with the qubits each clause acts upon.
    """
    file = open('data/{}'.format(file_name), 'r')
    control = list(map(int, file.readline().split()))
    solution = list(map(str, file.readline().split()))
    clauses = [list(map(int, file.readline().split())) for _ in range(control[1])]
    return control, solution, clauses


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


def h_ct():
    """Generate the 2 qubit Hamiltonian for Trotter evolution, equivalent to the clause Hamiltonian.
    Returns:
        h_ct (Hamiltonian): 4 x 4 Hamiltonian used as a base for the problem Hamiltonian.
    """
    m1 = 0.5 * (np.kron(matrices.Z, matrices.Z) - np.kron(matrices.Z, matrices.I))
    return hamiltonians.Hamiltonian(2, m1, numpy=True)
    
    
def h0t():
    """Generate the 2 qubit Hamiltonian for Trotter evolution, equivalent to the intial Hamiltonian.
    Returns:
        h0t (Hamiltonian): 4 x 4 Hamiltonian used as a base for the initial Hamiltonian.
    """
    m0 = 0.5 * np.kron(matrices.I - matrices.X, matrices.I)
    return hamiltonians.Hamiltonian(2, m0, numpy=True)


def trotter_dict(clauses):
    """Create the dicitonaries needed to create the Hamiltonians for an Exact Cover problem. The dictionaries
        must only contain commuting terms.
    Args:
        clauses (list): clauses for an Exact Cover instance.
        
    Returns:
        parts0 (tuple(dict)): corresponding pair of qubits and initial Hamiltonian applied to them.
        parts1 (tuple(dict)): corresponding pair of qubits and clause Hamiltonian applied to them.
    """
    # Count the number of times a pair (i, j) appears in the clauses in order to properly normalize
    pairs = collections.Counter()
    for q0, q1, q2 in clauses:
        pairs[(q0 - 1, q1 - 1)] += 1
        pairs[(q1 - 1, q2 - 1)] += 1
        pairs[(q2 - 1, q0 - 1)] += 1 
    # Separate dictionaries to commuting parts
    multi_pairs, multi_sets = [], []
    for pair, n in pairs.items():
        pair_set = set(pair)
        not_added = True
        for p, s in zip(multi_pairs, multi_sets):
            if not pair_set & s:
                p[pair] = n
                s |= pair_set
                not_added = False
                break
        if not_added:
            multi_pairs.append({pair: n})
            multi_sets.append(set(pair))
    # Define TrotterHamiltonian with proper normalization
    h0 = h0t()
    h1 = h_ct()
    parts0 = ({pair: h0 if n == 1 else n * h0 for pair, n in p.items()}
              for p in multi_pairs)
    parts1 = ({pair: h1 if n == 1 else n * h1 for pair, n in p.items()}
              for p in multi_pairs)
    return parts0, parts1


def plot(qubits, ground, first, gap, dt, T):
    """Get the first two eigenvalues and the gap energy
    Args:
        qubits (int): # of total qubits in the instance.
        ground (list): ground state energy during the evolution.
        first (list): first excited state during the evolution.
        gap (list): gap energy during the evolution.
        T (float): Final time for the schedue.
        dt (float): time interval for the evolution.
        
    Returns:
        {}_qubits_energy.png: energy evolution of the ground and first excited state.
        {}_qubits_gap_energy.png: gap evolution during the adiabatic process.
    """
    fig, ax = plt.subplots()
    times = np.arange(0, T+dt, dt)
    ax.plot(times, ground, label='ground state', color='C0')
    ax.plot(times, first, label='first excited state', color='C1')
    plt.ylabel('energy')
    plt.xlabel('schedule')
    plt.title('Energy during adiabatic evolution')
    ax.legend()
    fig.tight_layout()
    fig.savefig('{}_qubits_energy.png'.format(qubtis), dpi=300, bbox_inches='tight')
    fig, ax = plt.subplots()
    ax.plot(times, gap, label='gap energy', color='C0')
    plt.ylabel('energy')
    plt.xlabel('schedule')
    plt.title('Energy during adiabatic evolution')
    ax.legend()
    fig.tight_layout()
    fig.savefig('{}_qubits_gap.png'.format(qubits), dpi=300, bbox_inches='tight')
    