import numpy as np
from qibo import matrices
import matplotlib.pyplot as plt


def read_file(file_name):
    """Collect data from .txt file that characterizes the problem instance
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
    """Count the times each qubit appears in a clause to normalize H0
    Args:
        qubits (int): # of total qubits in the instance
        clauses (list): clauses of the Exact Cover instance
        
    Returns:
        times (list): number of times a qubit apears in all clauses
    """
    times = np.zeros(qubits)
    for clause in clauses:
        for num in clause:
            times[num-1] += 1
    return times


def h(qubits, n, matrix):
    """Apply a matrix to a single qubit in the register
    Args:
        qubits (int): # of total qubits in the instance
        n (int): qubit position to apply matrix
        matrix (np.array): 2x2 matrix to apply to a qubit
        
    Returns:
        h (np.array): 2**qubits x 2**qubits hamiltonian
    """
    a = np.eye(2 ** n, dtype=matrix.dtype)
    b = np.eye(2 ** (qubits - n - 1), dtype=matrix.dtype)
    return np.kron(np.kron(a, matrix), b)


def z(qubits, n):
    """Apply matrix [[0, 0], [0, 1]] to qubit n
    Args:
        qubits (int): # of total qubits in the instance
        n (int): qubit position to apply matrix
        matrix (np.array): 2x2 matrix to apply to a qubit
        
    Returns:
        h (np.array): 2**qubits x 2**qubits hamiltonian
    """
    matrix = 0.5*(matrices.I-matrices.Z)
    return h(qubits, n, matrix)
    
    
def x(qubits, n):
    """Apply matrix [[0, 1], [1, 0]] to qubit n
    Args:
        qubits (int): # of total qubits in the instance
        n (int): qubit position to apply matrix
        matrix (np.array): 2x2 matrix to apply to a qubit
        
    Returns:
        h (np.array): 2**qubits x 2**qubits hamiltonian
    """
    matrix = matrices.X
    return h(qubits, n, matrix)
    
    
def h_c(qubits, clause):
    """Hamiltonian with ground state that satisfies an Exact Cover clause
    Args:
        qubits (int): # of total qubits in the instance
        clause (list): Exact Cover clause
        
    Returns:
        h_c (np.array): 2**qubits x 2**qubits hamiltonian that satisfies clause
    """
    h_c = 0
    for i in clause:
        h_c += z(qubits, i-1)
    h_c -= np.eye(2**qubits)
    return h_c**2
    
    
def h_p(qubits, clauses):
    """Hamiltonian that satisfies all Exact Cover clauses
    Args:
        qubits (int): # of total qubits in the instance
        clauses (list): clauses for an Exact Cover instance
        
    Return:
        h_p (np.array): 2**qubits x 2**qubits problem Hamiltonian
    """
    h_p = 0
    for clause in clauses:
        h_p += h_c(qubits, clause)
    return h_p
    
    
def h0(qubits, times):
    """Initial hamiltonian for adiabatic evolution
    Args:
        qubits (int): # of total qubits in the instance
        times (list): number of times a qubit apears in all clauses
        
    Return:
        h0 (np.array): 2**qubits x 2**qubits initial Hamiltonian
    """
    h0 = 0
    for i in range(qubits):
        h0 += times[i]*0.5*(np.eye(2**qubits)-x(qubits, i))
    return h0


def extract_gap(evolve, T, dt):
    """Get the first two eigenvalues and the gap energy
    Args:
        evolve (AdiabaticEvolution): completed adiabatic evolution.
        T (float): Final time for the schedue.
        dt (float): time interval for the evolution.
        
    Returns:
        ground (list): ground state energy during the evolution.
        first (list): first excited state during the evolution.
        gap (list): gap energy during the evolution.
    """
    ground = []
    first = []
    gap = []
    for i in np.arange(0, T, dt):
        eig = evolve.solver.hamiltonian(i).eigenvalues()[:2]
        ground.append(np.real(eig[0]))
        first.append(np.real(eig[1]))
        gap.append(np.real(eig[1])-np.real(eig[0]))
    return ground, first, gap


def plot(ground, first, gap, dt, T):
    """Get the first two eigenvalues and the gap energy
    Args:
        ground (list): ground state energy during the evolution.
        first (list): first excited state during the evolution.
        gap (list): gap energy during the evolution.
        T (float): Final time for the schedue.
        dt (float): time interval for the evolution.
        
    Returns:
        energy.png: energy evolution of the ground and first excited state.
        gap_energy.png: gap evolution during the adiabatic process.
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
    fig.savefig('energy.png', dpi=300, bbox_inches='tight')
    fig, ax = plt.subplots()
    ax.plot(times, gap, label='gap energy', color='C0')
    plt.ylabel('energy')
    plt.xlabel('schedule')
    plt.title('Energy during adiabatic evolution')
    ax.legend()
    fig.tight_layout()
    fig.savefig('gap_energy.png', dpi=300, bbox_inches='tight')
    