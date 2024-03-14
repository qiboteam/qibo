import matplotlib.pyplot as plt
import numpy as np
import sympy

from qibo import hamiltonians, matrices, symbols


def read_file(file_name, instance):
    """Collect data from .txt file that characterizes the problem instance.
    Args:
        file_name (str): name of the file that contains the instance information.
        instance (str): number of intance to use.

    Returns:
        control (list): important parameters of the instance.
            [number of qubits, number of clauses, number of ones in the solution]
        solution (list): list of the correct outputs of the instance for testing.
        clauses (list): list of all clauses, with the qubits each clause acts upon.
    """
    file = open(f"../data3sat/{file_name}bit/n{file_name}i{instance}.txt")
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
            times[num - 1] += 1
    return times


def h_problem(qubits, clauses):
    """Hamiltonian that satisfies all Exact Cover clauses.
    Args:
        qubits (int): # of total qubits in the instance.
        clauses (list): clauses for an Exact Cover instance.

    Returns:
        sham (sympy.Expr): Symbolic form of the problem Hamiltonian.
        smap (dict): Dictionary that maps the symbols that appear in the
            Hamiltonian to the corresponding matrices and target qubits.
    """
    z_matrix = (matrices.I - matrices.Z) / 2.0
    z = [symbols.Symbol(i, z_matrix) for i in range(qubits)]
    return sum((sum(z[i - 1] for i in clause) - 1) ** 2 for clause in clauses)


def h_initial(qubits, times):
    """Initial hamiltonian for adiabatic evolution.
    Args:
        qubits (int): # of total qubits in the instance.
        times (list): number of times a qubit apears in all clauses.

    Returns:
        sham (sympy.Expr): Symbolic form of the easy Hamiltonian.
        smap (dict): Dictionary that maps the symbols that appear in the
            Hamiltonian to the corresponding matrices and target qubits.
    """
    return sum(0.5 * times[i] * (1 - symbols.X(i)) for i in range(qubits))


def spolynomial(t, params):
    """General polynomial scheduling satisfying s(0)=0 and s(1)=1."""
    f = sum(p * t ** (i + 2) for i, p in enumerate(params))
    f += (1 - np.sum(params)) * t
    return f


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
    times = np.arange(0, T + dt, dt)
    ax.plot(times, ground, label="ground state", color="C0")
    ax.plot(times, first, label="first excited state", color="C1")
    plt.ylabel("energy")
    plt.xlabel("schedule")
    plt.title("Energy during adiabatic evolution")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{qubits}_qubits_energy.png", dpi=300, bbox_inches="tight")
    fig, ax = plt.subplots()
    ax.plot(times, gap, label="gap energy", color="C0")
    plt.ylabel("energy")
    plt.xlabel("schedule")
    plt.title("Energy during adiabatic evolution")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{qubits}_qubits_gap.png", dpi=300, bbox_inches="tight")
