import numpy as np

from qibo import Circuit, gates


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


def create_qc(qubits, clause_num):
    """Create the quantum circuit necessary to solve the problem.
    Args:
        qubits (int): qubits needed to encode the problem.
        clause_num (int): number of clauses of the problem.

    Returns:
        q (list): quantum register that encodes the problem.
        c (list): quantum register that that records the satisfies clauses.
        ancilla (int): Grover ancillary qubit.
        circuit (Circuit): quantum circuit where the gates will be allocated.
    """
    q = [i for i in range(qubits)]
    ancilla = qubits
    c = [i + qubits + 1 for i in range(clause_num)]
    circuit = Circuit(qubits + clause_num + 1)
    return q, c, ancilla, circuit


def start_grover(q, ancilla):
    """Generator that performs the starting step in Grover's search algorithm.
    Args:
        q (list): quantum register that encodes the problem.
        ancilla (int): Grover ancillary qubit.

    Returns:
        quantum gate generator for the first step of Grover.
    """
    yield gates.X(ancilla)
    yield gates.H(ancilla)
    for i in q:
        yield gates.H(i)


def oracle(q, c, ancilla, clauses):
    """Generator that acts as the oracle for a 3SAT problem. Changes the sign of the amplitude of the quantum
    states that encode the solution.
    Args:
        q (list): quantum register that encodes the problem.
        c (list): quantum register that that records the satisfies clauses.
        ancilla (int): Grover ancillary qubit. Used to change the sign of the
            correct amplitudes.
        clauses (list): list of all clauses, with the qubits each clause acts upon.

    Returns:
        quantum gate generator for the 3SAT oracle
    """
    k = 0
    for clause in clauses:
        yield gates.CNOT(q[clause[0] - 1], c[k])
        yield gates.CNOT(q[clause[1] - 1], c[k])
        yield gates.CNOT(q[clause[2] - 1], c[k])
        yield gates.X(c[k]).controlled_by(
            q[clause[0] - 1], q[clause[1] - 1], q[clause[2] - 1]
        )
        k += 1
    yield gates.X(ancilla).controlled_by(*c)
    k = 0
    for clause in clauses:
        yield gates.CNOT(q[clause[0] - 1], c[k])
        yield gates.CNOT(q[clause[1] - 1], c[k])
        yield gates.CNOT(q[clause[2] - 1], c[k])
        yield gates.X(c[k]).controlled_by(
            q[clause[0] - 1], q[clause[1] - 1], q[clause[2] - 1]
        )
        k += 1


def diffusion(q):
    """Generator that performs the inversion over the average step in Grover's search algorithm.
    Args:
        q (list): quantum register that encodes the problem.

    Returns:
        quantum gate geenrator that applies the diffusion step.
    """
    for i in q:
        yield gates.H(i)
        yield gates.X(i)
    yield gates.H(q[0])
    yield gates.X(q[0]).controlled_by(*q[1 : len(q)])
    yield gates.H(q[0])
    for i in q:
        yield gates.X(i)
        yield gates.H(i)


def grover(circuit, q, c, ancilla, clauses, steps):
    """Generator that performs the inversion over the average step in Grover's search algorithm.
    Args:
        circuit (Circuit): empty quantum circuit onto which the grover algorithm is performed
        q (list): quantum register that encodes the problem.
        c (list): quantum register that that records the satisfies clauses.
        ancilla (int): Grover ancillary qubit.
        clauses (list): list of all clauses, with the qubits each clause acts upon.
        steps (int): number of times the oracle+diffuser operators have to be applied
            in order to find the solution. Grover's search algorihtm dictates O(sqrt(2**qubits)).

    Returns:
        circuit (Circuit): circuit with the full grover algorithm applied.
    """
    circuit.add(start_grover(q, ancilla))
    for i in range(steps):
        circuit.add(oracle(q, c, ancilla, clauses))
        circuit.add(diffusion(q))
    circuit.add(gates.M(*(q), register_name="result"))
    return circuit
