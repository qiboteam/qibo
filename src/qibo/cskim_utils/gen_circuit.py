import random

import numpy as np
from qibo.models import Circuit
from qibo import gates
from qibo.transpiler.optimizer import Preprocessing
from qibo.transpiler.pipeline import Passes
from qibo.transpiler.unroller import NativeGates, Unroller
from qiskit import QuantumCircuit


def random_cx_circuit_qiskit(n_qubits, n_cxs):
    """
    Generates a random control Qiskit circuit with n_qubits qubits and n_cxs control gates.
    """
    qc = None # QuantumCircuit(n_qubits)

    for _ in range(n_cxs):
        qubit1 = random.randint(0, n_qubits - 1)
        qubit2 = random.randint(0, n_qubits - 1)

        while (qubit1 == qubit2):
            qubit1 = random.randint(0, n_qubits - 1)
            qubit2 = random.randint(0, n_qubits - 1)

        qc.cx(qubit1, qubit2)

    return qc

def random_control_circuit_qibo(n_qubits, n_cs, cx=False):
    """
    Generates a random control Qibo circuit with n_qubits qubits and n_cs control gates.
    If cx is True, the control gates are CX, otherwise they are CZ.
    """
    qc = Circuit(n_qubits)

    for _ in range(n_cs):
        qubit1 = random.randint(0, n_qubits - 1)
        qubit2 = random.randint(0, n_qubits - 1)

        while (qubit1 == qubit2):
            qubit1 = random.randint(0, n_qubits - 1)
            qubit2 = random.randint(0, n_qubits - 1)

        if cx:
            qc.add(gates.CX(qubit1, qubit2))
        else:
            qc.add(gates.CZ(qubit1, qubit2))

    return qc

def gen_transpiled_circuits_qibo(circuit, conn, placer, router):
    """
    Transpile a Qibo circuit using the given connectivity, placer, and router.
    Optimiser and unroller are set to default.
    """
    default_gates = NativeGates.default()

    # Optimizer / Unroller
    opt_preprocessing = Preprocessing(connectivity=conn)
    unr = Unroller(default_gates)

    custom_passes = []
    custom_passes.append(opt_preprocessing)
    custom_passes.append(placer)
    custom_passes.append(router)
    custom_passes.append(unr)

    custom_pipeline = Passes(custom_passes, connectivity=conn)
    transpiled_circ, final_layout = custom_pipeline(circuit)
    return transpiled_circ


def qft_rotations(circuit, n):
    if n == 0:
        return circuit
    n -= 1
    circuit.h(n)
    for qubit in range(n):
        circuit.cp(np.pi/2**(n-qubit), qubit, n)
    qft_rotations(circuit, n)

def swap_registers(circuit, n):
    for qubit in range(n//2):
        circuit.swap(qubit, n-qubit-1)
    return circuit

def qft(circuit, n):
    qft_rotations(circuit, n)
    swap_registers(circuit, n)
    return circuit

def qiskit_qft(n):
    """
    Returns a Qiskit QuantumCircuit object for the QFT on n qubits.
    """
    qc = QuantumCircuit(n)
    qft(qc, n)
    qc = qc.reverse_bits()
    return qc
