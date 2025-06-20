# import projectors

import numpy as np
import RGD
from BasicTools import Plt_Err_Time

from qibo.models.encodings import ghz_state


def qibo_Pauli_symbols(labels):
    """convert string representation of Pauli operator list
        to list of symbolic representation

    Args:
        labels (list): list of Pauli operator labels,
            (eg) list of Pauli terms  ["YXY", "IXX", "ZYI", "XXX", "YZZ"]
    Returns:
        list (Proj_list): list of symbolic Pauli operators
    """
    from functools import reduce

    from qibo.hamiltonians import SymbolicHamiltonian
    from qibo.symbols import I, X, Y, Z

    symbol_map = {
        "X": X,
        "Y": Y,
        "Z": Z,
        "I": I,
    }

    Proj_list = []
    for label in labels:
        qubitPauli = [symbol_map[Ps](i) for i, Ps in enumerate(label)]
        symbolPauli = SymbolicHamiltonian(reduce(lambda x, y: x * y, qubitPauli))
        Proj_list.append(symbolPauli)

    return Proj_list


def qibochem_measure_expectation(circuit, labels, num_shots):
    """Use qibochem.measurement package to get the expecation values
        of given list of Pauli terms for a particular given circuit

    Args:
        circuit (circuit): the circuit generating the state
        labels (list): list of Pauli operator labels,
            (eg) list of Pauli terms  ["YXY", "IXX", "ZYI", "XXX", "YZZ"]
        num_shots (int): number of shot measurements

    Returns:
        list (coef_Pauli_exact): list of exact expectation value of Pauli Terms
        list (coef_Pauli_shots): list of Pauli expectation from shot measurements
    """
    from functools import reduce

    from qibochem.measurement import expectation, expectation_from_samples

    from qibo.hamiltonians import SymbolicHamiltonian
    from qibo.symbols import I, X, Y, Z

    symbol_map = {
        "X": X,
        "Y": Y,
        "Z": Z,
        "I": I,
    }

    _circuit = circuit.copy()
    coef_Pauli_exact = []  # list of exact expectation value of Pauli Terms
    coef_Pauli_shots = []  # list of Pauli expectation from shot measurements
    for label in labels:
        print(f"label = {label}")

        qubitPauli = [symbol_map[Ps](i) for i, Ps in enumerate(label)]
        symbolPauli = SymbolicHamiltonian(reduce(lambda x, y: x * y, qubitPauli))

        coef_Pauli_exact.append(expectation(_circuit, symbolPauli))
        coef_Pauli_shots.append(
            expectation_from_samples(_circuit, symbolPauli, n_shots=num_shots)
        )

    return coef_Pauli_exact, coef_Pauli_shots


def generate_random_label(n, symbols=["I", "X", "Y", "Z"]):
    num_symbols = len(symbols)
    label = "".join([symbols[i] for i in np.random.randint(0, num_symbols, size=n)])
    return label


if __name__ == "__main__":

    ############################################################
    ### Example of creating and running an experiment
    ############################################################
    num_labels = 5  #  number of Pauli labels
    n = 3  #  number of qubits
    # labels = [generate_random_label(n) for i in range(num_labels)]

    labels = ["YXY", "IXX", "ZYI", "XXX", "YZZ", "ZYY", "IXX", "XYY", "XZI"]
    # labels = ['YZY', 'YYZ', 'XIY', 'IZY', 'YYY','XZI','IXZ','IIY','XXY','YZZ']
    # labels = ['XXYY', 'YXZY', 'XZIZ', 'ZYYZ', 'YXYX', 'ZXXZ', 'ZIYZ', 'YXYY', 'YXYY', 'ZYYY']
    # labels = ['ZIYZ', 'YXZX', 'ZIXX', 'YXYX', 'ZYXX']
    # labels = ['IZZ', 'IIZ','ZIZ']    # NotImplementedError: Observable is not a Z Pauli string.

    symProj_list = qibo_Pauli_symbols(labels)

    #   generate circuit & shot measurements
    #
    Nr = 1  #  rank of the target density matrix

    stateGHZ = ghz_state(n)  # generate GHZ state circuit
    target_state_GHZ = stateGHZ.execute().state()  # get the state vector of the circuit
    target_density_matrix = np.outer(target_state_GHZ, target_state_GHZ.conj())

    num_shots = 200  # number of shot measurements

    coef_Pauli_exact, measurement_list = qibochem_measure_expectation(
        stateGHZ, labels, num_shots
    )  # get the expectation values of Pauli terms (exact and from shots)
    print(coef_Pauli_exact)
    print(measurement_list)

    #
    #   system parameters as the input to the RGD worker
    #
    params_dict = {
        "Nr": Nr,
        "target_DM": target_density_matrix,
        "labels": labels,
        "measurement_list": measurement_list,
        "symProj_list": symProj_list,
        "num_iterations": 150,
        "convergence_check_period": 1,
    }

    Ch_svd = -1  #   choice for initial SVD  (0: LA.svd, 1: svds, -1: rSVD)
    InitX_RGD = 1  #   method of choosing initial X0

    worker = RGD.BasicWorkerRGD(params_dict)
    worker.computeRGD(InitX_RGD, Ch_svd)

    Plt_Err_Time(worker)  # plot the error and time evolution of the RGD algorithm
