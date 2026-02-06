# import projectors

from functools import reduce

import numpy as np
import RGD
from BasicTools import Plt_Err_Time
from qibochem.measurement import expectation

from qibo.hamiltonians import SymbolicHamiltonian
from qibo.models.encodings import ghz_state
from qibo.symbols import I, X, Y, Z

symbol_map = {
    "X": X,
    "Y": Y,
    "Z": Z,
    "I": I,
}


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
    num_shots = 200  # number of shot measurements

    #   generate circuit & shot measurements
    #
    Nr = 1  #  rank of the target density matrix

    stateGHZ = ghz_state(n)  # generate GHZ state circuit
    target_state_GHZ = stateGHZ.execute().state()  # get the state vector of the circuit
    target_density_matrix = np.outer(target_state_GHZ, target_state_GHZ.conj())

    # labels = [generate_random_label(n) for i in range(num_labels)]

    # labels = ["YXY", "IXX", "ZYI", "XXX", "YZZ", "ZYY", "IXX", "XYY", "XZI"]
    # labels = ['YZY', 'YYZ', 'XIY', 'IZY', 'YYY','XZI','IXZ','IIY','XXY','YZZ']
    # labels = ['XXYY', 'YXZY', 'XZIZ', 'ZYYZ', 'YXYX', 'ZXXZ', 'ZIYZ', 'YXYY', 'YXYY', 'ZYYY']
    # labels = ['ZIYZ', 'YXZX', 'ZIXX', 'YXYX', 'ZYXX']
    labels = ["IIZ", "ZIZ", "IZI", "ZII", "IIZ", "IZZ"]

    Proj_list = []
    coef_exact = []
    coef_shots = []
    for label in labels:
        # print(f"label = {label}")

        qubitPauli = [symbol_map[Ps](i) for i, Ps in enumerate(label)]
        symbolPauli = SymbolicHamiltonian(reduce(lambda x, y: x * y, qubitPauli))

        _circuit = stateGHZ.copy()

        coef_Pauli_exact = expectation(_circuit, symbolPauli)
        if label == "III":
            coef_Pauli_shots = 1.0
        else:
            coef_Pauli_shots = symbolPauli.expectation_from_circuit(
                _circuit, nshots=num_shots
            )

        Proj_list.append(symbolPauli)
        coef_exact.append(coef_Pauli_exact)
        coef_shots.append(coef_Pauli_shots.real)

    #
    #   system parameters as the input to the RGD worker
    #
    params_dict = {
        "Nr": Nr,
        "target_DM": target_density_matrix,
        "labels": labels,
        "measurement_list": coef_shots,  # measurement_list,
        "symProj_list": Proj_list,  # symProj_list,
        # "num_iterations": 150,
        # "convergence_check_period": 1,
    }

    Ch_svd = -1  #   choice for initial SVD  (0: LA.svd, 1: svds, -1: rSVD)
    InitX_RGD = 1  #   method of choosing initial X0

    worker = RGD.BasicWorkerRGD(
        target_density_matrix, labels, coef_shots, Proj_list, Nr
    )
    worker.computeRGD(InitX_RGD, Ch_svd)

    Plt_Err_Time(worker)  # plot the error and time evolution of the RGD algorithm
