#!/usr/bin/env python3
import argparse

import numpy as np
from scipy.optimize import minimize
from sklearn.datasets import load_digits  # type: ignore

from qibo import Circuit, gates
from qibo.hamiltonians import TFIM, Hamiltonian, Z


def main(layers, autoencoder, example, maxiter):
    def encoder_hamiltonian_simple(nqubits, ncompress):
        """Creates the encoding Hamiltonian.
        Args:
            nqubits (int): total number of qubits.
            ncompress (int): number of discarded/trash qubits.

        Returns:
            Encoding Hamiltonian.
        """
        m0 = Z(ncompress).matrix
        m1 = np.eye(2 ** (nqubits - ncompress), dtype=m0.dtype)
        ham = Hamiltonian(nqubits, np.kron(m1, m0))
        return 0.5 * (ham + ncompress)

    def rotate(theta, x):
        new_theta = []
        index = 0
        for l in range(layers):
            for q in range(nqubits):
                new_theta.append(theta[index] * x + theta[index + 1])
                index += 2
            for q in range(nqubits):
                new_theta.append(theta[index] * x + theta[index + 1])
                index += 2
        for q in range(nqubits - compress, nqubits, 1):
            new_theta.append(theta[index] * x + theta[index + 1])
            index += 2
        return new_theta

    cost_function_steps = []
    nqubits = 6
    compress = 2
    encoder = encoder_hamiltonian_simple(nqubits, compress)
    count = [0]

    if example == 0:
        ising_groundstates = []
        lambdas = np.linspace(0.5, 1.0, 20)
        for lamb in lambdas:
            ising_ham = -1 * TFIM(nqubits, h=lamb)
            ising_groundstates.append(ising_ham.eigenvectors()[0])

        if autoencoder == 1:
            circuit = Circuit(nqubits)
            for l in range(layers):
                for q in range(nqubits):
                    circuit.add(gates.RY(q, theta=0))
                circuit.add(gates.CZ(5, 4))
                circuit.add(gates.CZ(5, 3))
                circuit.add(gates.CZ(5, 1))
                circuit.add(gates.CZ(4, 2))
                circuit.add(gates.CZ(4, 0))
                for q in range(nqubits):
                    circuit.add(gates.RY(q, theta=0))
                circuit.add(gates.CZ(5, 4))
                circuit.add(gates.CZ(5, 2))
                circuit.add(gates.CZ(4, 3))
                circuit.add(gates.CZ(5, 0))
                circuit.add(gates.CZ(4, 1))
            for q in range(nqubits - compress, nqubits, 1):
                circuit.add(gates.RY(q, theta=0))

            def cost_function_QAE_Ising(params, count):
                """Evaluates the cost function to be minimized for the QAE and Ising model.

                Args:
                    params (array or list): values of the parameters.

                Returns:
                    Value of the cost function.
                """
                cost = 0
                circuit.set_parameters(
                    params
                )  # this will change all thetas to the appropriate values
                for i in range(len(ising_groundstates)):
                    final_state = circuit.execute(
                        np.copy(ising_groundstates[i])
                    ).state()
                    cost += np.real(encoder.expectation(final_state))

                cost_function_steps.append(
                    cost / len(ising_groundstates)
                )  # save cost function value after each step

                if count[0] % 50 == 0:
                    print(count[0], cost / len(ising_groundstates))
                count[0] += 1

                return cost / len(ising_groundstates)

            nparams = 2 * nqubits * layers + compress
            initial_params = np.random.uniform(0, 2 * np.pi, nparams)

            result = minimize(
                cost_function_QAE_Ising,
                initial_params,
                args=(count),
                method="BFGS",
                options={"maxiter": maxiter},
            )

        elif autoencoder == 0:
            circuit = Circuit(nqubits)
            for l in range(layers):
                for q in range(nqubits):
                    circuit.add(gates.RY(q, theta=0))
                circuit.add(gates.CZ(5, 4))
                circuit.add(gates.CZ(5, 3))
                circuit.add(gates.CZ(5, 1))
                circuit.add(gates.CZ(4, 2))
                circuit.add(gates.CZ(4, 0))
                for q in range(nqubits):
                    circuit.add(gates.RY(q, theta=0))
                circuit.add(gates.CZ(5, 4))
                circuit.add(gates.CZ(5, 2))
                circuit.add(gates.CZ(4, 3))
                circuit.add(gates.CZ(5, 0))
                circuit.add(gates.CZ(4, 1))
            for q in range(nqubits - compress, nqubits, 1):
                circuit.add(gates.RY(q, theta=0))

            def cost_function_EF_QAE_Ising(params, count):
                """Evaluates the cost function to be minimized for the EF-QAE and Ising model.

                Args:
                    params (array or list): values of the parameters.

                Returns:
                    Value of the cost function.
                """
                cost = 0
                for i in range(len(ising_groundstates)):
                    newparams = rotate(params, lambdas[i])
                    circuit.set_parameters(newparams)
                    final_state = circuit.execute(
                        np.copy(ising_groundstates[i])
                    ).state()
                    cost += np.real(encoder.expectation(final_state))

                cost_function_steps.append(
                    cost / len(ising_groundstates)
                )  # save cost function value after each step

                if count[0] % 50 == 0:
                    print(count[0], cost / len(ising_groundstates))
                count[0] += 1

                return cost / len(ising_groundstates)

            nparams = 4 * nqubits * layers + 2 * compress
            initial_params = np.random.uniform(0, 2 * np.pi, nparams)

            result = minimize(
                cost_function_EF_QAE_Ising,
                initial_params,
                args=(count),
                method="BFGS",
                options={"maxiter": maxiter},
            )

        else:
            raise ValueError(
                "You have to introduce a value of 0 or 1 in the autoencoder argument."
            )

    elif example == 1:
        digits = load_digits()
        vector_0 = []
        vector_1 = []
        for value in [0, 10, 20, 30, 36, 48, 49, 55, 72, 78]:
            vector_0.append(
                np.array(digits.data[value])
                / np.linalg.norm(np.array(digits.data[value]))
            )
        for value in [1, 11, 21, 42, 47, 56, 70, 85, 90, 93]:
            vector_1.append(
                np.array(digits.data[value])
                / np.linalg.norm(np.array(digits.data[value]))
            )

        if autoencoder == 1:
            circuit = Circuit(nqubits)
            for l in range(layers):
                for q in range(nqubits):
                    circuit.add(gates.RY(q, theta=0))
                circuit.add(gates.CZ(5, 4))
                circuit.add(gates.CZ(5, 3))
                circuit.add(gates.CZ(5, 1))
                circuit.add(gates.CZ(4, 2))
                circuit.add(gates.CZ(4, 0))
                for q in range(nqubits):
                    circuit.add(gates.RY(q, theta=0))
                circuit.add(gates.CZ(5, 4))
                circuit.add(gates.CZ(5, 2))
                circuit.add(gates.CZ(4, 3))
                circuit.add(gates.CZ(5, 0))
                circuit.add(gates.CZ(4, 1))
            for q in range(nqubits - compress, nqubits, 1):
                circuit.add(gates.RY(q, theta=0))

            def cost_function_QAE_Digits(params, count):
                """Evaluates the cost function to be minimized for the QAE and Handwritten digits.

                Args:
                    params (array or list): values of the parameters.

                Returns:
                    Value of the cost function.
                """
                cost = 0
                circuit.set_parameters(
                    params
                )  # this will change all thetas to the appropriate values
                for i in range(len(vector_0)):
                    final_state = circuit.execute(np.copy(vector_0[i])).state()
                    cost += np.real(encoder.expectation(final_state))
                for i in range(len(vector_1)):
                    final_state = circuit.execute(np.copy(vector_1[i])).state()
                    cost += np.real(encoder.expectation(final_state))

                cost_function_steps.append(
                    cost / (len(vector_0) + len(vector_1))
                )  # save cost function value after each step

                if count[0] % 50 == 0:
                    print(count[0], cost / (len(vector_0) + len(vector_1)))
                count[0] += 1

                return cost / (len(vector_0) + len(vector_1))

            nparams = 2 * nqubits * layers + compress
            initial_params = np.random.uniform(0, 2 * np.pi, nparams)

            result = minimize(
                cost_function_QAE_Digits,
                initial_params,
                args=(count),
                method="BFGS",
                options={"maxiter": maxiter},
            )

        elif autoencoder == 0:
            circuit = Circuit(nqubits)
            for l in range(layers):
                for q in range(nqubits):
                    circuit.add(gates.RY(q, theta=0))
                circuit.add(gates.CZ(5, 4))
                circuit.add(gates.CZ(5, 3))
                circuit.add(gates.CZ(5, 1))
                circuit.add(gates.CZ(4, 2))
                circuit.add(gates.CZ(4, 0))
                for q in range(nqubits):
                    circuit.add(gates.RY(q, theta=0))
                circuit.add(gates.CZ(5, 4))
                circuit.add(gates.CZ(5, 2))
                circuit.add(gates.CZ(4, 3))
                circuit.add(gates.CZ(5, 0))
                circuit.add(gates.CZ(4, 1))
            for q in range(nqubits - compress, nqubits, 1):
                circuit.add(gates.RY(q, theta=0))

            def cost_function_EF_QAE_Digits(params, count):
                """Evaluates the cost function to be minimized for the EF-QAE and Handwritten digits.

                Args:
                    params (array or list): values of the parameters.

                Returns:
                    Value of the cost function.
                """
                cost = 0
                newparams = rotate(params, 1)
                circuit.set_parameters(newparams)
                for i in range(len(vector_0)):
                    final_state = circuit.execute(np.copy(vector_0[i])).state()
                    cost += np.real(encoder.expectation(final_state))
                newparams = rotate(params, 2)
                circuit.set_parameters(newparams)
                for i in range(len(vector_1)):
                    final_state = circuit.execute(np.copy(vector_1[i])).state()
                    cost += np.real(encoder.expectation(final_state))

                cost_function_steps.append(
                    cost / (len(vector_0) + len(vector_1))
                )  # save cost function value after each step

                if count[0] % 50 == 0:
                    print(count[0], cost / (len(vector_0) + len(vector_1)))
                count[0] += 1

                return cost / (len(vector_0) + len(vector_1))

            nparams = 4 * nqubits * layers + 2 * compress
            initial_params = np.random.uniform(0, 2 * np.pi, nparams)

            result = minimize(
                cost_function_EF_QAE_Digits,
                initial_params,
                args=(count),
                method="BFGS",
                options={"maxiter": maxiter},
            )

        else:
            raise ValueError(
                "You have to introduce a value of 0 or 1 in the autoencoder argument."
            )

    else:
        raise ValueError(
            "You have to introduce a value of 0 or 1 in the example argument."
        )

    print("Final parameters: ", result.x)
    print("Final cost function: ", result.fun)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--layers", default=3, type=int, help="(int): number of ansatz layers"
    )
    parser.add_argument(
        "--autoencoder",
        default=0,
        type=int,
        help="(int): 0 to run the EF-QAE or 1 to run the QAE",
    )
    parser.add_argument(
        "--example",
        default=0,
        type=int,
        help="(int): 0 to run Ising model example or 1 to run the Handwritten digits example",
    )
    parser.add_argument(
        "--maxiter", default=50000, type=int, help="(int): maximum number of iterations"
    )
    args = parser.parse_args()
    main(**vars(args))
