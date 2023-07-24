#!/usr/bin/env python3
import numpy as np

from qibo import Circuit, gates


class QSVD:
    def __init__(self, nqubits, subsize, nlayers, RY=False):
        """
        Class for the Quantum Singular Value Decomposer variational algorithm

        Args:
            nqubits: number of qubits
            subsize: size of the subsystem with qubits 0,1,...,sub_size-1
            nlayers: number of layers of the varitional ansatz
            RY: if True, parameterized Ry gates are used in the circuit
                if False, parameterized Rx,Rz,Rx gates are used in the circuit (default=False)
        """
        self.nqubits = nqubits
        self.subsize = subsize
        self.subsize2 = nqubits - subsize

        if RY:

            def rotations():
                for q in range(self.nqubits):
                    yield gates.RY(q, theta=0)

        else:

            def rotations():
                for q in range(self.nqubits):
                    yield gates.RX(q, theta=0)
                    yield gates.RZ(q, theta=0)
                    yield gates.RX(q, theta=0)

        self._circuit = self.ansatz(nlayers, rotations)

    def _CZ_gates(self):
        """Yields CZ gates used in the variational circuit."""
        # U
        for q in range(0, self.subsize - 1, 2):
            yield gates.CZ(q, q + 1)
        # V
        for q in range(self.subsize, self.nqubits - 1, 2):
            yield gates.CZ(q, q + 1)

    def ansatz(self, nlayers, rotations):
        """
        Args:
            nlayers: number of layers of the varitional ansatz
            rotations: Function that generates rotation gates (defined in __init__)

        Returns:
            Circuit model implementing the variational ansatz
        """
        c = Circuit(self.nqubits)
        for _ in range(nlayers):
            c.add(rotations())
            c.add(self._CZ_gates())
            c.add(rotations())
            c.add(self._CZ_gates())
        # Final rotations
        c.add(rotations())
        # Measurements
        small = min(self.subsize, self.subsize2)
        for q in range(small):
            c.add(gates.M(q))
            c.add(gates.M(q + self.subsize))
        return c

    def QSVD_circuit(self, theta):
        """
        Args:
            theta: list or numpy.array with the angles to be used in the circuit

        Returns:
            Circuit model implementing the variational ansatz for QSVD
        """
        self._circuit.set_parameters(theta)
        return self._circuit

    def QSVD_cost(self, theta, init_state=None, nshots=100000):
        """
        Args:
            theta: list or numpy.array with the angles to be used in the circuit
            init_state: numpy.array with the quantum state to be Schmidt-decomposed
            nshots: int number of runs of the circuit during the sampling process (default=100000)

        Returns:
            numpy.float32 with the value of the cost function for the QSVD with angles theta
        """

        def Hamming(string1, string2):
            """
            Args:
                two strings to be compared

            Returns:
                Hamming distance of the strings
            """
            return sum(q1 != q2 for q1, q2 in zip(string1, string2))

        Circuit_ansatz = self.QSVD_circuit(theta)
        result = Circuit_ansatz(init_state, nshots)
        result = result.frequencies(binary=True)

        loss = 0
        for bit_string in result:
            a = bit_string[: self.subsize2]
            b = bit_string[self.subsize2 :]
            loss += Hamming(a, b) * result[bit_string]
        return loss / nshots

    def minimize(
        self, init_theta, init_state=None, nshots=100000, method="Powell", maxiter=None
    ):
        """
        Args:
            theta: list or numpy.array with the angles to be used in the circuit
            init_state: numpy.array with the quantum state to be Schmidt-decomposed
            nshots: int number of runs of the circuit during the sampling process (default=100000)
            method: 'classical optimizer for the minimization. All methods from scipy.optimize.minmize are suported (default='Powell')

        Returns:
            numpy.float64 with value of the minimum found, numpy.ndarray with the optimal angles
        """
        from scipy.optimize import minimize

        result = minimize(
            self.QSVD_cost,
            init_theta,
            args=(init_state, nshots),
            method=method,
            options={"disp": True, "maxiter": maxiter},
        )
        loss = result.fun
        optimal_angles = result.x

        return loss, optimal_angles

    def Schmidt_coeff(self, theta, init_state, nshots=100000):
        """
        Args:
            theta: list or numpy.array with the angles to be used in the circuit
            init_state: numpy.array with the quantum state to be Schmidt-decomposed
            nshots: int number of runs of the circuit during the sampling process (default=100000)

        Returns:
            numpy.array with the Schmidt coefficients given by the QSVD, in decreasing order
        """
        Qsvd = self.QSVD_circuit(theta)
        result = Qsvd(init_state, nshots)

        result = result.frequencies(binary=True)
        small = min(self.subsize, self.subsize2)

        Schmidt = []
        for i in range(2**small):
            bit_string = bin(i)[2:].zfill(small)
            Schmidt.append(result[2 * bit_string])

        Schmidt = np.array(sorted(Schmidt, reverse=True))
        Schmidt = np.sqrt(Schmidt / nshots)

        return Schmidt / np.linalg.norm(Schmidt)

    def VonNeumann_entropy(self, theta, init_state, tol=1e-14, nshots=100000):
        """
        Args:
            theta: list or numpy.array with the angles to be used in the circuit
            init_state: numpy.array with the quantum state to be Schmidt-decomposed
            nshots: int number of runs of the circuit during the sampling process (default=10000)

        Returns:
            numpy.float64 with the value of the Von Neumann entropy for the given bipartition
        """
        Schmidt = self.Schmidt_coeff(theta, init_state, nshots=nshots)
        Schmidt = Schmidt**2

        non_zero_coeff = np.array([coeff for coeff in Schmidt if coeff > tol])

        return -np.sum(non_zero_coeff * np.log2(non_zero_coeff))
