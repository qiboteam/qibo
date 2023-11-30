"""Module defining the Clifford backend."""
from functools import cache

from qibo import gates
from qibo.backends.numpy import NumpyBackend
from qibo.config import raise_error
from qibo.result import CircuitResult


class CliffordOperations:
    """Operations performed by Clifford gates on the phase-space representation of stabilizer states.

    See `Aaronson & Gottesman (2004) <https://arxiv.org/abs/quant-ph/0406196>`_.
    """

    def __init__(self):
        import numpy as np

        self.np = np

    def I(self, symplectic_matrix, q, nqubits):
        return symplectic_matrix

    def H(self, symplectic_matrix, q, nqubits):
        r, x, z = self.get_rxz(symplectic_matrix, nqubits)
        self.set_r(
            symplectic_matrix,
            r ^ (x[:, q] * z[:, q]).flatten(),
        )
        symplectic_matrix[:, [q, nqubits + q]] = symplectic_matrix[:, [nqubits + q, q]]
        return symplectic_matrix

    def CNOT(self, symplectic_matrix, control_q, target_q, nqubits):
        r, x, z = self.get_rxz(symplectic_matrix, nqubits)
        self.set_r(
            symplectic_matrix,
            r
            ^ (x[:, control_q] * z[:, target_q]).flatten()
            * (x[:, target_q] ^ z[:, control_q] ^ 1).flatten(),
        )

        symplectic_matrix[:-1, target_q] = x[:, target_q] ^ x[:, control_q]
        symplectic_matrix[:-1, nqubits + control_q] = z[:, control_q] ^ z[:, target_q]
        return symplectic_matrix

    def CZ(self, symplectic_matrix, control_q, target_q, nqubits):
        """Decomposition --> H-CNOT-H"""

        r, x, z = self.get_rxz(symplectic_matrix, nqubits)
        self.set_r(
            symplectic_matrix,
            r
            ^ (x[:, target_q] * z[:, target_q]).flatten()
            ^ (
                x[:, control_q]
                * x[:, target_q]
                * (z[:, target_q] ^ z[:, control_q] ^ 1)
            ).flatten()
            ^ (x[:, target_q] * (z[:, target_q] ^ x[:, control_q])).flatten(),
        )
        symplectic_matrix[
            :-1, [nqubits + control_q, nqubits + target_q]
        ] = self.np.vstack(
            (x[:, target_q] ^ z[:, control_q], z[:, target_q] ^ x[:, control_q])
        ).T
        return symplectic_matrix

    def S(self, symplectic_matrix, q, nqubits):
        r, x, z = self.get_rxz(symplectic_matrix, nqubits)
        self.set_r(
            symplectic_matrix,
            r ^ (x[:, q] * z[:, q]).flatten(),
        )
        symplectic_matrix[:-1, nqubits + q] = z[:, q] ^ x[:, q]
        return symplectic_matrix

    def Z(self, symplectic_matrix, q, nqubits):
        """Decomposition --> S-S"""

        r, x, z = self.get_rxz(symplectic_matrix, nqubits)
        self.set_r(
            symplectic_matrix,
            r ^ ((x[:, q] * z[:, q]) ^ x[:, q] * (z[:, q] ^ x[:, q])).flatten(),
        )
        return symplectic_matrix

    def X(self, symplectic_matrix, q, nqubits):
        """Decomposition --> H-S-S-H"""

        r, x, z = self.get_rxz(symplectic_matrix, nqubits)
        self.set_r(
            symplectic_matrix,
            r
            ^ (z[:, q] * (z[:, q] ^ x[:, q])).flatten()
            ^ (z[:, q] * x[:, q]).flatten(),
        )
        return symplectic_matrix

    def Y(self, symplectic_matrix, q, nqubits):
        """Decomposition --> S-S-H-S-S-H"""  # double check this, cause it should be
        # Y = i * HZHZ --> HSSHSS
        r, x, z = self.get_rxz(symplectic_matrix, nqubits)
        self.set_r(
            symplectic_matrix,
            r
            ^ (z[:, q] * (z[:, q] ^ x[:, q])).flatten()
            ^ (x[:, q] * (z[:, q] ^ x[:, q])).flatten(),
        )
        return symplectic_matrix

    def SX(self, symplectic_matrix, q, nqubits):
        """Decomposition --> H-S-H"""

        r, x, z = self.get_rxz(symplectic_matrix, nqubits)
        self.set_r(
            symplectic_matrix,
            r ^ (z[:, q] * (z[:, q] ^ x[:, q])).flatten(),
        )
        symplectic_matrix[:-1, q] = z[:, q] ^ x[:, q]
        return symplectic_matrix

    def SDG(self, symplectic_matrix, q, nqubits):
        """Decomposition --> S-S-S"""

        r, x, z = self.get_rxz(symplectic_matrix, nqubits)
        self.set_r(
            symplectic_matrix,
            r ^ (x[:, q] * (z[:, q] ^ x[:, q])).flatten(),
        )
        symplectic_matrix[:-1, nqubits + q] = z[:, q] ^ x[:, q]
        return symplectic_matrix

    def SXDG(self, symplectic_matrix, q, nqubits):
        """Decomposition --> H-S-S-S-H"""

        r, x, z = self.get_rxz(symplectic_matrix, nqubits)
        self.set_r(
            symplectic_matrix,
            r ^ (z[:, q] * x[:, q]).flatten(),
        )
        symplectic_matrix[:-1, q] = z[:, q] ^ x[:, q]
        return symplectic_matrix

    def RX(self, symplectic_matrix, q, nqubits, theta):
        if theta % (2 * self.np.pi) == 0:
            return self.I(symplectic_matrix, q, nqubits)
        elif (theta / self.np.pi - 1) % 2 == 0:
            return self.X(symplectic_matrix, q, nqubits)
        elif (theta / (self.np.pi / 2) - 1) % 4 == 0:
            return self.SX(symplectic_matrix, q, nqubits)
        else:  # theta == 3*pi/2 + 2*n*pi
            return self.SXDG(symplectic_matrix, q, nqubits)

    def RZ(self, symplectic_matrix, q, nqubits, theta):
        if theta % (2 * self.np.pi) == 0:
            return self.I(symplectic_matrix, q, nqubits)
        elif (theta / self.np.pi - 1) % 2 == 0:
            return self.Z(symplectic_matrix, q, nqubits)
        elif (theta / (self.np.pi / 2) - 1) % 4 == 0:
            return self.S(symplectic_matrix, q, nqubits)
        else:  # theta == 3*pi/2 + 2*n*pi
            return self.SDG(symplectic_matrix, q, nqubits)

    def RY(self, symplectic_matrix, q, nqubits, theta):
        if theta % (2 * self.np.pi) == 0:
            return self.I(symplectic_matrix, q, nqubits)
        elif (theta / self.np.pi - 1) % 2 == 0:
            return self.Y(symplectic_matrix, q, nqubits)
        elif (theta / (self.np.pi / 2) - 1) % 4 == 0:
            """Decomposition --> H-S-S"""

            r, x, z = self.get_rxz(symplectic_matrix, nqubits)
            self.set_r(
                symplectic_matrix,
                r ^ (x[:, q] * (z[:, q] ^ x[:, q])).flatten(),
            )
            symplectic_matrix[:-1, [nqubits + q, q]] = symplectic_matrix[
                :-1, [q, nqubits + q]
            ]
            return symplectic_matrix
        else:  # theta == 3*pi/2 + 2*n*pi
            """Decomposition --> H-S-S-H-S-S-H-S-S"""

            r, x, z = self.get_rxz(symplectic_matrix, nqubits)
            self.set_r(
                symplectic_matrix,
                r ^ (z[:, q] * (z[:, q] ^ x[:, q])).flatten(),
            )
            symplectic_matrix[:-1, [nqubits + q, q]] = symplectic_matrix[
                :-1, [q, nqubits + q]
            ]
            return symplectic_matrix

    def SWAP(self, symplectic_matrix, control_q, target_q, nqubits):
        """Decomposition --> CNOT-CNOT-CNOT"""

        r, x, z = self.get_rxz(symplectic_matrix, nqubits)
        self.set_r(
            symplectic_matrix,
            r
            ^ (
                x[:, control_q]
                * z[:, target_q]
                * (x[:, target_q] ^ z[:, control_q] ^ 1)
            ).flatten()
            ^ (
                (x[:, target_q] ^ x[:, control_q])
                * (z[:, target_q] ^ z[:, control_q])
                * (z[:, target_q] ^ x[:, control_q] ^ 1)
            ).flatten()
            ^ (
                x[:, target_q]
                * z[:, control_q]
                * (
                    x[:, control_q]
                    ^ x[:, target_q]
                    ^ z[:, control_q]
                    ^ z[:, target_q]
                    ^ 1
                )
            ).flatten(),
        )
        symplectic_matrix[
            :-1, [control_q, target_q, nqubits + control_q, nqubits + target_q]
        ] = symplectic_matrix[
            :-1, [target_q, control_q, nqubits + target_q, nqubits + control_q]
        ]
        return symplectic_matrix

    def iSWAP(self, symplectic_matrix, control_q, target_q, nqubits):
        """Decomposition --> H-CNOT-CNOT-H-S-S"""

        r, x, z = self.get_rxz(symplectic_matrix, nqubits)
        self.set_r(
            symplectic_matrix,
            r
            ^ (x[:, target_q] * z[:, target_q]).flatten()
            ^ (x[:, control_q] * z[:, control_q]).flatten()
            ^ (x[:, control_q] * (z[:, control_q] ^ x[:, control_q])).flatten()
            ^ (
                (z[:, control_q] ^ x[:, control_q])
                * (z[:, target_q] ^ x[:, target_q])
                * (x[:, target_q] ^ x[:, control_q] ^ 1)
            ).flatten()
            ^ (
                (x[:, target_q] ^ z[:, control_q] ^ x[:, control_q])
                * (x[:, target_q] ^ z[:, target_q] ^ x[:, control_q])
                * (
                    x[:, target_q]
                    ^ z[:, target_q]
                    ^ x[:, control_q]
                    ^ z[:, control_q]
                    ^ 1
                )
            ).flatten()
            ^ (
                x[:, control_q] * (x[:, target_q] ^ x[:, control_q] ^ z[:, control_q])
            ).flatten(),
        )
        symplectic_matrix[
            :-1, [nqubits + control_q, nqubits + target_q]
        ] = self.np.vstack(
            (
                x[:, target_q] ^ z[:, target_q] ^ x[:, control_q],
                x[:, target_q] ^ z[:, control_q] ^ x[:, control_q],
            )
        ).T
        symplectic_matrix[:-1, [control_q, target_q]] = symplectic_matrix[
            :-1, [target_q, control_q]
        ]
        return symplectic_matrix

    def FSWAP(self, symplectic_matrix, control_q, target_q, nqubits):
        """Decomposition --> X-CNOT-RY-CNOT-RY-CNOT-CNOT-X"""
        symplectic_matrix = self.X(symplectic_matrix, target_q, nqubits)
        symplectic_matrix = self.CNOT(symplectic_matrix, control_q, target_q, nqubits)
        symplectic_matrix = self.RY(
            symplectic_matrix, control_q, nqubits, self.np.pi / 2
        )
        symplectic_matrix = self.CNOT(symplectic_matrix, target_q, control_q, nqubits)
        symplectic_matrix = self.RY(
            symplectic_matrix, control_q, nqubits, -self.np.pi / 2
        )
        symplectic_matrix = self.CNOT(symplectic_matrix, target_q, control_q, nqubits)
        symplectic_matrix = self.CNOT(symplectic_matrix, control_q, target_q, nqubits)
        return self.X(symplectic_matrix, control_q, nqubits)

    def CY(self, symplectic_matrix, control_q, target_q, nqubits):
        """Decomposition --> S-CNOT-SDG"""

        r, x, z = self.get_rxz(symplectic_matrix, nqubits)
        self.set_r(
            symplectic_matrix,
            r
            ^ (x[:, target_q] * (z[:, target_q] ^ x[:, target_q])).flatten()
            ^ (
                x[:, control_q]
                * (x[:, target_q] ^ z[:, target_q])
                * (z[:, control_q] ^ x[:, target_q] ^ 1)
            ).flatten()
            ^ (
                (x[:, target_q] ^ x[:, control_q]) * (z[:, target_q] ^ x[:, target_q])
            ).flatten(),
        )

        symplectic_matrix[
            :-1, [target_q, nqubits + control_q, nqubits + target_q]
        ] = self.np.vstack(
            (
                x[:, control_q] ^ x[:, target_q],
                z[:, control_q] ^ z[:, target_q] ^ x[:, target_q],
                z[:, target_q] ^ x[:, control_q],
            )
        ).T
        return symplectic_matrix

    def CRX(self, symplectic_matrix, control_q, target_q, nqubits, theta):
        # theta = 4 * n * pi
        if theta % (4 * self.np.pi) == 0:
            return self.I(symplectic_matrix, target_q, nqubits)
        # theta = pi + 4 * n * pi
        elif (theta / self.np.pi - 1) % 4 == 0:
            symplectic_matrix = self.X(symplectic_matrix, target_q, nqubits)
            symplectic_matrix = self.CZ(symplectic_matrix, control_q, target_q, nqubits)
            symplectic_matrix = self.X(symplectic_matrix, target_q, nqubits)
            return self.CY(symplectic_matrix, control_q, target_q, nqubits)
        # theta = 2 * pi + 4 * n * pi
        elif (theta / (2 * self.np.pi) - 1) % 2 == 0:
            symplectic_matrix = self.CZ(symplectic_matrix, control_q, target_q, nqubits)
            symplectic_matrix = self.Y(symplectic_matrix, target_q, nqubits)
            symplectic_matrix = self.CZ(symplectic_matrix, control_q, target_q, nqubits)
            return self.Y(symplectic_matrix, target_q, nqubits)
        # theta = 3 * pi + 4 * n * pi
        elif (theta / self.np.pi - 3) % 4 == 0:
            symplectic_matrix = self.X(symplectic_matrix, target_q, nqubits)
            symplectic_matrix = self.CY(symplectic_matrix, control_q, target_q, nqubits)
            symplectic_matrix = self.X(symplectic_matrix, target_q, nqubits)
            return self.CZ(symplectic_matrix, control_q, target_q, nqubits)

    def CRZ(self, symplectic_matrix, control_q, target_q, nqubits, theta):
        # theta = 4 * n * pi
        if theta % (4 * self.np.pi) == 0:
            return self.I(symplectic_matrix, target_q, nqubits)
        # theta = pi + 4 * n * pi
        elif (theta / self.np.pi - 1) % 4 == 0:
            symplectic_matrix = self.X(symplectic_matrix, target_q, nqubits)
            symplectic_matrix = self.CY(symplectic_matrix, control_q, target_q, nqubits)
            symplectic_matrix = self.X(symplectic_matrix, target_q, nqubits)
            return self.CNOT(symplectic_matrix, control_q, target_q, nqubits)
        # theta = 2 * pi + 4 * n * pi
        elif (theta / (2 * self.np.pi) - 1) % 2 == 0:
            symplectic_matrix = self.CZ(symplectic_matrix, control_q, target_q, nqubits)
            symplectic_matrix = self.X(symplectic_matrix, target_q, nqubits)
            symplectic_matrix = self.CZ(symplectic_matrix, control_q, target_q, nqubits)
            return self.X(symplectic_matrix, target_q, nqubits)
        # theta = 3 * pi + 4 * n * pi
        elif (theta / self.np.pi - 3) % 4 == 0:
            symplectic_matrix = self.CNOT(
                symplectic_matrix, control_q, target_q, nqubits
            )
            symplectic_matrix = self.X(symplectic_matrix, target_q, nqubits)
            symplectic_matrix = self.CY(symplectic_matrix, control_q, target_q, nqubits)
            return self.X(symplectic_matrix, target_q, nqubits)

    def CRY(self, symplectic_matrix, control_q, target_q, nqubits, theta):
        # theta = 4 * n * pi
        if theta % (4 * self.np.pi) == 0:
            return self.I(symplectic_matrix, target_q, nqubits)
        # theta = pi + 4 * n * pi
        elif (theta / self.np.pi - 1) % 4 == 0:
            symplectic_matrix = self.Z(symplectic_matrix, target_q, nqubits)
            symplectic_matrix = self.CNOT(
                symplectic_matrix, control_q, target_q, nqubits
            )
            symplectic_matrix = self.Z(symplectic_matrix, target_q, nqubits)
            return self.CZ(symplectic_matrix, control_q, target_q, nqubits)
        # theta = 2 * pi + 4 * n * pi
        elif (theta / (2 * self.np.pi) - 1) % 2 == 0:
            return self.CRZ(symplectic_matrix, control_q, target_q, nqubits, theta)
        # theta = 3 * pi + 4 * n * pi
        elif (theta / self.np.pi - 3) % 4 == 0:
            symplectic_matrix = self.CZ(symplectic_matrix, control_q, target_q, nqubits)
            symplectic_matrix = self.Z(symplectic_matrix, target_q, nqubits)
            symplectic_matrix = self.CNOT(
                symplectic_matrix, control_q, target_q, nqubits
            )
            return self.Z(symplectic_matrix, target_q, nqubits)

    def ECR(self, symplectic_matrix, control_q, target_q, nqubits):
        symplectic_matrix = self.S(symplectic_matrix, control_q, nqubits)
        symplectic_matrix = self.SX(symplectic_matrix, target_q, nqubits)
        symplectic_matrix = self.CNOT(symplectic_matrix, control_q, target_q, nqubits)
        return self.X(symplectic_matrix, control_q, nqubits)

    # valid for a standard basis measurement only
    def M(self, state, qubits, nqubits, collapse=False):
        sample = []
        state_copy = state if collapse else state.copy()
        for q in qubits:
            x = CliffordOperations.get_x(state_copy, nqubits)
            p = x[nqubits:, q].nonzero()[0] + nqubits
            # random outcome, affects the state
            if len(p) > 0:
                p = p[0].item()
                for i in x[:, q].nonzero()[0]:
                    if i != p:
                        state_copy = self.rowsum(state_copy, i.item(), p, nqubits)
                state_copy[p - nqubits, :] = state_copy[p, :]
                outcome = self.np.random.randint(2, size=1).item()
                state_copy[p, :] = 0
                state_copy[p, -1] = outcome
                state_copy[p, nqubits + q] = 1
                sample.append(outcome)
            # determined outcome, state unchanged
            else:
                CliffordOperations.set_scratch(state_copy, 0)
                for i in x[:nqubits, q].nonzero()[0]:
                    state_copy = self.rowsum(
                        state_copy,
                        2 * nqubits,
                        i.item() + nqubits,
                        nqubits,
                        include_scratch=True,
                    )
                sample.append(int(CliffordOperations.get_scratch(state_copy)[-1]))
        return sample

    @staticmethod
    def get_r(symplectic_matrix, nqubits, include_scratch=False):
        return symplectic_matrix[
            : -1 + (2 * nqubits + 2) * int(include_scratch),
            -1,
        ]

    @staticmethod
    def set_r(symplectic_matrix, val):
        symplectic_matrix[:-1, -1] = val

    @staticmethod
    def get_x(symplectic_matrix, nqubits, include_scratch=False):
        return symplectic_matrix[
            : -1 + (2 * nqubits + 2) * int(include_scratch), :nqubits
        ]

    @staticmethod
    def get_z(symplectic_matrix, nqubits, include_scratch=False):
        return symplectic_matrix[
            : -1 + (2 * nqubits + 2) * int(include_scratch), nqubits:-1
        ]

    @staticmethod
    def get_rxz(symplectic_matrix, nqubits, include_scratch=False):
        return (
            CliffordOperations.get_r(symplectic_matrix, nqubits, include_scratch),
            CliffordOperations.get_x(symplectic_matrix, nqubits, include_scratch),
            CliffordOperations.get_z(symplectic_matrix, nqubits, include_scratch),
        )

    @staticmethod
    def get_scratch(symplectic_matrix):
        return symplectic_matrix[-1, :]

    @staticmethod
    def set_scratch(symplectic_matrix, val):
        symplectic_matrix[-1, :] = val

    @staticmethod
    @cache
    def exponent(x1, z1, x2, z2):
        if x1 == z1:
            if x1 == 0:
                return 0
            return int(z2) - int(x2)
        if x1 == 1:
            return int(z2) * (2 * int(x2) - 1)
        return int(x2) * (1 - 2 * int(z2))

    def rowsum(self, symplectic_matrix, h, i, nqubits, include_scratch: bool = False):
        exponents = []
        x, z = self.get_x(symplectic_matrix, nqubits, include_scratch), self.get_z(
            symplectic_matrix, nqubits, include_scratch
        )
        for j in range(nqubits):
            x1, x2 = x[[i, h], [j, j]]
            z1, z2 = z[[i, h], [j, j]]
            exponents.append(CliffordOperations.exponent(x1, z1, x2, z2))
        r = (
            0
            if (
                2 * symplectic_matrix[h, -1]
                + 2 * symplectic_matrix[i, -1]
                + self.np.sum(exponents)
            )
            % 4
            == 0
            else 1
        )
        symplectic_matrix[h, -1] = r
        symplectic_matrix[h, :nqubits] = x[i, :] ^ x[h, :]
        symplectic_matrix[h, nqubits:-1] = z[i, :] ^ z[h, :]
        return symplectic_matrix


class CliffordBackend(NumpyBackend):
    def __init__(self):
        super().__init__()

        import numpy as np

        self.name = "clifford"
        self.clifford_operations = CliffordOperations()
        self.np = np

    def zero_state(self, nqubits):
        """Construct the zero state |00...00>.

        Args:
            nqubits (int): Number of qubits.

        Returns:
            symplectic_matrix (np.ndarray): The symplectic_matrix for the zero state.
        """
        I = self.np.eye(nqubits)
        symplectic_matrix = self.np.zeros(
            (2 * nqubits + 1, 2 * nqubits + 1), dtype=bool
        )
        symplectic_matrix[:nqubits, :nqubits] = I.copy()
        symplectic_matrix[nqubits:-1, nqubits : 2 * nqubits] = I.copy()
        return symplectic_matrix

    def clifford_operation(self, gate):
        """Retrieves the symplectic_matrix operation corresponding to a gate.

        Args:
            gate (qibo.gates.abstract.gate): Input gate.

        Returns:
            operation (method): The corrsponding Clifford operation.
        """
        name = gate.__class__.__name__
        return getattr(self.clifford_operations, name)

    def apply_gate_clifford(self, gate, symplectic_matrix, nqubits, nshots):
        operation = gate.clifford_operation()
        kwargs = (
            {"theta": gate.init_kwargs["theta"]} if "theta" in gate.init_kwargs else {}
        )
        return operation(symplectic_matrix, *gate.init_args, nqubits, **kwargs)

    def execute_circuit(self, circuit, initial_state=None, nshots=1000):
        """Execute a Clifford circuits.

        Args:
            circuit (qibo.models.Circuit): Input circuit.
            initial_state (np.ndarray): The symplectic_matrix of the initial state.
            nshots (int): Number of shots.

        Returns:
            result (qibo.quantum_info.Clifford): The result object giving access to the final results.
        """
        for gate in circuit.queue:
            if not gate.clifford and not gate.__class__.__name__ == "M":
                raise_error(RuntimeError, "Circuit contains non-Clifford gates.")

        if circuit.repeated_execution and not nshots == 1:
            return self.execute_circuit_repeated(circuit, initial_state, nshots)

        try:
            nqubits = circuit.nqubits

            if initial_state is None:
                state = self.zero_state(nqubits)
            else:
                state = initial_state

            for gate in circuit.queue:
                state = gate.apply_clifford(self, state, nqubits)

            from qibo.quantum_info.clifford import Clifford

            return Clifford(state, measurements=circuit.measurements, nshots=nshots)

        except self.oom_error:  # pragma: no cover
            raise_error(
                RuntimeError,
                f"State does not fit in {self.device} memory."
                "Please switch the execution device to a "
                "different one using ``qibo.set_device``.",
            )

    def execute_circuit_repeated(self, circuit, initial_state=None, nshots=1000):
        """Execute a Clifford circuits ``nshots`` times. This is used for all the simulations that involve repeated execution, for instance when collapsing measurement or noise channels are present.

        Args:
            circuit (qibo.model.Circuit): The input Circuit.
            initial_state (np.ndarray): The symplectic_matrix of the initial state.
            nshots (int): Number of times to repeat the execution.

        Returns:
            result (qibo.quantum_info.Clifford): The result object giving access to the final results.
        """
        circuit_copy = circuit.copy()
        samples = []
        states = []
        for i in range(nshots):
            res = self.execute_circuit(circuit_copy, initial_state, nshots=1)
            [m.result.reset() for m in circuit_copy.measurements]
            states.append(res.state())
            samples.append(res.samples())
        samples = self.np.vstack(samples)

        from qibo.quantum_info.clifford import Clifford

        result = Clifford(
            self.zero_state(circuit.nqubits), circuit_copy.measurements, nshots=nshots
        )
        result.symplectic_matrix, result._samples = None, None
        for m in result.measurements:
            m.result.register_samples(samples[:, m.target_qubits], self)

        return result

    def sample_shots(self, state, qubits, nqubits, nshots, collapse: bool = False):
        """Sample shots by measuring the selected qubits from the provided state tableu.

        Args:
            state (np.ndarray): The tableu from which to sample shots from.
            qubits: (tuple): The qubits to measure.
            nqubits (int): The total number of qubits of the state.
            nshots (int): Number of shots to sample.
            collapse (bool): If ``True`` the input state is going to be collapsed with the last shot.

        Returns:
            samples (np.ndarray): The samples shots.
        """
        qubits = qubits
        operation = CliffordOperations()
        if collapse:
            samples = [operation.M(state, qubits, nqubits) for _ in range(nshots - 1)]
            samples.append(operation.M(state, qubits, nqubits, collapse))
        else:
            samples = [operation.M(state, qubits, nqubits) for _ in range(nshots)]
        return self.np.array(samples).reshape(nshots, len(qubits))

    def symplectic_matrix_to_generators(self, symplectic_matrix, return_array=False):
        """Extract both the stabilizers and de-stabilizers generators from the input symplectic_matrix.

        Args:
            symplectic_matrix (np.ndarray): The input symplectic_matrix.
            return_array (bool): If ``True`` returns the generators as numpy arrays, otherwise they are returned as strings.

        Returns:
            (generators, phases) (list, list): Lists of the extracted generators and their corresponding phases.
        """
        bits_to_gate = {"00": "I", "01": "X", "10": "Z", "11": "Y"}

        nqubits = int((symplectic_matrix.shape[1] - 1) / 2)
        phases = (-1) ** symplectic_matrix[:-1, -1]
        tmp = 1 * symplectic_matrix[:-1, :-1]
        X, Z = tmp[:, :nqubits], tmp[:, nqubits:]
        generators = []
        for x, z in zip(X, Z):
            paulis = [bits_to_gate[f"{zz}{xx}"] for i, (xx, zz) in enumerate(zip(x, z))]
            if return_array:
                paulis = [getattr(gates, p)(0).matrix() for p in paulis]
                matrix = paulis[0]
                for p in paulis[1:]:
                    matrix = self.np.kron(matrix, p)
                generators.append(matrix)
            else:
                generators.append("".join(paulis))
        return generators, phases