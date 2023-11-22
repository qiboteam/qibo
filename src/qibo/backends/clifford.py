"""Module defining the Clifford backend."""
from functools import cache

from qibo import gates
from qibo.backends.numpy import NumpyBackend
from qibo.config import raise_error
from qibo.result import CircuitResult


class CliffordOperations:
    """Operations performed by clifford gates on the stabilizers state tableau representation discussed in https://arxiv.org/abs/quant-ph/0406196."""

    def __init__(self):
        import numpy as np

        self.np = np

    def I(self, tableau, q, nqubits):
        return tableau

    def H(self, tableau, q, nqubits):
        new_tab = tableau.copy()
        self.set_r(
            new_tab,
            self.get_r(new_tab, nqubits)
            ^ (
                self.get_x(new_tab, nqubits)[:, q] * self.get_z(new_tab, nqubits)[:, q]
            ).flatten(),
        )
        new_tab[:, [q, nqubits + q]] = new_tab[:, [nqubits + q, q]]
        return new_tab

    def CNOT(self, tableau, control_q, target_q, nqubits):
        new_tab = tableau.copy()
        x, z = self.get_x(new_tab, nqubits), self.get_z(new_tab, nqubits)
        self.set_r(
            new_tab,
            self.get_r(new_tab, nqubits)
            ^ (x[:, control_q] * z[:, target_q]).flatten()
            * (x[:, target_q] ^ z[:, control_q] ^ 1).flatten(),
        )

        new_tab[:-1, target_q] = x[:, target_q] ^ x[:, control_q]
        new_tab[:-1, nqubits + control_q] = z[:, control_q] ^ z[:, target_q]
        return new_tab

    def CZ(self, tableau, control_q, target_q, nqubits):
        """Decomposition --> H-CNOT-H"""
        new_tab = tableau.copy()
        r, x, z = (
            self.get_r(new_tab, nqubits),
            self.get_x(new_tab, nqubits),
            self.get_z(new_tab, nqubits),
        )
        self.set_r(
            new_tab,
            r
            ^ (x[:, target_q] * z[:, target_q]).flatten()
            ^ (
                x[:, control_q]
                * x[:, target_q]
                * (z[:, target_q] ^ z[:, control_q] ^ 1)
            ).flatten()
            ^ (x[:, target_q] * (z[:, target_q] ^ x[:, control_q])).flatten(),
        )
        new_tab[:-1, [nqubits + control_q, nqubits + target_q]] = self.np.vstack(
            (x[:, target_q] ^ z[:, control_q], z[:, target_q] ^ x[:, control_q])
        ).T
        return new_tab

    def S(self, tableau, q, nqubits):
        new_tab = tableau.copy()
        x, z = self.get_x(new_tab, nqubits), self.get_z(new_tab, nqubits)
        self.set_r(
            new_tab,
            self.get_r(new_tab, nqubits) ^ (x[:, q] * z[:, q]).flatten(),
        )
        new_tab[:-1, nqubits + q] = z[:, q] ^ x[:, q]
        return new_tab

    def Z(self, tableau, q, nqubits):
        """Decomposition --> S-S"""
        new_tab = tableau.copy()
        r, x, z = (
            self.get_r(new_tab, nqubits),
            self.get_x(new_tab, nqubits),
            self.get_z(new_tab, nqubits),
        )
        self.set_r(
            new_tab, r ^ ((x[:, q] * z[:, q]) ^ x[:, q] * (z[:, q] ^ x[:, q])).flatten()
        )
        return new_tab

    def X(self, tableau, q, nqubits):
        """Decomposition --> H-S-S-H"""
        new_tab = tableau.copy()
        r, x, z = (
            self.get_r(new_tab, nqubits),
            self.get_x(new_tab, nqubits),
            self.get_z(new_tab, nqubits),
        )
        self.set_r(
            new_tab,
            r
            ^ (z[:, q] * (z[:, q] ^ x[:, q])).flatten()
            ^ (z[:, q] * x[:, q]).flatten(),
        )
        return new_tab

    def Y(self, tableau, q, nqubits):
        """Decomposition --> S-S-H-S-S-H"""  # double check this, cause it should be
        new_tab = tableau.copy()  # Y = i * HZHZ --> HSSHSS
        r, x, z = (
            self.get_r(new_tab, nqubits),
            self.get_x(new_tab, nqubits),
            self.get_z(new_tab, nqubits),
        )
        self.set_r(
            new_tab,
            r
            ^ (z[:, q] * (z[:, q] ^ x[:, q])).flatten()
            ^ (x[:, q] * (z[:, q] ^ x[:, q])).flatten(),
        )
        return new_tab

    def SX(self, tableau, q, nqubits):
        """Decomposition --> H-S-H"""
        new_tab = tableau.copy()
        r, x, z = (
            self.get_r(new_tab, nqubits),
            self.get_x(new_tab, nqubits),
            self.get_z(new_tab, nqubits),
        )
        self.set_r(
            new_tab,
            r ^ (z[:, q] * (z[:, q] ^ x[:, q])).flatten(),
        )
        new_tab[:-1, q] = z[:, q] ^ x[:, q]
        return new_tab

    def SDG(self, tableau, q, nqubits):
        """Decomposition --> S-S-S"""
        new_tab = tableau.copy()
        r, x, z = (
            self.get_r(new_tab, nqubits),
            self.get_x(new_tab, nqubits),
            self.get_z(new_tab, nqubits),
        )
        self.set_r(
            new_tab,
            r ^ (x[:, q] * (z[:, q] ^ x[:, q])).flatten(),
        )
        new_tab[:-1, nqubits + q] = z[:, q] ^ x[:, q]
        return new_tab

    def SXDG(self, tableau, q, nqubits):
        """Decomposition --> H-S-S-S-H"""
        new_tab = tableau.copy()
        r, x, z = (
            self.get_r(new_tab, nqubits),
            self.get_x(new_tab, nqubits),
            self.get_z(new_tab, nqubits),
        )
        self.set_r(
            new_tab,
            r ^ (z[:, q] * x[:, q]).flatten(),
        )
        new_tab[:-1, q] = z[:, q] ^ x[:, q]
        return new_tab

    def RX(self, tableau, q, nqubits, theta):
        if theta % (2 * self.np.pi) == 0:
            return self.I(tableau, q, nqubits)
        elif (theta / self.np.pi - 1) % 2 == 0:
            return self.X(tableau, q, nqubits)
        elif (theta / (self.np.pi / 2) - 1) % 4 == 0:
            return self.SX(tableau, q, nqubits)
        else:  # theta == 3*pi/2 + 2*n*pi
            return self.SXDG(tableau, q, nqubits)

    def RZ(self, tableau, q, nqubits, theta):
        if theta % (2 * self.np.pi) == 0:
            return self.I(tableau, q, nqubits)
        elif (theta / self.np.pi - 1) % 2 == 0:
            return self.Z(tableau, q, nqubits)
        elif (theta / (self.np.pi / 2) - 1) % 4 == 0:
            return self.S(tableau, q, nqubits)
        else:  # theta == 3*pi/2 + 2*n*pi
            return self.SDG(tableau, q, nqubits)

    def RY(self, tableau, q, nqubits, theta):
        if theta % (2 * self.np.pi) == 0:
            return self.I(tableau, q, nqubits)
        elif (theta / self.np.pi - 1) % 2 == 0:
            return self.Y(tableau, q, nqubits)
        elif (theta / (self.np.pi / 2) - 1) % 4 == 0:
            """Decomposition --> H-S-S"""
            new_tab = tableau.copy()
            r, x, z = (
                self.get_r(new_tab, nqubits),
                self.get_x(new_tab, nqubits),
                self.get_z(new_tab, nqubits),
            )
            self.set_r(
                new_tab,
                r ^ (x[:, q] * (z[:, q] ^ x[:, q])).flatten(),
            )
            new_tab[:-1, [nqubits + q, q]] = new_tab[:-1, [q, nqubits + q]]
            return new_tab
        else:  # theta == 3*pi/2 + 2*n*pi
            """Decomposition --> H-S-S-H-S-S-H-S-S"""
            new_tab = tableau.copy()
            r, x, z = (
                self.get_r(new_tab, nqubits),
                self.get_x(new_tab, nqubits),
                self.get_z(new_tab, nqubits),
            )
            self.set_r(
                new_tab,
                r ^ (z[:, q] * (z[:, q] ^ x[:, q])).flatten(),
            )
            new_tab[:-1, [nqubits + q, q]] = new_tab[:-1, [q, nqubits + q]]
            return new_tab

    def SWAP(self, tableau, control_q, target_q, nqubits):
        """Decomposition --> CNOT-CNOT-CNOT"""
        new_tab = tableau.copy()
        r, x, z = (
            self.get_r(new_tab, nqubits),
            self.get_x(new_tab, nqubits),
            self.get_z(new_tab, nqubits),
        )
        self.set_r(
            new_tab,
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
        new_tab[
            :-1, [control_q, target_q, nqubits + control_q, nqubits + target_q]
        ] = new_tab[:-1, [target_q, control_q, nqubits + target_q, nqubits + control_q]]
        return new_tab

    def iSWAP(self, tableau, control_q, target_q, nqubits):
        """Decomposition --> H-CNOT-CNOT-H-S-S"""
        new_tab = tableau.copy()
        r, x, z = (
            self.get_r(new_tab, nqubits),
            self.get_x(new_tab, nqubits),
            self.get_z(new_tab, nqubits),
        )
        self.set_r(
            new_tab,
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
        new_tab[:-1, [nqubits + control_q, nqubits + target_q]] = self.np.vstack(
            (
                x[:, target_q] ^ z[:, target_q] ^ x[:, control_q],
                x[:, target_q] ^ z[:, control_q] ^ x[:, control_q],
            )
        ).T
        new_tab[:-1, [control_q, target_q]] = new_tab[:-1, [target_q, control_q]]
        return new_tab

    def FSWAP(self, tableau, control_q, target_q, nqubits):
        """Decomposition --> X-CNOT-RY-CNOT-RY-CNOT-CNOT-X"""
        new_tab = self.X(tableau, target_q, nqubits)
        new_tab = self.CNOT(new_tab, control_q, target_q, nqubits)
        new_tab = self.RY(new_tab, control_q, nqubits, self.np.pi / 2)
        new_tab = self.CNOT(new_tab, target_q, control_q, nqubits)
        new_tab = self.RY(new_tab, control_q, nqubits, -self.np.pi / 2)
        new_tab = self.CNOT(new_tab, target_q, control_q, nqubits)
        new_tab = self.CNOT(new_tab, control_q, target_q, nqubits)
        return self.X(new_tab, control_q, nqubits)

    def CY(self, tableau, control_q, target_q, nqubits):
        """Decomposition --> S-CNOT-SDG"""
        new_tab = tableau.copy()
        r, x, z = (
            self.get_r(new_tab, nqubits),
            self.get_x(new_tab, nqubits),
            self.get_z(new_tab, nqubits),
        )
        self.set_r(
            new_tab,
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

        new_tab[
            :-1, [target_q, nqubits + control_q, nqubits + target_q]
        ] = self.np.vstack(
            (
                x[:, control_q] ^ x[:, target_q],
                z[:, control_q] ^ z[:, target_q] ^ x[:, target_q],
                z[:, target_q] ^ x[:, control_q],
            )
        ).T
        return new_tab

    # this is not actually Clifford...
    def RZX(self, tableau, control_q, target_q, nqubits, theta):
        """Decomposition --> H-CNOT-RZ-CNOT-H"""
        new_tab = self.H(tableau, target_q, nqubits)
        new_tab = self.CNOT(new_tab, control_q, target_q, nqubits)
        new_tab = self.RZ(new_tab, target_q, nqubits, theta)
        new_tab = self.CNOT(new_tab, control_q, target_q, nqubits)
        return self.H(new_tab, target_q, nqubits)

    def CRX(self, tableau, control_q, target_q, nqubits, theta):
        # theta = 4 * n * pi
        if theta % (4 * self.np.pi) == 0:
            return self.I(tableau, target_q, nqubits)
        # theta = pi + 4 * n * pi
        elif (theta / self.np.pi - 1) % 4 == 0:
            new_tab = self.X(tableau, target_q, nqubits)
            new_tab = self.CZ(new_tab, control_q, target_q, nqubits)
            new_tab = self.X(new_tab, target_q, nqubits)
            return self.CY(new_tab, control_q, target_q, nqubits)
        # theta = 2 * pi + 4 * n * pi
        elif (theta / (2 * self.np.pi) - 1) % 2 == 0:
            new_tab = self.CZ(tableau, control_q, target_q, nqubits)
            new_tab = self.Y(new_tab, target_q, nqubits)
            new_tab = self.CZ(new_tab, control_q, target_q, nqubits)
            return self.Y(new_tab, target_q, nqubits)
        # theta = 3 * pi + 4 * n * pi
        elif (theta / self.np.pi - 3) % 4 == 0:
            new_tab = self.X(tableau, target_q, nqubits)
            new_tab = self.CY(new_tab, control_q, target_q, nqubits)
            new_tab = self.X(new_tab, target_q, nqubits)
            return self.CZ(new_tab, control_q, target_q, nqubits)

    def CRZ(self, tableau, control_q, target_q, nqubits, theta):
        # theta = 4 * n * pi
        if theta % (4 * self.np.pi) == 0:
            return self.I(tableau, target_q, nqubits)
        # theta = pi + 4 * n * pi
        elif (theta / self.np.pi - 1) % 4 == 0:
            new_tab = self.X(tableau, target_q, nqubits)
            new_tab = self.CY(new_tab, control_q, target_q, nqubits)
            new_tab = self.X(new_tab, target_q, nqubits)
            return self.CNOT(new_tab, control_q, target_q, nqubits)
        # theta = 2 * pi + 4 * n * pi
        elif (theta / (2 * self.np.pi) - 1) % 2 == 0:
            new_tab = self.CZ(tableau, control_q, target_q, nqubits)
            new_tab = self.X(new_tab, target_q, nqubits)
            new_tab = self.CZ(new_tab, control_q, target_q, nqubits)
            return self.X(new_tab, target_q, nqubits)
        # theta = 3 * pi + 4 * n * pi
        elif (theta / self.np.pi - 3) % 4 == 0:
            new_tab = self.CNOT(tableau, control_q, target_q, nqubits)
            new_tab = self.X(new_tab, target_q, nqubits)
            new_tab = self.CY(new_tab, control_q, target_q, nqubits)
            return self.X(new_tab, target_q, nqubits)

    def CRY(self, tableau, control_q, target_q, nqubits, theta):
        # theta = 4 * n * pi
        if theta % (4 * self.np.pi) == 0:
            return self.I(tableau, target_q, nqubits)
        # theta = pi + 4 * n * pi
        elif (theta / self.np.pi - 1) % 4 == 0:
            new_tab = self.Z(tableau, target_q, nqubits)
            new_tab = self.CNOT(new_tab, control_q, target_q, nqubits)
            new_tab = self.Z(new_tab, target_q, nqubits)
            return self.CZ(new_tab, control_q, target_q, nqubits)
        # theta = 2 * pi + 4 * n * pi
        elif (theta / (2 * self.np.pi) - 1) % 2 == 0:
            return self.CRZ(tableau, control_q, target_q, nqubits, theta)
        # theta = 3 * pi + 4 * n * pi
        elif (theta / self.np.pi - 3) % 4 == 0:
            new_tab = self.CZ(tableau, control_q, target_q, nqubits)
            new_tab = self.Z(new_tab, target_q, nqubits)
            new_tab = self.CNOT(new_tab, control_q, target_q, nqubits)
            return self.Z(new_tab, target_q, nqubits)

    def ECR(self, tableau, control_q, target_q, nqubits):
        new_tab = self.S(tableau, control_q, nqubits)
        new_tab = self.SX(new_tab, target_q, nqubits)
        new_tab = self.CNOT(new_tab, control_q, target_q, nqubits)
        return self.X(new_tab, control_q, nqubits)

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
                outcome = int(self.np.random.randint(2, size=1))
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
    def get_r(tableau, nqubits, include_scratch=False):
        return tableau[
            : -1 + (2 * nqubits + 2) * int(include_scratch),
            -1,
        ]

    @staticmethod
    def set_r(tableau, val):
        tableau[:-1, -1] = val

    @staticmethod
    def get_x(tableau, nqubits, include_scratch=False):
        return tableau[: -1 + (2 * nqubits + 2) * int(include_scratch), :nqubits]

    @staticmethod
    def set_x(tableau, nqubits, val):
        tableau[:-1, nqubits] = val

    @staticmethod
    def get_z(tableau, nqubits, include_scratch=False):
        return tableau[: -1 + (2 * nqubits + 2) * int(include_scratch), nqubits:-1]

    @staticmethod
    def set_z(tableau, nqubits, val):
        tableau[:-1, nqubits:-1] = val

    @staticmethod
    def get_scratch(tableau):
        return tableau[-1, :]

    @staticmethod
    def set_scratch(tableau, val):
        tableau[-1, :] = val

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

    def rowsum(self, tableau, h, i, nqubits, include_scratch: bool = False):
        exponents = []
        x, z = self.get_x(tableau, nqubits, include_scratch), self.get_z(
            tableau, nqubits, include_scratch
        )
        for j in range(nqubits):
            x1, x2 = x[[i, h], [j, j]]
            z1, z2 = z[[i, h], [j, j]]
            exponents.append(CliffordOperations.exponent(x1, z1, x2, z2))
        r = (
            0
            if (2 * tableau[h, -1] + 2 * tableau[i, -1] + self.np.sum(exponents)) % 4
            == 0
            else 1
        )
        tableau[h, -1] = r
        tableau[h, :nqubits] = x[i, :] ^ x[h, :]
        tableau[h, nqubits:-1] = z[i, :] ^ z[h, :]
        return tableau


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
            tableau (np.ndarray): The tableau for the zero state.
        """
        I = self.np.eye(nqubits)
        tableau = self.np.zeros((2 * nqubits + 1, 2 * nqubits + 1), dtype=bool)
        tableau[:nqubits, :nqubits] = I.copy()
        tableau[nqubits:-1, nqubits : 2 * nqubits] = I.copy()
        return tableau

    def clifford_operation(self, gate):
        """Retrieves the tableau operation corresponding to a gate.

        Args:
            gate (qibo.gates.abstract.gate): Input gate.

        Returns:
            operation (method): The corrsponding Clifford operation.
        """
        name = gate.__class__.__name__
        return getattr(self.clifford_operations, name)

    def apply_gate_clifford(self, gate, tableau, nqubits, nshots):
        operation = gate.clifford_operation(self)
        kwargs = (
            {"theta": gate.init_kwargs["theta"]} if "theta" in gate.init_kwargs else {}
        )
        return operation(tableau, *gate.init_args, nqubits, **kwargs)

    def execute_circuit(self, circuit, initial_state=None, nshots=1000):
        """Execute a clifford circuits.

        Args:
            circuit (qibo.models.Circuit): Input circuit.
            initial_state (np.ndarray): The tableau of the initial state.
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

        except self.oom_error:
            raise_error(
                RuntimeError,
                f"State does not fit in {self.device} memory."
                "Please switch the execution device to a "
                "different one using ``qibo.set_device``.",
            )

    def execute_circuit_repeated(self, circuit, initial_state=None, nshots=1000):
        """Execute a clifford circuits ``nshots`` times. This is used for all the simulations that involve repeated execution, for instance when collapsing measurement or noise channels are present.

        Args:
            circuit (qibo.model.Circuit): The input Circuit.
            initial_state (np.ndarray): The tableau of the initial state.
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
        result.tableau, result._samples = None, None
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
        operation = CliffordOperations()
        if collapse:
            samples = [operation.M(state, qubits, nqubits) for _ in range(nshots - 1)]
            samples.append(operation.M(state, qubits, nqubits, collapse))
        else:
            samples = [operation.M(state, qubits, nqubits) for _ in range(nshots)]
        return self.np.array(samples).reshape(nshots, len(qubits))

    def tableau_to_generators(self, tableau, return_array=False):
        """Extract both the stabilizers and de-stabilizers generators from the input tableau.

        Args:
            tableau (np.ndarray): The input tableau.
            return_array (bool): If ``True`` returns the generators as numpy arrays, otherwise they are returned as strings.

        Returns:
            (generators, phases) (list, list): Lists of the extracted generators and their corresponding phases.
        """
        bits_to_gate = {"00": "I", "01": "X", "10": "Z", "11": "Y"}

        nqubits = int((tableau.shape[1] - 1) / 2)
        phases = (-1) ** tableau[:-1, -1]
        tmp = 1 * tableau[:-1, :-1]
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
