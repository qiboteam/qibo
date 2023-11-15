"""Module defining the Clifford backend."""
from functools import cache

from qibo import gates
from qibo.backends.numpy import NumpyBackend
from qibo.config import raise_error
from qibo.result import CircuitResult


class CliffordOperations:
    """Operations performed by clifford gates on the stabilizers state tableau representation"""

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
        """Decomposition --> HCNOTH"""
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
        """Decomposition --> SS"""
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
        """Decomposition --> HSSH"""
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
        """Decomposition --> SSHSSH"""  # double check this, cause it should be
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
        """Decomposition --> HSH"""
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
        """Decomposition --> SSS"""
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
        """Decomposition --> HSSSH"""
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
        # not working
        elif (theta / (self.np.pi / 2) - 1) % 4 == 0:
            """Decomposition --> HSS"""
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
            """Decomposition --> HSSHSSHSS"""
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
        """Decomposition --> CNOTCNOTCNOT"""
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
        """Decomposition --> HCNOTCNOTHSS"""
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
        new_tab = self.X(tableau, target_q, nqubits)
        new_tab = self.CNOT(new_tab, control_q, target_q, nqubits)
        new_tab = self.RY(new_tab, control_q, nqubits, self.np.pi / 2)
        new_tab = self.CNOT(new_tab, target_q, control_q, nqubits)
        new_tab = self.RY(new_tab, control_q, nqubits, -self.np.pi / 2)
        new_tab = self.CNOT(new_tab, target_q, control_q, nqubits)
        new_tab = self.CNOT(new_tab, control_q, target_q, nqubits)
        return self.X(new_tab, control_q, nqubits)

    def CY(self, tableau, control_q, target_q, nqubits):
        """Decomposition --> HCNOTCNOTHSS"""
        new_tab = tableau.copy()
        r, x, z = (
            self.get_r(new_tab, nqubits),
            self.get_x(new_tab, nqubits),
            self.get_z(new_tab, nqubits),
        )
        self.set_r(
            new_tab,
            r
            ^ x[:, target_q]
            ^ z[:, target_q]
            ^ (
                x[:, control_q]
                * x[:, target_q]
                * (z[:, control_q] ^ z[:, target_q] ^ 1)
            ).flatten()
            ^ (x[:, target_q] * (x[:, control_q] ^ z[:, target_q])).flatten()
            ^ (
                x[:, control_q]
                * (x[:, control_q] ^ z[:, target_q])
                * (z[:, control_q] ^ x[:, target_q] ^ 1)
            ).flatten(),
        )
        new_tab[:-1, target_q] = x[:, control_q] ^ x[:, target_q]
        new_tab[:-1, [nqubits + control_q, nqubit + target_q]] = np.vstack(
            (
                z[:, control_q] ^ x[:, control_q] ^ z[:, target_q],
                z[:, target_q] ^ x[:, control_q],
            )
        ).T
        return new_tab

    def RZX(self, tableau, control_q, target_q, nqubits, theta):
        """Decomposition --> H-CNOT-RZ-CNOT-H"""
        new_tab = tableau.copy()
        new_tab = self.H(new_tab, target_q, nqubits)
        new_tab = self.CNOT(new_tab, control_q, target_q, nqubits)
        new_tab = self.RZ(new_tab, target_q, nqubits, theta)
        new_tab = self.CNOT(new_tab, control_q, target_q, nqubits)
        return self.H(new_tab, target_q, nqubits)

    def CRX(self, tableau, control_q, target_q, nqubits, theta):
        if theta % (2 * self.np.pi) == 0:
            return self.I(tableau, target_q, nqubits)
        elif (theta / self.np.pi - 1) % 2 == 0:
            return self.CNOT(tableau, control_q, target_q, nqubits)

    def CRZ(self, tableau, control_q, target_q, nqubits, theta):
        if theta % (2 * self.np.pi) == 0:
            return self.I(tableau, target_q, nqubits)
        elif (theta / self.np.pi - 1) % 2 == 0:
            return self.CZ(tableau, control_q, target_q, nqubits)

    def CRY(self, tableau, control_q, target_q, nqubits, theta):
        if theta % (2 * self.np.pi) == 0:
            return self.I(tableau, target_q, nqubits)
        elif (theta / self.np.pi - 1) % 2 == 0:
            return self.CY(tableau, control_q, target_q, nqubits)

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
                for i in (x[:, q] == 1).nonzero()[0]:
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

    @cache
    @staticmethod
    def nqubits(shape):
        return int((shape - 1) / 2)

    @cache
    @staticmethod
    def exponent(x1, z1, x2, z2):
        if x1 == z1:
            if x1 == 0:
                return 0
            return z2 ^ x2
        if x1 == 1:
            return z2 * (2 * x2 ^ 1)
        return x2 * (1 ^ 2 * z2)

    def rowsum(self, tableau, h, i, nqubits, include_scratch: bool = False):
        exponents = []
        new_tab = tableau.copy()
        x, z = self.get_x(new_tab, nqubits, include_scratch), self.get_z(
            new_tab, nqubits, include_scratch
        )
        for j in range(nqubits):
            x1, x2 = x[[i, h], [j, j]]
            z1, z2 = z[[i, h], [j, j]]
            exponents.append(CliffordOperations.exponent(x1, z1, x2, z2))
        if (2 * new_tab[h, -1] + 2 * new_tab[i, -1] + self.np.sum(exponents)) % 4 == 0:
            new_tab[h, -1] = 0
        else:  # could be good to check that the expression above is == 2 here...
            new_tab[h, -1] = 1
        new_tab[h, :nqubits] = x[i, :] ^ x[h, :]
        new_tab[h, nqubits:-1] = z[i, :] ^ z[h, :]
        return new_tab


class CliffordBackend(NumpyBackend):
    def __init__(self):
        super().__init__()

        import numpy as np

        self.name = "clifford"
        self.clifford_operations = CliffordOperations()
        self.np = np

    def zero_state(self, nqubits):
        I = self.np.eye(nqubits)
        tableau = self.np.zeros((2 * nqubits + 1, 2 * nqubits + 1), dtype=bool)
        tableau[:nqubits, :nqubits] = I.copy()
        tableau[nqubits:-1, nqubits : 2 * nqubits] = I.copy()
        return tableau

    def clifford_operation(self, gate):
        name = gate.__class__.__name__
        return getattr(self.clifford_operations, name)

    def apply_gate_clifford(self, gate, tableau, nqubits, nshots):
        operation = gate.clifford_operation(self)

        if len(gate.init_args) > 0:
            if "theta" in gate.init_kwargs:
                return operation(
                    tableau,
                    self.np.array(gate.qubits),
                    nqubits,
                    gate.init_kwargs["theta"],
                )
            return operation(
                tableau,
                *gate.init_args,
                nqubits,
            )

        return operation(tableau, self.np.array(gate.qubits), nqubits)

    def execute_circuit(self, circuit, initial_state=None, nshots=1000):
        for gate in circuit.queue:
            if not gate.clifford and not gate.__class__.__name__ == "M":
                raise_error(RuntimeError, "Circuit contains non-Clifford gates.")

        try:
            nqubits = circuit.nqubits

            if initial_state is None:
                state = self.zero_state(nqubits)
            else:
                state = initial_state

            for gate in circuit.queue:
                state = gate.apply_clifford(self, state, nqubits, nshots)

            from qibo.quantum_info.clifford import Clifford

            return Clifford(state, measurements=circuit.measurements)

        except self.oom_error:
            raise_error(
                RuntimeError,
                f"State does not fit in {self.device} memory."
                "Please switch the execution device to a "
                "different one using ``qibo.set_device``.",
            )

    def sample_shots(self, state, qubits, nqubits, nshots, collapse: bool = False):
        operation = CliffordOperations()
        if collapse:
            samples = [operation.M(state, qubits, nqubits) for _ in range(nshots - 1)]
            samples.append(operation.M(state, qubits, nqubits, collapse))
        else:
            samples = [operation.M(state, qubits, nqubits) for _ in range(nshots)]
        return self.np.array(samples).reshape(nshots, len(qubits))

    def tableau_to_generators(self, tableau, return_array=False):
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
