from functools import cache


class CliffordOperations:
    """Operations performed by clifford gates on the stabilizers state tableau representation"""

    def __init__(self):
        import numpy as np

        self.np = np

    def H(self, tableau, q):
        new_tab = tableau.copy()
        nqubits = self.nqubits(tableau.shape[0])
        self.set_r(
            new_tab,
            self.get_r(tableau)
            ^ (self.get_x(tableau)[:, q] * self.get_z(tableau)[:, q]),
        )
        new_tab[:, [q, nqubits + q]] = new_tab[:, [nqubits + q, q]]
        return new_tab

    def CNOT(self, tableau, control_q, target_q):
        new_tab = tableau.copy()
        nqubits = self.nqubits(tableau.shape[0])
        self.set_r(
            new_tab,
            self.get_r(tableau)
            ^ (self.get_x(tableau)[:, control_q] * self.get_z(tableau)[:, target_q])
            * (
                self.get_x(tableau)[:, target_q] ^ self.get_z(tableau)[:, control_q] ^ 1
            ),
        )

        new_tab[:, target_q] = (
            self.get_x(tableau)[:, target_q] ^ self.get_x(tableau)[:, control_q]
        )
        new_tab[:, nqubits + control_q] = (
            self.get_z(tableau)[:, control_q] ^ self.get_z(tableau)[:, control_q]
        )
        return new_tab

    def S(self, tableau, q):
        new_tab = tableau.copy()
        nqubits = self.nqubits(tableau.shape[0])
        self.set_r(
            new_tab,
            self.get_r(tableau)
            ^ (self.get_x(tableau)[:, q] * self.get_z(tableau)[:, q]),
        )
        new_tab[:, nqubits + q] = self.get_z(tableau)[:, q] ^ self.get_x(tableau)[:, q]
        return new_tab

    def M(self, tableau, qubits):
        for q in qubits:
            nonzero = (self.get_x(tableau)[:, q] == 1).nonzero()
            if len(nonzero) > 0:
                if len(nonzero) > 1:
                    nonzero = nonzero[0]

    @staticmethod
    def get_r(tableau):
        return tableau[:, -1]

    @staticmethod
    def set_r(tableau, val):
        tableau[:, -1] = val

    def get_x(self, tableau):
        return tableau[:, : self.nqubits(tableau.shape[0])]

    def set_x(self, tableau, val):
        tableau[:, self.nqubits(tableau.shape[0])] = val

    def get_z(self, tableau):
        return tableau[:, self.nqubits(tableau.shape[0]) : -1]

    def set_z(self, tableau, val):
        tableau[:, self.nqubits(tableau.shape[0]) : -1] = val

    @cache
    @staticmethod
    def nqubits(shape):
        return int(shape / 2)
