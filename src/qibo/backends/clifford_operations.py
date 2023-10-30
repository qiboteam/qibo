from functools import cache


class CliffordOperations:
    """Operations performed by clifford gates on the stabilizers state tableau representation"""

    def __init__(self):
        import numpy as np

        self.np = np

    def H(self, tableau, q):
        new_tab = tableau.copy()
        nqubits = CliffordOperations.nqubits(tableau.shape[0])
        self.set_r(
            new_tab,
            self.get_r(tableau)
            ^ (self.get_x(tableau)[:, q] * self.get_z(tableau)[:, q]).flatten(),
        )
        new_tab[:, [q, nqubits + q]] = new_tab[:, [nqubits + q, q]]
        return new_tab

    def CNOT(self, tableau, control_q, target_q):
        new_tab = tableau.copy()
        nqubits = CliffordOperations.nqubits(tableau.shape[0])
        self.set_r(
            new_tab,
            self.get_r(tableau)
            ^ (
                self.get_x(tableau)[:, control_q] * self.get_z(tableau)[:, target_q]
            ).flatten()
            * (
                self.get_x(tableau)[:, target_q] ^ self.get_z(tableau)[:, control_q] ^ 1
            ).flatten(),
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
        nqubits = CliffordOperations.nqubits(tableau.shape[0])
        self.set_r(
            new_tab,
            self.get_r(tableau)
            ^ (self.get_x(tableau)[:, q] * self.get_z(tableau)[:, q]).flatten(),
        )
        new_tab[:, nqubits + q] = self.get_z(tableau)[:, q] ^ self.get_x(tableau)[:, q]
        return new_tab

    # valid for standard basis measurement only
    def M(self, tableau, qubits):
        new_tab = tableau.copy()
        nqubits = CliffordOperations.nqubits(tableau.shape[0])
        for q in qubits:
            p = (self.get_x(tableau)[nqubits:, q] == 1).nonzero()
            if len(nonzero) > 0:
                if len(nonzero) > 1:
                    p = nonzero[0]
                xq = self.get_x(tableau)[:, q]
                for i in (xq == 1 and xq != p).nonzero():
                    new_tab = self.rowsum(new_tab, i, p)
                new_tab[p - nqubits, :] = new_tab[p, :]
                outcome = self.np.random.randint(2, size=1)
                new_tab[p, -1] = outcome
                new_tab[p, nqubits + q] = 0
            else:
                # to do
                pass

    @staticmethod
    def get_r(tableau):
        return tableau[:, -1]

    @staticmethod
    def set_r(tableau, val):
        tableau[:, -1] = val

    @staticmethod
    def get_x(tableau):
        return tableau[:, : CliffordOperations.nqubits(tableau.shape[0])]

    @staticmethod
    def set_x(tableau, val):
        tableau[:, CliffordOperations.nqubits(tableau.shape[0])] = val

    @staticmethod
    def get_z(tableau):
        return tableau[:, CliffordOperations.nqubits(tableau.shape[0]) : -1]

    @staticmethod
    def set_z(tableau, val):
        tableau[:, CliffordOperations.nqubits(tableau.shape[0]) : -1] = val

    @cache
    @staticmethod
    def nqubits(shape):
        return int(shape / 2)

    @cache
    @staticmethod
    def exponent(x1, z1, x2, z2):
        if x1 == z1:
            if x1 == 0:
                return 0
            elif x1 == 1:
                return z2 - x2
        else:
            if x1 == 1:
                return z2 * (2 * x2 - 1)
            elif x1 == 0:
                return x2 * (1 - 2 * z2)

    def rowsum(self, tableau, h, i):
        nqubits = CliffordOperations.nqubits(tableau.shape[0])
        exponents = []
        for j in range(nqubits):
            x1, x2 = self.get_x(tableau)[[i, j], [h, j]]
            z1, z2 = self.get_z(tableau)[[i, j], [h, j]]
            exponents.append(ClifordOperations.exponent(x1, z1, x2, z2))
        new_tab = tableau.copy()
        if (2 * tableau[h, -1] + 2 * tableau[i, -1] + np.sum(exponents)) % 4 == 0:
            new_tab[h, -1] = 0
        else:  # could be good to check that the expression above is == 2 here...
            new_tab[h, -1] = 1
        new_tab[h, :nqubits] = self.get_x(tableau)[i, :] ^ self.get_x(tableau)[h, :]
        new_tab[h, nqubits:-1] = self.get_z(tableau)[i, :] ^ self.get_z(tableau)[h, :]
        return tableau
