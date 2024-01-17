import numpy as np


def _get_rxz(symplectic_matrix, nqubits):
    return (
        symplectic_matrix[:-1, -1],
        symplectic_matrix[:-1, :nqubits],
        symplectic_matrix[:-1, nqubits:-1],
    )


def I(symplectic_matrix, q, nqubits):
    return symplectic_matrix


def H(symplectic_matrix, q, nqubits):
    r, x, z = _get_rxz(symplectic_matrix, nqubits)
    symplectic_matrix[:-1, -1] = r ^ (x[:, q] & z[:, q])
    symplectic_matrix[:, [q, nqubits + q]] = symplectic_matrix[:, [nqubits + q, q]]
    return symplectic_matrix


def CNOT(symplectic_matrix, control_q, target_q, nqubits):
    ind_zt = nqubits + target_q
    ind_zc = nqubits + control_q
    r = symplectic_matrix[:-1, -1]
    xcq = symplectic_matrix[:-1, control_q]
    xtq = symplectic_matrix[:-1, target_q]
    ztq = symplectic_matrix[:-1, ind_zt]
    zcq = symplectic_matrix[:-1, ind_zc]
    symplectic_matrix[:-1, -1] = r ^ (xcq & ztq) & (xtq ^ ~zcq)
    symplectic_matrix[:-1, target_q] = xtq ^ xcq
    symplectic_matrix[:-1, ind_zc] = zcq ^ ztq
    return symplectic_matrix


def CZ(symplectic_matrix, control_q, target_q, nqubits):
    """Decomposition --> H-CNOT-H"""
    ind_zt = nqubits + target_q
    ind_zc = nqubits + control_q
    r = symplectic_matrix[:-1, -1]
    xcq = symplectic_matrix[:-1, control_q]
    xtq = symplectic_matrix[:-1, target_q]
    ztq = symplectic_matrix[:-1, ind_zt]
    zcq = symplectic_matrix[:-1, ind_zc]
    ztq_xor_xcq = ztq ^ xcq
    symplectic_matrix[:-1, -1] = (
        r ^ (xtq & ztq) ^ (xcq & xtq & (ztq ^ ~zcq)) ^ (xtq & ztq_xor_xcq)
    )
    z_control_q = xtq ^ zcq
    z_target_q = ztq_xor_xcq
    symplectic_matrix[:-1, ind_zc] = z_control_q
    symplectic_matrix[:-1, ind_zt] = z_target_q
    return symplectic_matrix


def S(symplectic_matrix, q, nqubits):
    r, x, z = _get_rxz(symplectic_matrix, nqubits)
    symplectic_matrix[:-1, -1] = r ^ (x[:, q] & z[:, q])
    symplectic_matrix[:-1, nqubits + q] = z[:, q] ^ x[:, q]
    return symplectic_matrix


def Z(symplectic_matrix, q, nqubits):
    """Decomposition --> S-S"""
    r, x, z = _get_rxz(symplectic_matrix, nqubits)
    symplectic_matrix[:-1, -1] = r ^ (
        (x[:, q] & z[:, q]) ^ x[:, q] & (z[:, q] ^ x[:, q])
    )
    return symplectic_matrix


def X(symplectic_matrix, q, nqubits):
    """Decomposition --> H-S-S-H"""
    r, x, z = _get_rxz(symplectic_matrix, nqubits)
    symplectic_matrix[:-1, -1] = (
        r ^ (z[:, q] & (z[:, q] ^ x[:, q])) ^ (z[:, q] & x[:, q])
    )
    return symplectic_matrix


def Y(symplectic_matrix, q, nqubits):
    """Decomposition --> S-S-H-S-S-H"""
    r, x, z = _get_rxz(symplectic_matrix, nqubits)
    symplectic_matrix[:-1, -1] = (
        r ^ (z[:, q] & (z[:, q] ^ x[:, q])) ^ (x[:, q] & (z[:, q] ^ x[:, q]))
    )
    return symplectic_matrix


def SX(symplectic_matrix, q, nqubits):
    """Decomposition --> H-S-H"""
    r, x, z = _get_rxz(symplectic_matrix, nqubits)
    symplectic_matrix[:-1, -1] = r ^ (z[:, q] & (z[:, q] ^ x[:, q]))
    symplectic_matrix[:-1, q] = z[:, q] ^ x[:, q]
    return symplectic_matrix


def SDG(symplectic_matrix, q, nqubits):
    """Decomposition --> S-S-S"""
    r, x, z = _get_rxz(symplectic_matrix, nqubits)
    symplectic_matrix[:-1, -1] = r ^ (x[:, q] & (z[:, q] ^ x[:, q]))
    symplectic_matrix[:-1, nqubits + q] = z[:, q] ^ x[:, q]
    return symplectic_matrix


def SXDG(symplectic_matrix, q, nqubits):
    """Decomposition --> H-S-S-S-H"""
    r, x, z = _get_rxz(symplectic_matrix, nqubits)
    symplectic_matrix[:-1, -1] = r ^ (z[:, q] & x[:, q])
    symplectic_matrix[:-1, q] = z[:, q] ^ x[:, q]
    return symplectic_matrix


def RX(symplectic_matrix, q, nqubits, theta):
    if theta % (2 * np.pi) == 0:
        return I(symplectic_matrix, q, nqubits)
    elif (theta / np.pi - 1) % 2 == 0:
        return X(symplectic_matrix, q, nqubits)
    elif (theta / (np.pi / 2) - 1) % 4 == 0:
        return SX(symplectic_matrix, q, nqubits)
    else:  # theta == 3*pi/2 + 2*n*pi
        return SXDG(symplectic_matrix, q, nqubits)


def RZ(symplectic_matrix, q, nqubits, theta):
    if theta % (2 * np.pi) == 0:
        return I(symplectic_matrix, q, nqubits)
    elif (theta / np.pi - 1) % 2 == 0:
        return Z(symplectic_matrix, q, nqubits)
    elif (theta / (np.pi / 2) - 1) % 4 == 0:
        return S(symplectic_matrix, q, nqubits)
    else:  # theta == 3*pi/2 + 2*n*pi
        return SDG(symplectic_matrix, q, nqubits)


def RY_pi(symplectic_matrix, q, nqubits):
    """Decomposition --> H-S-S"""
    r, x, z = _get_rxz(symplectic_matrix, nqubits)
    symplectic_matrix[:-1, -1] = r ^ (x[:, q] & (z[:, q] ^ x[:, q]))
    symplectic_matrix[:-1, [nqubits + q, q]] = symplectic_matrix[:-1, [q, nqubits + q]]
    return symplectic_matrix


def RY_3pi_2(symplectic_matrix, q, nqubits):
    """Decomposition --> H-S-S-H-S-S-H-S-S"""
    r, x, z = _get_rxz(symplectic_matrix, nqubits)
    symplectic_matrix[:-1, -1] = r ^ (z[:, q] & (z[:, q] ^ x[:, q]))
    symplectic_matrix[:-1, [nqubits + q, q]] = symplectic_matrix[:-1, [q, nqubits + q]]
    return symplectic_matrix


def RY(symplectic_matrix, q, nqubits, theta):
    if theta % (2 * np.pi) == 0:
        return I(symplectic_matrix, q, nqubits)
    elif (theta / np.pi - 1) % 2 == 0:
        return Y(symplectic_matrix, q, nqubits)
    elif (theta / (np.pi / 2) - 1) % 4 == 0:
        """Decomposition --> H-S-S"""
        return RY_pi(symplectic_matrix, q, nqubits)
    else:  # theta == 3*pi/2 + 2*n*pi
        """Decomposition --> H-S-S-H-S-S-H-S-S"""
        return RY_3pi_2(symplectic_matrix, q, nqubits)


def SWAP(symplectic_matrix, control_q, target_q, nqubits):
    """Decomposition --> CNOT-CNOT-CNOT"""
    r, x, z = _get_rxz(symplectic_matrix, nqubits)
    symplectic_matrix[:-1, -1] = (
        r
        ^ (x[:, control_q] & z[:, target_q] & (x[:, target_q] ^ ~z[:, control_q]))
        ^ (
            (x[:, target_q] ^ x[:, control_q])
            & (z[:, target_q] ^ z[:, control_q])
            & (z[:, target_q] ^ ~x[:, control_q])
        )
        ^ (
            x[:, target_q]
            & z[:, control_q]
            & (x[:, control_q] ^ x[:, target_q] ^ z[:, control_q] ^ ~z[:, target_q])
        )
    )
    symplectic_matrix[
        :-1, [control_q, target_q, nqubits + control_q, nqubits + target_q]
    ] = symplectic_matrix[
        :-1, [target_q, control_q, nqubits + target_q, nqubits + control_q]
    ]
    return symplectic_matrix


def iSWAP(symplectic_matrix, control_q, target_q, nqubits):
    """Decomposition --> H-CNOT-CNOT-H-S-S"""
    r, x, z = _get_rxz(symplectic_matrix, nqubits)
    symplectic_matrix[:-1, -1] = (
        r
        ^ (x[:, target_q] & z[:, target_q])
        ^ (x[:, control_q] & z[:, control_q])
        ^ (x[:, control_q] & (z[:, control_q] ^ x[:, control_q]))
        ^ (
            (z[:, control_q] ^ x[:, control_q])
            & (z[:, target_q] ^ x[:, target_q])
            & (x[:, target_q] ^ ~x[:, control_q])
        )
        ^ (
            (x[:, target_q] ^ z[:, control_q] ^ x[:, control_q])
            & (x[:, target_q] ^ z[:, target_q] ^ x[:, control_q])
            & (x[:, target_q] ^ z[:, target_q] ^ x[:, control_q] ^ ~z[:, control_q])
        )
        ^ (x[:, control_q] & (x[:, target_q] ^ x[:, control_q] ^ z[:, control_q]))
    )
    z_control_q = x[:, target_q] ^ z[:, target_q] ^ x[:, control_q]
    z_target_q = x[:, target_q] ^ z[:, control_q] ^ x[:, control_q]
    symplectic_matrix[:-1, nqubits + control_q] = z_control_q
    symplectic_matrix[:-1, nqubits + target_q] = z_target_q
    symplectic_matrix[:-1, [control_q, target_q]] = symplectic_matrix[
        :-1, [target_q, control_q]
    ]
    return symplectic_matrix


def FSWAP(symplectic_matrix, control_q, target_q, nqubits):
    """Decomposition --> X-CNOT-RY-CNOT-RY-CNOT-CNOT-X"""
    symplectic_matrix = X(symplectic_matrix, target_q, nqubits)
    symplectic_matrix = CNOT(symplectic_matrix, control_q, target_q, nqubits)
    symplectic_matrix = RY(symplectic_matrix, control_q, nqubits, np.pi / 2)
    symplectic_matrix = CNOT(symplectic_matrix, target_q, control_q, nqubits)
    symplectic_matrix = RY(symplectic_matrix, control_q, nqubits, -np.pi / 2)
    symplectic_matrix = CNOT(symplectic_matrix, target_q, control_q, nqubits)
    symplectic_matrix = CNOT(symplectic_matrix, control_q, target_q, nqubits)
    return X(symplectic_matrix, control_q, nqubits)


def CY(symplectic_matrix, control_q, target_q, nqubits):
    """Decomposition --> S-CNOT-SDG"""
    r, x, z = _get_rxz(symplectic_matrix, nqubits)
    symplectic_matrix[:-1, -1] = (
        r
        ^ (x[:, target_q] & (z[:, target_q] ^ x[:, target_q]))
        ^ (
            x[:, control_q]
            & (x[:, target_q] ^ z[:, target_q])
            & (z[:, control_q] ^ ~x[:, target_q])
        )
        ^ ((x[:, target_q] ^ x[:, control_q]) & (z[:, target_q] ^ x[:, target_q]))
    )
    x_target_q = x[:, control_q] ^ x[:, target_q]
    z_control_q = z[:, control_q] ^ z[:, target_q] ^ x[:, target_q]
    z_target_q = z[:, target_q] ^ x[:, control_q]
    symplectic_matrix[:-1, target_q] = x_target_q
    symplectic_matrix[:-1, nqubits + control_q] = z_control_q
    symplectic_matrix[:-1, nqubits + target_q] = z_target_q
    return symplectic_matrix


def CRX(symplectic_matrix, control_q, target_q, nqubits, theta):
    # theta = 4 * n * pi
    if theta % (4 * np.pi) == 0:
        return I(symplectic_matrix, target_q, nqubits)
    # theta = pi + 4 * n * pi
    elif (theta / np.pi - 1) % 4 == 0:
        symplectic_matrix = X(symplectic_matrix, target_q, nqubits)
        symplectic_matrix = CZ(symplectic_matrix, control_q, target_q, nqubits)
        symplectic_matrix = X(symplectic_matrix, target_q, nqubits)
        return CY(symplectic_matrix, control_q, target_q, nqubits)
    # theta = 2 * pi + 4 * n * pi
    elif (theta / (2 * np.pi) - 1) % 2 == 0:
        symplectic_matrix = CZ(symplectic_matrix, control_q, target_q, nqubits)
        symplectic_matrix = Y(symplectic_matrix, target_q, nqubits)
        symplectic_matrix = CZ(symplectic_matrix, control_q, target_q, nqubits)
        return Y(symplectic_matrix, target_q, nqubits)
    # theta = 3 * pi + 4 * n * pi
    elif (theta / np.pi - 3) % 4 == 0:
        symplectic_matrix = X(symplectic_matrix, target_q, nqubits)
        symplectic_matrix = CY(symplectic_matrix, control_q, target_q, nqubits)
        symplectic_matrix = X(symplectic_matrix, target_q, nqubits)
        return CZ(symplectic_matrix, control_q, target_q, nqubits)


def CRZ(symplectic_matrix, control_q, target_q, nqubits, theta):
    # theta = 4 * n * pi
    if theta % (4 * np.pi) == 0:
        return I(symplectic_matrix, target_q, nqubits)
    # theta = pi + 4 * n * pi
    elif (theta / np.pi - 1) % 4 == 0:
        symplectic_matrix = X(symplectic_matrix, target_q, nqubits)
        symplectic_matrix = CY(symplectic_matrix, control_q, target_q, nqubits)
        symplectic_matrix = X(symplectic_matrix, target_q, nqubits)
        return CNOT(symplectic_matrix, control_q, target_q, nqubits)
    # theta = 2 * pi + 4 * n * pi
    elif (theta / (2 * np.pi) - 1) % 2 == 0:
        symplectic_matrix = CZ(symplectic_matrix, control_q, target_q, nqubits)
        symplectic_matrix = X(symplectic_matrix, target_q, nqubits)
        symplectic_matrix = CZ(symplectic_matrix, control_q, target_q, nqubits)
        return X(symplectic_matrix, target_q, nqubits)
    # theta = 3 * pi + 4 * n * pi
    elif (theta / np.pi - 3) % 4 == 0:
        symplectic_matrix = CNOT(symplectic_matrix, control_q, target_q, nqubits)
        symplectic_matrix = X(symplectic_matrix, target_q, nqubits)
        symplectic_matrix = CY(symplectic_matrix, control_q, target_q, nqubits)
        return X(symplectic_matrix, target_q, nqubits)


def CRY(symplectic_matrix, control_q, target_q, nqubits, theta):
    # theta = 4 * n * pi
    if theta % (4 * np.pi) == 0:
        return I(symplectic_matrix, target_q, nqubits)
    # theta = pi + 4 * n * pi
    elif (theta / np.pi - 1) % 4 == 0:
        symplectic_matrix = Z(symplectic_matrix, target_q, nqubits)
        symplectic_matrix = CNOT(symplectic_matrix, control_q, target_q, nqubits)
        symplectic_matrix = Z(symplectic_matrix, target_q, nqubits)
        return CZ(symplectic_matrix, control_q, target_q, nqubits)
    # theta = 2 * pi + 4 * n * pi
    elif (theta / (2 * np.pi) - 1) % 2 == 0:
        return CRZ(symplectic_matrix, control_q, target_q, nqubits, theta)
    # theta = 3 * pi + 4 * n * pi
    elif (theta / np.pi - 3) % 4 == 0:
        symplectic_matrix = CZ(symplectic_matrix, control_q, target_q, nqubits)
        symplectic_matrix = Z(symplectic_matrix, target_q, nqubits)
        symplectic_matrix = CNOT(symplectic_matrix, control_q, target_q, nqubits)
        return Z(symplectic_matrix, target_q, nqubits)


def ECR(symplectic_matrix, control_q, target_q, nqubits):
    symplectic_matrix = S(symplectic_matrix, control_q, nqubits)
    symplectic_matrix = SX(symplectic_matrix, target_q, nqubits)
    symplectic_matrix = CNOT(symplectic_matrix, control_q, target_q, nqubits)
    return X(symplectic_matrix, control_q, nqubits)


def _exponent(x1, z1, x2, z2):
    """Helper function that computes the exponent to which i is raised for the product of the x and z paulis encoded in the symplectic matrix. This is used in _rowsum. The computation is performed parallely over the separated paulis x1[i], z1[i], x2[i] and z2[i].

    Args:
        x1 (np.array): Bits of the first x paulis.
        z1 (np.array): Bits of the first z paulis.
        x2 (np.array): Bits of the second x paulis.
        z2 (np.array): Bits of the second z paulis.

    Returns:
        (np.array): The calculated exponents.
    """
    exp = np.zeros(x1.shape, dtype=int)
    x1_eq_z1 = (x1 ^ z1) == False
    x1_neq_z1 = ~x1_eq_z1
    x1_eq_0 = x1 == False
    x1_eq_1 = ~x1_eq_0
    ind2 = x1_eq_z1 & x1_eq_1
    ind3 = x1_eq_1 & x1_neq_z1
    ind4 = x1_eq_0 & x1_neq_z1
    exp[ind2] = z2[ind2].astype(int) - x2[ind2].astype(int)
    exp[ind3] = z2[ind3].astype(int) * (2 * x2[ind3].astype(int) - 1)
    exp[ind4] = x2[ind4].astype(int) * (1 - 2 * z2[ind4].astype(int))
    return exp


def _rowsum(symplectic_matrix, h, i, nqubits):
    """Helper function that updates the symplectic matrix by setting the h-th generator equal to the (i+h)-th one. This is done to keep track of the phase of the h-th row of the symplectic matrix (r[h]). The function is applied parallely over all the rows h and i passed.

    Args:
        symplectic_matrix (np.array): Input symplectic matrix.
        h (np.array): Indices of the rows encoding the generators to update.
        i (np.array): Indices of the rows encoding the generators to use.
        nqubits (int): Total number of qubits.

    Returns:
        (np.array): The updated symplectic matrix.
    """
    xi, xh = symplectic_matrix[i, :nqubits], symplectic_matrix[h, :nqubits]
    zi, zh = symplectic_matrix[i, nqubits:-1], symplectic_matrix[h, nqubits:-1]
    exponents = _exponent(xi, zi, xh, zh)
    ind = (
        2 * symplectic_matrix[h, -1]
        + 2 * symplectic_matrix[i, -1]
        + np.sum(exponents, axis=-1)
    ) % 4 == 0
    r = np.ones(h.shape[0], dtype=bool)
    r[ind] = False

    symplectic_matrix[h, -1] = r
    symplectic_matrix[h, :nqubits] = xi ^ xh
    symplectic_matrix[h, nqubits:-1] = zi ^ zh
    return symplectic_matrix


def _determined_outcome(state, q, nqubits):
    state[-1, :] = False
    for i in state[:nqubits, q].nonzero()[0]:
        state = _rowsum(
            state,
            np.array([2 * nqubits], dtype=np.uint),
            np.array([i + nqubits], dtype=np.uint),
            nqubits,
        )
    return state, np.uint(state[-1, -1])


def _random_outcome(state, p, q, nqubits):
    p = p[0] + nqubits
    tmp = state[p, q].copy()
    state[p, q] = False
    h = state[:-1, q].nonzero()[0]
    state[p, q] = tmp
    if h.shape[0] > 0:
        state = _rowsum(
            state,
            h,
            p * np.ones(h.shape[0], dtype=np.uint),
            nqubits,
        )
    state[p - nqubits, :] = state[p, :]
    outcome = np.random.randint(2, size=1).item()
    state[p, :] = False
    state[p, -1] = outcome
    state[p, nqubits + q] = True
    return state, outcome


def _get_p(state, q, nqubits):
    return state[nqubits:-1, q].nonzero()[0]


# valid for a standard basis measurement only
def M(state, qubits, nqubits, collapse=False):
    sample = []
    state_copy = state if collapse else state.copy()
    for q in qubits:
        p = _get_p(state_copy, q, nqubits)
        # random outcome, affects the state
        if len(p) > 0:
            state_copy, outcome = _random_outcome(state_copy, p, q, nqubits)
        # determined outcome, state unchanged
        else:
            state_copy, outcome = _determined_outcome(state_copy, q, nqubits)
        sample.append(outcome)
    return sample
