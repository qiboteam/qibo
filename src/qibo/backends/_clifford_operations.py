import math
from functools import cache, reduce
from typing import Optional, Tuple

import numpy as np
from numpy.typing import ArrayLike, DTypeLike
from scipy import sparse

name = "numpy"


def _get_rxz(symplectic_matrix: ArrayLike, nqubits: int) -> Tuple[ArrayLike, ...]:
    return (
        symplectic_matrix[:, -1],
        symplectic_matrix[:, :nqubits],
        symplectic_matrix[:, nqubits:-1],
    )


def I(symplectic_matrix: ArrayLike, q: int, nqubits: int) -> ArrayLike:
    return symplectic_matrix


def H(symplectic_matrix: ArrayLike, q: int, nqubits: int) -> ArrayLike:
    r, x, z = _get_rxz(symplectic_matrix, nqubits)
    symplectic_matrix[:, -1] = r ^ (x[:, q] & z[:, q])
    symplectic_matrix[:, [q, nqubits + q]] = symplectic_matrix[:, [nqubits + q, q]]
    return symplectic_matrix


def CNOT(
    symplectic_matrix: ArrayLike, control_q: int, target_q: int, nqubits: int
) -> ArrayLike:
    ind_zt = nqubits + target_q
    ind_zc = nqubits + control_q
    r = symplectic_matrix[:, -1]
    xcq = symplectic_matrix[:, control_q]
    xtq = symplectic_matrix[:, target_q]
    ztq = symplectic_matrix[:, ind_zt]
    zcq = symplectic_matrix[:, ind_zc]
    symplectic_matrix[:, -1] = r ^ (xcq & ztq) & (xtq ^ ~zcq)
    symplectic_matrix[:, target_q] = xtq ^ xcq
    symplectic_matrix[:, ind_zc] = zcq ^ ztq
    return symplectic_matrix


def CZ(symplectic_matrix: ArrayLike, control_q: int, target_q: int, nqubits: int):
    """Decomposition --> H-CNOT-H"""
    ind_zt = nqubits + target_q
    ind_zc = nqubits + control_q
    r = symplectic_matrix[:, -1]
    xcq = symplectic_matrix[:, control_q]
    xtq = symplectic_matrix[:, target_q]
    ztq = symplectic_matrix[:, ind_zt]
    zcq = symplectic_matrix[:, ind_zc]
    ztq_xor_xcq = ztq ^ xcq
    symplectic_matrix[:, -1] = (
        r ^ (xtq & ztq) ^ (xcq & xtq & (ztq ^ ~zcq)) ^ (xtq & ztq_xor_xcq)
    )
    z_control_q = xtq ^ zcq
    z_target_q = ztq_xor_xcq
    symplectic_matrix[:, ind_zc] = z_control_q
    symplectic_matrix[:, ind_zt] = z_target_q
    return symplectic_matrix


def S(symplectic_matrix: ArrayLike, q: int, nqubits: int) -> ArrayLike:
    r, x, z = _get_rxz(symplectic_matrix, nqubits)
    symplectic_matrix[:, -1] = r ^ (x[:, q] & z[:, q])
    symplectic_matrix[:, q + nqubits] = z[:, q] ^ x[:, q]
    return symplectic_matrix


def Z(symplectic_matrix: ArrayLike, q: int, nqubits: int) -> ArrayLike:
    """Decomposition --> S-S"""
    r, x, z = _get_rxz(symplectic_matrix, nqubits)
    symplectic_matrix[:, -1] = r ^ ((x[:, q] & z[:, q]) ^ x[:, q] & (z[:, q] ^ x[:, q]))
    return symplectic_matrix


def X(symplectic_matrix: ArrayLike, q: int, nqubits: int) -> ArrayLike:
    """Decomposition --> H-S-S-H"""
    r, x, z = _get_rxz(symplectic_matrix, nqubits)
    symplectic_matrix[:, -1] = r ^ (z[:, q] & (z[:, q] ^ x[:, q])) ^ (z[:, q] & x[:, q])
    return symplectic_matrix


def Y(symplectic_matrix: ArrayLike, q: int, nqubits: int) -> ArrayLike:
    """Decomposition --> S-S-H-S-S-H"""
    r, x, z = _get_rxz(symplectic_matrix, nqubits)
    symplectic_matrix[:, -1] = (
        r ^ (z[:, q] & (z[:, q] ^ x[:, q])) ^ (x[:, q] & (z[:, q] ^ x[:, q]))
    )
    return symplectic_matrix


def SX(symplectic_matrix: ArrayLike, q: int, nqubits: int) -> ArrayLike:
    """Decomposition --> H-S-H"""
    r, x, z = _get_rxz(symplectic_matrix, nqubits)
    symplectic_matrix[:, -1] = r ^ (z[:, q] & (z[:, q] ^ x[:, q]))
    symplectic_matrix[:, q] = z[:, q] ^ x[:, q]
    return symplectic_matrix


def SDG(symplectic_matrix: ArrayLike, q: int, nqubits: int) -> ArrayLike:
    """Decomposition --> S-S-S"""
    r, x, z = _get_rxz(symplectic_matrix, nqubits)
    symplectic_matrix[:, -1] = r ^ (x[:, q] & (z[:, q] ^ x[:, q]))
    symplectic_matrix[:, nqubits + q] = z[:, q] ^ x[:, q]
    return symplectic_matrix


def SXDG(symplectic_matrix: ArrayLike, q: int, nqubits: int) -> ArrayLike:
    """Decomposition --> H-S-S-S-H"""
    r, x, z = _get_rxz(symplectic_matrix, nqubits)
    symplectic_matrix[:, -1] = r ^ (z[:, q] & x[:, q])
    symplectic_matrix[:, q] = z[:, q] ^ x[:, q]
    return symplectic_matrix


def RX(symplectic_matrix: ArrayLike, q: int, nqubits: int, theta: float) -> ArrayLike:
    if theta % (2 * math.pi) == 0:
        return I(symplectic_matrix, q, nqubits)

    if (theta / math.pi - 1) % 2 == 0:
        return X(symplectic_matrix, q, nqubits)

    if (theta / (math.pi / 2) - 1) % 4 == 0:
        return SX(symplectic_matrix, q, nqubits)

    # theta == 3*pi/2 + 2*n*pi
    return SXDG(symplectic_matrix, q, nqubits)


def RZ(symplectic_matrix: ArrayLike, q: int, nqubits: int, theta: float) -> ArrayLike:
    if theta % (2 * math.pi) == 0:
        return I(symplectic_matrix, q, nqubits)

    if (theta / math.pi - 1) % 2 == 0:
        return Z(symplectic_matrix, q, nqubits)

    if (theta / (math.pi / 2) - 1) % 4 == 0:
        return S(symplectic_matrix, q, nqubits)

    # theta == 3*pi/2 + 2*n*pi
    return SDG(symplectic_matrix, q, nqubits)


def RY_pi(symplectic_matrix: ArrayLike, q: int, nqubits: int) -> ArrayLike:
    """Decomposition --> H-S-S"""
    r, x, z = _get_rxz(symplectic_matrix, nqubits)
    symplectic_matrix[:, -1] = r ^ (x[:, q] & (z[:, q] ^ x[:, q]))
    symplectic_matrix[:, [nqubits + q, q]] = symplectic_matrix[:, [q, nqubits + q]]
    return symplectic_matrix


def RY_3pi_2(symplectic_matrix: ArrayLike, q: int, nqubits: int) -> ArrayLike:
    """Decomposition --> H-S-S-H-S-S-H-S-S"""
    r, x, z = _get_rxz(symplectic_matrix, nqubits)
    symplectic_matrix[:, -1] = r ^ (z[:, q] & (z[:, q] ^ x[:, q]))
    symplectic_matrix[:, [nqubits + q, q]] = symplectic_matrix[:, [q, nqubits + q]]
    return symplectic_matrix


def RY(symplectic_matrix: ArrayLike, q: int, nqubits: int, theta: float) -> ArrayLike:
    if theta % (2 * math.pi) == 0:
        return I(symplectic_matrix, q, nqubits)

    if (theta / math.pi - 1) % 2 == 0:
        return Y(symplectic_matrix, q, nqubits)

    if (theta / (math.pi / 2) - 1) % 4 == 0:
        """Decomposition --> H-S-S"""
        return RY_pi(symplectic_matrix, q, nqubits)

    # theta == 3*pi/2 + 2*n*pi
    # Decomposition --> H-S-S-H-S-S-H-S-S
    return RY_3pi_2(symplectic_matrix, q, nqubits)


def GPI2(symplectic_matrix: ArrayLike, q: int, nqubits: int, phi: float) -> ArrayLike:
    if phi % (2 * math.pi) == 0:
        return RX(symplectic_matrix, q, nqubits, math.pi / 2)

    if (phi / math.pi - 1) % 2 == 0:
        return RX(symplectic_matrix, q, nqubits, -math.pi / 2)

    if (phi / (math.pi / 2) - 1) % 4 == 0:
        return RY(symplectic_matrix, q, nqubits, math.pi / 2)

    # theta == 3*pi/2 + 2*n*pi
    return RY(symplectic_matrix, q, nqubits, -math.pi / 2)


def SWAP(
    symplectic_matrix: ArrayLike, control_q: int, target_q: int, nqubits: int
) -> ArrayLike:
    """Decomposition --> CNOT-CNOT-CNOT"""
    r, x, z = _get_rxz(symplectic_matrix, nqubits)
    symplectic_matrix[:, -1] = (
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
        :, [control_q, target_q, nqubits + control_q, nqubits + target_q]
    ] = symplectic_matrix[
        :, [target_q, control_q, nqubits + target_q, nqubits + control_q]
    ]
    return symplectic_matrix


def iSWAP(
    symplectic_matrix: ArrayLike, control_q: int, target_q: int, nqubits: int
) -> ArrayLike:
    """Decomposition --> H-CNOT-CNOT-H-S-S"""
    r, x, z = _get_rxz(symplectic_matrix, nqubits)
    symplectic_matrix[:, -1] = (
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
    symplectic_matrix[:, nqubits + control_q] = z_control_q
    symplectic_matrix[:, nqubits + target_q] = z_target_q
    symplectic_matrix[:, [control_q, target_q]] = symplectic_matrix[
        :, [target_q, control_q]
    ]
    return symplectic_matrix


def FSWAP(
    symplectic_matrix: ArrayLike, control_q: int, target_q: int, nqubits: int
) -> ArrayLike:
    """Decomposition --> X-CNOT-RY-CNOT-RY-CNOT-CNOT-X"""
    symplectic_matrix = X(symplectic_matrix, target_q, nqubits)
    symplectic_matrix = CNOT(symplectic_matrix, control_q, target_q, nqubits)
    symplectic_matrix = RY(symplectic_matrix, control_q, nqubits, math.pi / 2)
    symplectic_matrix = CNOT(symplectic_matrix, target_q, control_q, nqubits)
    symplectic_matrix = RY(symplectic_matrix, control_q, nqubits, -math.pi / 2)
    symplectic_matrix = CNOT(symplectic_matrix, target_q, control_q, nqubits)
    symplectic_matrix = CNOT(symplectic_matrix, control_q, target_q, nqubits)
    return X(symplectic_matrix, control_q, nqubits)


def CY(
    symplectic_matrix: ArrayLike, control_q: int, target_q: int, nqubits: int
) -> ArrayLike:
    """Decomposition --> S-CNOT-SDG"""
    r, x, z = _get_rxz(symplectic_matrix, nqubits)
    symplectic_matrix[:, -1] = (
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
    symplectic_matrix[:, target_q] = x_target_q
    symplectic_matrix[:, nqubits + control_q] = z_control_q
    symplectic_matrix[:, nqubits + target_q] = z_target_q
    return symplectic_matrix


def CRX(
    symplectic_matrix: ArrayLike,
    control_q: int,
    target_q: int,
    nqubits: int,
    theta: float,
) -> ArrayLike:
    # theta = 4 * n * pi
    if theta % (4 * math.pi) == 0:
        return I(symplectic_matrix, target_q, nqubits)

    # theta = pi + 4 * n * pi
    if (theta / math.pi - 1) % 4 == 0:
        symplectic_matrix = X(symplectic_matrix, target_q, nqubits)
        symplectic_matrix = CZ(symplectic_matrix, control_q, target_q, nqubits)
        symplectic_matrix = X(symplectic_matrix, target_q, nqubits)
        return CY(symplectic_matrix, control_q, target_q, nqubits)

    # theta = 2 * pi + 4 * n * pi
    if (theta / (2 * math.pi) - 1) % 2 == 0:
        symplectic_matrix = CZ(symplectic_matrix, control_q, target_q, nqubits)
        symplectic_matrix = Y(symplectic_matrix, target_q, nqubits)
        symplectic_matrix = CZ(symplectic_matrix, control_q, target_q, nqubits)
        return Y(symplectic_matrix, target_q, nqubits)

    # theta = 3 * pi + 4 * n * pi
    symplectic_matrix = X(symplectic_matrix, target_q, nqubits)
    symplectic_matrix = CY(symplectic_matrix, control_q, target_q, nqubits)
    symplectic_matrix = X(symplectic_matrix, target_q, nqubits)
    return CZ(symplectic_matrix, control_q, target_q, nqubits)


def CRZ(
    symplectic_matrix: ArrayLike,
    control_q: int,
    target_q: int,
    nqubits: int,
    theta: float,
) -> ArrayLike:
    # theta = 4 * n * pi
    if theta % (4 * math.pi) == 0:
        return I(symplectic_matrix, target_q, nqubits)

    # theta = pi + 4 * n * pi
    if (theta / math.pi - 1) % 4 == 0:
        symplectic_matrix = X(symplectic_matrix, target_q, nqubits)
        symplectic_matrix = CY(symplectic_matrix, control_q, target_q, nqubits)
        symplectic_matrix = X(symplectic_matrix, target_q, nqubits)
        return CNOT(symplectic_matrix, control_q, target_q, nqubits)

    # theta = 2 * pi + 4 * n * pi
    if (theta / (2 * math.pi) - 1) % 2 == 0:
        symplectic_matrix = CZ(symplectic_matrix, control_q, target_q, nqubits)
        symplectic_matrix = X(symplectic_matrix, target_q, nqubits)
        symplectic_matrix = CZ(symplectic_matrix, control_q, target_q, nqubits)
        return X(symplectic_matrix, target_q, nqubits)

    # theta = 3 * pi + 4 * n * pi
    symplectic_matrix = CNOT(symplectic_matrix, control_q, target_q, nqubits)
    symplectic_matrix = X(symplectic_matrix, target_q, nqubits)
    symplectic_matrix = CY(symplectic_matrix, control_q, target_q, nqubits)
    return X(symplectic_matrix, target_q, nqubits)


def CRY(
    symplectic_matrix: ArrayLike,
    control_q: int,
    target_q: int,
    nqubits: int,
    theta: float,
) -> ArrayLike:
    # theta = 4 * n * pi
    if theta % (4 * math.pi) == 0:
        return I(symplectic_matrix, target_q, nqubits)

    # theta = pi + 4 * n * pi
    if (theta / math.pi - 1) % 4 == 0:
        symplectic_matrix = Z(symplectic_matrix, target_q, nqubits)
        symplectic_matrix = CNOT(symplectic_matrix, control_q, target_q, nqubits)
        symplectic_matrix = Z(symplectic_matrix, target_q, nqubits)
        return CZ(symplectic_matrix, control_q, target_q, nqubits)

    # theta = 2 * pi + 4 * n * pi
    if (theta / (2 * math.pi) - 1) % 2 == 0:
        return CRZ(symplectic_matrix, control_q, target_q, nqubits, theta)

    # theta = 3 * pi + 4 * n * pi
    symplectic_matrix = CZ(symplectic_matrix, control_q, target_q, nqubits)
    symplectic_matrix = Z(symplectic_matrix, target_q, nqubits)
    symplectic_matrix = CNOT(symplectic_matrix, control_q, target_q, nqubits)
    return Z(symplectic_matrix, target_q, nqubits)


def ECR(
    symplectic_matrix: ArrayLike, control_q: int, target_q: int, nqubits: int
) -> ArrayLike:
    symplectic_matrix = S(symplectic_matrix, control_q, nqubits)
    symplectic_matrix = SX(symplectic_matrix, target_q, nqubits)
    symplectic_matrix = CNOT(symplectic_matrix, control_q, target_q, nqubits)
    return X(symplectic_matrix, control_q, nqubits)


def _exponent(x1: ArrayLike, z1: ArrayLike, x2: ArrayLike, z2: ArrayLike) -> ArrayLike:
    """Helper function that computes the exponent to which i is raised for the product
    of the x and z paulis encoded in the symplectic matrix.

    This is used in _rowsum. The computation is performed parallely over the separated
    paulis x1[i], z1[i], x2[i] and z2[i].

    Args:
        x1 (ArrayLike): Bits of the first x paulis.
        z1 (ArrayLike): Bits of the first z paulis.
        x2 (ArrayLike): Bits of the second x paulis.
        z2 (ArrayLike): Bits of the second z paulis.

    Returns:
        ArrayLike: The calculated exponents.
    """
    # this cannot be performed in the packed representation for measurements (thus packed rows)
    # because bitwise arithmetic difference and sum are needed, which cannot be done directly
    # in the packed representation.
    return 2 * (x1 * x2 * (z2 - z1) + z1 * z2 * (x1 - x2)) - x1 * z2 + x2 * z1


def _rowsum(
    symplectic_matrix: ArrayLike,
    h: ArrayLike,
    i: ArrayLike,
    nqubits: int,
    determined: bool = False,
) -> ArrayLike:
    """Helper function that updates the symplectic matrix by setting the h-th generator
    equal to the (i+h)-th one.

    This is done to keep track of the phase of the h-th row of the symplectic matrix (r[h]).
    The function is applied parallely over all the rows h and i passed.

    Args:
        symplectic_matrix (ArrayLike): Input symplectic matrix.
        h (ArrayLike): Indices of the rows encoding the generators to update.
        i (ArrayLike): Indices of the rows encoding the generators to use.
        nqubits (int): Total number of qubits.

    Returns:
        ArrayLike: The updated symplectic matrix.
    """
    # calculate the exponent in the unpacked representation
    xi, zi = symplectic_matrix[i, :nqubits], symplectic_matrix[i, nqubits:-1]
    xh, zh = symplectic_matrix[h, :nqubits], symplectic_matrix[h, nqubits:-1]
    exponents = _exponent(xi, zi, xh, zh)
    ind = (
        2 * symplectic_matrix[h, -1]
        + 2 * symplectic_matrix[i, -1]
        + np.sum(exponents, axis=-1)
    ) % 4 == 0
    r = np.ones(h.shape[0], dtype=np.uint8)
    r[ind] = 0

    # the rest can be done in the packed representation
    symplectic_matrix = _pack_for_measurements(symplectic_matrix, nqubits)
    packed_n = _packed_size(nqubits)
    xi, zi = symplectic_matrix[i, :packed_n], symplectic_matrix[i, packed_n:-1]
    xh, zh = symplectic_matrix[h, :packed_n], symplectic_matrix[h, packed_n:-1]
    xi_xh = xi ^ xh
    zi_zh = zi ^ zh
    if determined:
        r = reduce(np.logical_xor, r)
        xi_xh = reduce(np.logical_xor, xi_xh)
        zi_zh = reduce(np.logical_xor, zi_zh)
        symplectic_matrix[h[0], -1] = r
        symplectic_matrix[h[0], :packed_n] = xi_xh
        symplectic_matrix[h[0], packed_n:-1] = zi_zh
    else:
        symplectic_matrix[h, -1] = r
        symplectic_matrix[h, :packed_n] = xi_xh
        symplectic_matrix[h, packed_n:-1] = zi_zh
    return _unpack_for_measurements(symplectic_matrix, nqubits)


def _determined_outcome(state: ArrayLike, q: int, nqubits: int) -> ArrayLike:
    """Extracts the outcome for a measurement in case it is determined."""
    state[-1, :] = 0
    idx = (state[:nqubits, q].nonzero()[0] + nqubits).astype(np.uint)

    if len(idx) == 0:
        return state, state[-1, -1]

    state = _rowsum(
        state,
        _dim_xz(nqubits) * np.ones(idx.shape, dtype=np.uint),
        idx,
        nqubits,
        True,
    )

    return state, state[-1, -1]


def _random_outcome(state: ArrayLike, p: int, q: int, nqubits: int) -> ArrayLike:
    """Extracts the outcome for a measurement in case it is random."""
    p = p[0] + nqubits
    state[p, q] = 0
    h = state[:-1, q].nonzero()[0]
    state[p, q] = 1
    if h.shape[0] > 0:
        state = _rowsum(
            state,
            h.astype(np.uint),
            np.uint(p) * np.ones(h.shape[0], dtype=np.uint),
            nqubits,
            False,
        )
    state[p - nqubits, :] = state[p, :]
    outcome = np.random.randint(2, size=1).item()
    state[p, :] = 0
    state[p, -1] = outcome
    state[p, nqubits + q] = 1
    return state, outcome


@cache
def _dim(nqubits: int) -> int:
    """Return the dimension of the symplectic matrix for a given number of qubits."""
    return _dim_xz(nqubits) + 1


@cache
def _dim_xz(nqubits: int) -> int:
    """Returns the dimension of the symplectic matrix (only the de/stabilizers generators part,
    without the phases and scratch row) for a given number of qubits."""
    return 2 * nqubits


@cache
def _packed_size(n: int) -> int:
    """Return the size of an array of `n` booleans after packing."""
    return np.ceil(n / 8).astype(int)


def _packbits(array: ArrayLike, axis: int) -> ArrayLike:
    return np.packbits(array, axis=axis)


def _unpackbits(array: ArrayLike, axis: int, count: int) -> ArrayLike:
    return np.unpackbits(array, axis=axis, count=count)


def _pack_for_measurements(state: ArrayLike, nqubits: int) -> ArrayLike:
    """Prepares the state for measurements by packing the rows of the X and Z sections of the symplectic matrix."""
    r, x, z = _get_rxz(state, nqubits)
    x = _packbits(x, axis=1)
    z = _packbits(z, axis=1)
    return np.hstack((x, z, r[:, None]))


def _unpack_for_measurements(state: ArrayLike, nqubits: int) -> ArrayLike:
    """Unpacks the symplectc matrix that was packed for measurements."""
    x = _unpackbits(state[:, : _packed_size(nqubits)], axis=1, count=nqubits)
    z = _unpackbits(state[:, _packed_size(nqubits) : -1], axis=1, count=nqubits)
    return np.hstack((x, z, state[:, -1][:, None]))


def _init_state_for_measurements(
    state: ArrayLike, nqubits: int, collapse: bool
) -> ArrayLike:
    if collapse:
        return _unpackbits(state, axis=0, count=_dim(nqubits))

    return state.copy()


# valid for a standard basis measurement only
def M(
    state: ArrayLike, qubits: Tuple[int, ...], nqubits: int, collapse: bool = False
) -> ArrayLike:
    sample = []
    state = _init_state_for_measurements(state, nqubits, collapse)
    # TODO: parallelize this and get rid of the loop
    for q in qubits:
        p = state[nqubits:-1, q].nonzero()[0]
        # random outcome, affects the state
        if len(p) > 0:
            state, outcome = _random_outcome(state, p, q, nqubits)
        # determined outcome, state unchanged
        else:
            _, outcome = _determined_outcome(state, q, nqubits)
        sample.append(outcome)

    if collapse:
        state = _packbits(state, axis=0)

    return sample


def cast(
    array: ArrayLike, dtype: Optional[DTypeLike] = None, copy: bool = False
) -> ArrayLike:
    if dtype is None:
        dtype = "complex128"

    if isinstance(array, np.ndarray):
        return array.astype(dtype, copy=copy)

    if sparse.issparse(array):  # pragma: no cover
        return array.astype(dtype, copy=copy)

    return np.asarray(array, dtype=dtype, copy=copy if copy else None)


def _clifford_pre_execution_reshape(state: ArrayLike) -> ArrayLike:
    """Reshape and packing applied to the symplectic matrix before execution to prepare
    the state in the form needed by each engine.

    Args:
        state (ArrayLike): Input state.

    Returns:
        ArrayLike: The packed and reshaped state.
    """
    return _packbits(state, axis=0)


def _clifford_post_execution_reshape(state: ArrayLike, nqubits: int) -> ArrayLike:
    """Reshape and unpacking applied to the state after execution to retrieve
    the standard symplectic matrix form.

    Args:
        state (ArrayLike): Input state.
        nqubits (int): Number of qubits.

    Returns:
        ArrayLike: The unpacked and reshaped state.
    """
    state = _unpackbits(state, axis=0, count=_dim(nqubits))[: _dim(nqubits)]
    return state


def csr_matrix(array: ArrayLike, **kwargs) -> ArrayLike:
    return sparse.csr_matrix(array, **kwargs)


def _identity_sparse(
    dims: int, dtype: Optional[DTypeLike] = None, **kwargs
) -> ArrayLike:
    if dtype is None:  # pragma: no cover
        dtype = "complex128"

    sparsity_format = kwargs.get("format", "csr")

    return sparse.eye(dims, dtype=dtype, format=sparsity_format, **kwargs)
