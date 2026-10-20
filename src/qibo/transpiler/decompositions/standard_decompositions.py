import math

import numpy as np
from numpy.typing import ArrayLike

from qibo import gates
from qibo.backends import Backend, _check_backend
from qibo.transpiler.decompositions.abstract import GateDecompositions


def _decomposition_generalized_rbs(gate):
    """Generalized RBS gate as in Fig. 2 of arXiv:2405.20408"""
    ins = list(range(len(gate.init_args[0])))
    outs = list(range(len(ins), len(ins) + len(gate.init_args[1])))
    theta = gate.init_kwargs["theta"]
    phi = gate.init_kwargs["phi"]

    # If one register is empty, the gate rotates between the all-zeros and
    # all-ones states of the other register. A single qubit of that register
    # is borrowed to play the role of the empty one and the resulting
    # :math:`|00\rangle`, :math:`|11\rangle` pair is mapped to the
    # Hamming-weight-1 pair with a pair of X gates.
    flipped_qubit = None

    if len(ins) + len(outs) == 1:
        list_gates = [gates.RY(0, 2 * theta)]
        if phi != 0.0:
            list_gates.append(gates.RZ(0, -2 * phi))
        if len(outs) == 0:
            list_gates = [gates.X(0)] + list_gates + [gates.X(0)]
        return list_gates

    if len(ins) == 0:
        ins, outs = outs[:1], outs[1:]
        flipped_qubit = ins[-1]
    elif len(outs) == 0:
        ins, outs = ins[:-1], ins[-1:]
        flipped_qubit = outs[0]

    rotation_controls = ins[:-1] + outs

    list_gates = []
    if len(ins) > 1:
        list_gates.append(gates.X(ins[-1]))
        list_gates.append(gates.FanOut(ins[-1], *ins[:-1]))
        list_gates.append(gates.X(ins[-1]))
    if len(outs) > 1:
        list_gates.append(gates.X(outs[0]))
        list_gates.append(gates.FanOut(outs[0], *outs[1:][::-1]))
        list_gates.append(gates.X(outs[0]))
    if flipped_qubit is not None:
        list_gates.append(gates.X(flipped_qubit))
    list_gates.append(gates.CNOT(ins[-1], outs[0]))
    list_gates.append(gates.RY(ins[-1], -2 * theta).controlled_by(*rotation_controls))
    if phi != 0.0:
        list_gates.append(gates.RZ(ins[-1], 2 * phi).controlled_by(*rotation_controls))
    list_gates.append(gates.CNOT(ins[-1], outs[0]))
    if flipped_qubit is not None:
        list_gates.append(gates.X(flipped_qubit))
    if len(outs) > 1:
        list_gates.append(gates.X(outs[0]))
        list_gates.append(gates.FanOut(outs[0], *outs[1:][::-1]))
        list_gates.append(gates.X(outs[0]))
    if len(ins) > 1:
        list_gates.append(gates.X(ins[-1]))
        list_gates.append(gates.FanOut(ins[-1], *ins[:-1]))
        list_gates.append(gates.X(ins[-1]))

    return list_gates


def _decomposition_multiplexor(gate: gates.Gate) -> list[gates.Gate]:
    """Multiplexed rotation in Gray-code order, see Fig. 2 of Möttönen et al. (2004)."""
    backend = _check_backend(None)

    controls = list(reversed(range(len(gate.init_args[0]))))
    target = len(controls)
    rotation = gates.RY if gate.init_kwargs["axis"] == "y" else gates.RZ

    thetas = _mottonen_compute_theta(gate.parameters[0], backend=backend)
    code = _mottonen_gray_code(len(controls), backend=backend)
    indices = backend.log2(code ^ backend.roll(code, -1))
    indices = backend.cast(indices, dtype=backend.int64)

    list_gates = []
    for theta, index in zip(thetas, indices):
        list_gates.append(rotation(target, theta))
        list_gates.append(gates.CNOT(controls[int(index)], target))

    return list_gates


def _mottonen_compute_theta(
    alpha: ArrayLike, backend: Backend | None = None
) -> ArrayLike:
    """Map uniformly controlled rotation angles to Gray-code decomposition angles.

    See Eq. (3) in Möttönen et al. (2004).
    """
    backend = _check_backend(backend)

    alpha = backend.cast(alpha, dtype=backend.float64)
    orig_shape = alpha.shape
    nqubits = int(math.log2(orig_shape[-1]))
    if nqubits == 0:
        return alpha

    broadcasted = len(orig_shape) > 1
    new_shape = (orig_shape[0],) + (2,) * nqubits if broadcasted else (2,) * nqubits
    theta = backend.reshape(alpha, new_shape)

    # cast to a backend tensor: cupy's tensordot rejects a raw numpy operand
    hadamard = backend.cast(np.array([[1, 1], [1, -1]]) / 2, dtype=theta.dtype)
    for i in range(broadcasted, nqubits + broadcasted):
        theta = backend.tensordot(hadamard, theta, axes=[[1], [i]])

    if nqubits > 1:
        cnot = backend.reshape(backend.matrices.CNOT, (2, 2, 2, 2))
        theta = backend.tensordot(
            cnot, theta, axes=[[2, 3], [nqubits - 1, nqubits - 2]]
        )
        for _ in range(broadcasted + 1, nqubits + broadcasted - 1):
            theta = backend.tensordot(cnot, theta, axes=[[2, 3], [1, nqubits - 1]])
        theta = backend.moveaxis(theta, 0, 1)

    return backend.reshape(backend.transpose(theta), orig_shape)


def _mottonen_gray_code(rank: int, backend: Backend | None = None) -> ArrayLike:
    """Return Gray code of given rank as integer array."""
    backend = _check_backend(backend)

    code = backend.cast([0, 1], dtype=backend.int64)
    for elem in range(1, rank):
        code = backend.concatenate([code, code[::-1] + 2**elem])

    return code


# standard gate decompositions used by :meth:`qibo.gates.gates.Gate.decompose`
standard_decompositions = GateDecompositions()
standard_decompositions.add(gates.CNOT, [gates.CNOT(0, 1)])
standard_decompositions.add(gates.SX, [gates.RX(0, math.pi / 2, trainable=False)])
standard_decompositions.add(gates.SXDG, [gates.RX(0, -math.pi / 2, trainable=False)])
standard_decompositions.add(
    gates.PRX,
    lambda gate: [
        gates.RZ(0, -gate.parameters[1] - math.pi / 2),
        gates.RY(0, -gate.parameters[0]),
        gates.RZ(0, gate.parameters[1] + math.pi / 2),
    ],
)
standard_decompositions.add(
    gates.U3,
    lambda gate: [
        gates.RZ(0, gate.parameters[2]),
        gates.SX(0),
        gates.RZ(0, gate.parameters[0] + math.pi),
        gates.SX(0),
        gates.RZ(0, gate.parameters[1] + math.pi),
    ],
)
standard_decompositions.add(gates.CY, [gates.SDG(1), gates.CNOT(0, 1), gates.S(1)])
standard_decompositions.add(gates.CZ, [gates.H(1), gates.CNOT(0, 1), gates.H(1)])
standard_decompositions.add(
    gates.CH,
    [
        gates.S(1),
        gates.H(1),
        gates.T(1),
        gates.CNOT(0, 1),
        gates.TDG(1),
        gates.H(1),
        gates.SDG(1),
    ],
)
standard_decompositions.add(
    gates.CSX, [gates.H(1), gates.CU1(0, 1, math.pi / 2), gates.H(1)]
)
standard_decompositions.add(
    gates.CSXDG, [gates.H(1), gates.CU1(0, 1, -math.pi / 2), gates.H(1)]
)
standard_decompositions.add(
    gates.CRX,
    lambda gate: [
        gates.RX(1, gate.parameters[0] / 2.0),
        gates.H(1),
        gates.CNOT(0, 1),
        gates.RZ(1, -gate.parameters[0] / 2.0),
        gates.CNOT(0, 1),
        gates.H(1),
    ],
)
standard_decompositions.add(
    gates.CRY,
    lambda gate: [
        gates.RY(1, gate.parameters[0] / 4),
        gates.CNOT(0, 1),
        gates.RY(1, -gate.parameters[0] / 2),
        gates.CNOT(0, 1),
        gates.RY(1, gate.parameters[0] / 4),
    ],
)
standard_decompositions.add(
    gates.CRZ,
    lambda gate: [
        gates.RZ(1, gate.parameters[0] / 2.0),
        gates.CNOT(0, 1),
        gates.RZ(1, -gate.parameters[0] / 2.0),
        gates.CNOT(0, 1),
    ],
)
standard_decompositions.add(
    gates.CU1,
    lambda gate: [
        gates.RZ(0, gate.parameters[0] / 2.0),
        gates.CNOT(0, 1),
        gates.RZ(1, -gate.parameters[0] / 2.0),
        gates.CNOT(0, 1),
        gates.RZ(1, gate.parameters[0] / 2.0),
    ],
)
standard_decompositions.add(
    gates.CU2,
    lambda gate: [
        gates.RZ(1, (gate.parameters[1] - gate.parameters[0]) / 2.0),
        gates.CNOT(0, 1),
        gates.U3(1, -math.pi / 4, 0, -(gate.parameters[1] + gate.parameters[0]) / 2.0),
        gates.CNOT(0, 1),
        gates.U3(1, math.pi / 4, gate.parameters[0], 0),
    ],
)
standard_decompositions.add(
    gates.CU3,
    lambda gate: [
        gates.RZ(1, (gate.parameters[2] - gate.parameters[1]) / 2.0),
        gates.CNOT(0, 1),
        gates.U3(
            1,
            -gate.parameters[0] / 2.0,
            0,
            -(gate.parameters[2] + gate.parameters[1]) / 2.0,
        ),
        gates.CNOT(0, 1),
        gates.U3(1, gate.parameters[0] / 2.0, gate.parameters[1], 0),
    ],
)
standard_decompositions.add(
    gates.SWAP, [gates.CNOT(0, 1), gates.CNOT(1, 0), gates.CNOT(0, 1)]
)
standard_decompositions.add(
    gates.iSWAP,
    [
        gates.S(0),
        gates.S(1),
        gates.H(0),
        gates.CNOT(0, 1),
        gates.CNOT(1, 0),
        gates.H(1),
    ],
)
standard_decompositions.add(
    gates.SiSWAP,
    [
        gates.SX(0),
        gates.SX(1),
        gates.RZ(0, math.pi / 2),
        gates.CNOT(0, 1),
        gates.SX(0),
        gates.RZ(1, -math.pi / 4),
        gates.RZ(0, -math.pi / 4),
        gates.SX(1),
        gates.SX(0),
        gates.RZ(0, math.pi / 2),
        gates.CNOT(0, 1),
        gates.SX(0),
    ],
)
standard_decompositions.add(
    gates.SiSWAPDG, [gate.dagger() for gate in gates.SiSWAP(0, 1).decompose()[::-1]]
)
standard_decompositions.add(
    gates.RXX,
    lambda gate: [
        gates.H(0),
        gates.H(1),
        gates.CNOT(0, 1),
        gates.RZ(1, gate.parameters[0]),
        gates.CNOT(0, 1),
        gates.H(1),
        gates.H(0),
    ],
)
standard_decompositions.add(
    gates.RYY,
    lambda gate: [
        gates.RX(0, math.pi / 2),
        gates.RX(1, math.pi / 2),
        gates.CNOT(0, 1),
        gates.RZ(1, gate.parameters[0]),
        gates.CNOT(0, 1),
        gates.RX(1, -math.pi / 2),
        gates.RX(0, -math.pi / 2),
    ],
)
standard_decompositions.add(
    gates.RZZ,
    lambda gate: [
        gates.CNOT(0, 1),
        gates.RZ(1, gate.parameters[0]),
        gates.CNOT(0, 1),
    ],
)
standard_decompositions.add(
    gates.RZX,
    lambda gate: [
        gates.H(1),
        gates.CNOT(0, 1),
        gates.RZ(1, gate.parameters[0]),
        gates.CNOT(0, 1),
        gates.H(1),
    ],
)
standard_decompositions.add(
    gates.RXXYY,
    lambda gate: [
        gates.RZ(1, -math.pi / 2),
        gates.S(0),
        gates.SX(1),
        gates.RZ(1, math.pi / 2),
        gates.CNOT(1, 0),
        gates.RY(0, -gate.parameters[0] / 2),
        gates.RY(1, -gate.parameters[0] / 2),
        gates.CNOT(1, 0),
        gates.SDG(0),
        gates.RZ(1, -math.pi / 2),
        gates.SXDG(1),
        gates.RZ(1, math.pi / 2),
    ],
)
standard_decompositions.add(
    gates.RBS,
    lambda gate: [
        gates.H(0),
        gates.CNOT(0, 1),
        gates.RY(0, gate.parameters[0]),
        gates.RY(1, gate.parameters[0]),
        gates.CNOT(0, 1),
        gates.H(0),
    ],
)
standard_decompositions.add(
    gates.GIVENS, lambda gate: gates.RBS(0, 1, -gate.parameters[0]).decompose()
)
standard_decompositions.add(
    gates.FSWAP,
    [gates.X(1)] + gates.GIVENS(0, 1, math.pi / 2).decompose() + [gates.X(0)],
)
standard_decompositions.add(
    gates.ECR,
    [gates.S(0), gates.RX(1, math.pi / 2), gates.CNOT(0, 1), gates.X(0)],
)
standard_decompositions.add(
    gates.CCZ,
    [
        gates.CNOT(1, 2),
        gates.TDG(2),
        gates.CNOT(0, 2),
        gates.T(2),
        gates.CNOT(1, 2),
        gates.T(1),
        gates.TDG(2),
        gates.CNOT(0, 2),
        gates.CNOT(0, 1),
        gates.T(2),
        gates.T(0),
        gates.TDG(1),
        gates.CNOT(0, 1),
    ],
)
standard_decompositions.add(
    gates.TOFFOLI,
    [
        gates.H(2),
        gates.CNOT(1, 2),
        gates.TDG(2),
        gates.CNOT(0, 2),
        gates.T(2),
        gates.CNOT(1, 2),
        gates.T(1),
        gates.TDG(2),
        gates.CNOT(0, 2),
        gates.CNOT(0, 1),
        gates.T(2),
        gates.T(0),
        gates.TDG(1),
        gates.H(2),
        gates.CNOT(0, 1),
    ],
)
standard_decompositions.add(
    gates.DEUTSCH,
    lambda gate: [
        gates.H(2),
        gates.CNOT(1, 2),
        gates.RZ(2, -gate.parameters[0] / 2),
        gates.CNOT(0, 2),
        gates.RZ(2, gate.parameters[0] / 2),
        gates.CNOT(1, 2),
        gates.T(1),
        gates.RZ(2, -gate.parameters[0] / 2),
        gates.CNOT(0, 2),
        gates.CNOT(0, 1),
        gates.RZ(2, gate.parameters[0] / 2),
        gates.T(0),
        gates.TDG(1),
        gates.H(2),
        gates.CNOT(0, 1),
    ],
)
standard_decompositions.add(
    gates.FanOut,
    lambda gate: [gates.CNOT(0, qub) for qub in range(1, len(gate.qubits))],
)
standard_decompositions.add(
    gates.GeneralizedRBS,
    lambda gate: _decomposition_generalized_rbs(gate),
)
standard_decompositions.add(gates.Multiplexor, _decomposition_multiplexor)
