import math
from inspect import signature
from typing import List, Optional, Set, Tuple, Union

import numpy as np
from numpy.typing import ArrayLike, DTypeLike
from scipy.special import binom

from qibo import gates
from qibo.backends import Backend, _check_backend
from qibo.config import raise_error
from qibo.gates.abstract import Gate
from qibo.models.circuit import Circuit  # to avoid circular import


def _add_dicke_unitary_gate(circuit: Circuit, qubits: List[int], weight: int) -> None:
    """In-place addition to ``circuit`` of a U_{n,k} gate from Definition 2 of the paper [1]."""
    nqubits = len(qubits)
    for m in range(nqubits, weight, -1):
        # Add SCS_{m,k} acting on last k+1 qubits
        _add_scs_gate(circuit, qubits[:m], weight)

    # Recursively build the unitary U_n,k
    for m in range(weight, 0, -1):
        # Add SCS_{m,m-1} acting on last m qubits
        _add_scs_gate(circuit, qubits[:m], m - 1)


def _add_scs_gate(circuit: Circuit, qubits: List[int], weight: int) -> None:
    """In-place addition of a Split & Cyclic Shift (SCS) gate to ``circuit``.
    Implements the SCS_{n,k} unitary from Definition 3 of the paper [1].
    Acts on the last weight+1 qubits of the nqubits passed qubits.
    """
    if weight == 0:
        return  # SCS_{n,0} is identity

    nqubits = len(qubits)
    last_qubit = qubits[-1]  # qubits[nqubits - 1]
    target_qubit = qubits[-2]
    # first_qubit = qubits[nqubits - 1 - weight]

    # Gate (i) - acts on last two qubits
    theta = 2 * np.arccos(np.sqrt(1 / nqubits))
    circuit.add(gates.CNOT(target_qubit, last_qubit))
    circuit.add(gates.RY(target_qubit, theta).controlled_by(last_qubit))
    circuit.add(gates.CNOT(target_qubit, last_qubit))

    # Gates (ii)_ℓ for ℓ from 2 to k
    for l in range(2, weight + 1):
        theta = 2 * np.arccos(np.sqrt(l / nqubits))
        target_qubit = qubits[-(l + 1)]
        control_qubit = qubits[-l]

        # Implement the three-qubit gate (ii)_ℓ
        circuit.add(gates.CNOT(target_qubit, last_qubit))
        circuit.add(
            gates.RY(target_qubit, theta).controlled_by(control_qubit, last_qubit)
        )
        circuit.add(gates.CNOT(target_qubit, last_qubit))


def _add_wbd_gate(
    circuit: Circuit,
    first_register: List[int],
    second_register: List[int],
    nqubits: int,
    mqubits: int,
    weight: int,
) -> None:
    """In-place addition of a Weight Distribution Block (WBD) to ``circuit``.
    Implements the :math:`WBD^{n,m}_k` unitary from Definition 2 of the paper [2].
    Only acts on first_register and second_register, last k qubits
    Our circuit is mirrored, as paper [2] uses a top-bottom circuit <-> right-left bitstring convention
    """

    if mqubits > nqubits / 2:
        raise_error(ValueError, "``m`` must not be greater than ``n - m``.")

    # only acts on last k qubits
    first_register = first_register[-weight:]
    second_register = second_register[-weight:]

    # if m>k, m is truncated. Operations involving the most significant k-m digits can be removed

    # (1) Switching from unary encoding to one hot encoding
    circuit.add(gates.CNOT(q, q + 1) for q in first_register[-2::-1])

    # (2) Adding a supperposition of hamming weights into the second register
    # this can be optimized
    # We follow Figure 4, but adjust definition of xi and si (suffix sum) to match
    theta_gate = lambda qubit, theta: gates.RY(qubit, 2 * math.acos(theta))
    for l in range(weight, 0, -1):
        x = [
            math.comb(mqubits, elem) * math.comb(nqubits - mqubits, l - elem)
            for elem in range(l)
        ]
        s = math.comb(nqubits, l)
        circuit.add(
            theta_gate(second_register[-1], math.sqrt(x[0] / s)).controlled_by(
                first_register[-l]
            )
        )
        s -= x[0]
        for qubit in range(1, min(l, mqubits)):
            circuit.add(
                theta_gate(
                    second_register[-qubit - 1], math.sqrt(x[qubit] / s)
                ).controlled_by(first_register[-l], second_register[-qubit])
            )
            s -= x[qubit]

    # (3) Go back to unary encoding, undoing one hot encoding
    circuit.add(gates.CNOT(q, q + 1) for q in first_register[:-1])

    # (4) Substracting weight i in the first register from weight l in the second register.
    # fredkin, controlled swaps (decomposed into CNOT and Toffoli)
    fredkin = lambda control, s1, s2: (
        gates.CNOT(s2, s1),
        gates.TOFFOLI(control, s1, s2),
        gates.CNOT(s2, s1),
    )

    dif = max(0, weight - mqubits)
    for control in range(dif, weight):
        for q in range(control):
            circuit.add(
                fredkin(
                    second_register[control - dif],
                    first_register[control - q],
                    first_register[control - q - 1],
                )
            )
        circuit.add(gates.CNOT(second_register[control - dif], first_register[0]))


def _angle_mod_two_pi(angle: float) -> float:
    """Return angle mod 2pi."""
    return angle % (2 * np.pi)


def _binary_codewords(dims: int, backend: Optional[Backend] = None) -> ArrayLike:
    """Return a array of integers produced by `_binary_codewords_ehrlich(d)`.

    Adjusted so that the Hamming weight is strictly nondecreasing,
    and consecutive words have Hamming distance <= 2.

    """
    from qibo.quantum_info.utils import (  # pylint: disable=import-outside-toplevel
        hamming_distance,
        hamming_weight,
    )

    backend = _check_backend(backend)

    cw_binary = _binary_codewords_ehrlich(dims, backend=backend)

    cw = backend.cast(
        [int(bin_cw, 2) for bin_cw in cw_binary],
        dtype=_get_int_type(dims, backend=backend),
    )

    if (dims & (dims - 1)) != 0:
        n = int(backend.floor(backend.log2(dims)))
        dres = 2**n
        # split the list to fix the order
        cwres, cw = cw[dres:], cw[:dres]

        # keep weights for O(1) lookups
        weights = backend.cast(
            [hamming_weight(int(w)) for w in cw],
            dtype=_get_int_type(n, backend=backend),
        )

        # insert the remainder words at positions that preserve
        # strictly increasing weights and distance ≤ 2 to neighbors
        for word in cwres:

            hw = hamming_weight(int(word))

            inserted = False
            for i in range(len(cw) - 1):
                wi, wj = weights[i], weights[i + 1]
                if wi <= hw <= wj:
                    if (
                        hamming_distance(int(word), int(cw[i])) <= 2
                        and hamming_distance(int(word), int(cw[i + 1])) <= 2
                    ):
                        cw = backend.engine.insert(cw, i + 1, word, axis=0)
                        weights = backend.engine.insert(weights, i + 1, hw)
                        inserted = True
                        break
            if not inserted:
                # append if no suitable interior gap is found
                cw = backend.engine.hstack((cw, word))
                weights = backend.engine.hstack((weights, hw))

    return cw


def _binary_codewords_ehrlich(dims: int, backend: Optional[Backend] = None):
    """
    Yield fixed-width bitstrings representing integers 0..d-1, arranged so that
    Hamming weights are strictly nondecreasing. Uses at most hamming_weight(d) calls
    to `_ehrlich_codewords_up_to_k(k)`; when d is a power of two, exactly one call.
    """
    backend = _check_backend(backend)

    # Check power-of-two case: b == 2^k
    if (dims & (dims - 1)) == 0:
        k = dims.bit_length() - 1  # since 2^k has bit_length k+1
        # get_ehrlich_codewords(k) yields k-bit strings for 0..2^k-1 == 0..d-1
        yield from _ehrlich_codewords_up_to_k(k, False, backend=backend)
        return

    # General case
    n = int(max(1, backend.ceil(backend.log2(dims))))  # width so that 2^(n-1) < b < 2^n
    # Bits of b in most-significant-bit → least-significant-bit order
    bits = [(dims >> i) & 1 for i in range(n - 1, -1, -1)]
    prefix = ""
    flip = False
    for i, bit in enumerate(bits):
        k = n - 1 - i
        if bit == 1:
            # emit the full 0-prefixed block of width k, using Ehrlich order
            # flip every other time to keep consecutive codes close
            for suffix in _ehrlich_codewords_up_to_k(k, flip, backend=backend):
                yield prefix + "0" + suffix
            flip = not flip
            prefix += "1"
        else:
            prefix += "0"
    # no final singleton: we've generated exactly d bitstrings (0..d-1)


def _binary_encoder_hopf(
    data: ArrayLike,
    nqubits: int,
    complex_data: bool,
    backend: Optional[Backend] = None,
    **kwargs,
) -> Circuit:  # pylint: disable=unused-argument
    # TODO: generalize to complex-valued data
    backend = _check_backend(backend)

    dims = 2**nqubits

    base_strings = [f"{elem:0{nqubits}b}" for elem in range(dims)]
    base_strings = np.reshape(base_strings, (-1, 2))
    strings = [base_strings]
    for _ in range(nqubits - 1):
        base_strings = np.reshape(base_strings[:, 0], (-1, 2))
        strings.append(base_strings)
    strings = strings[::-1]

    targets_and_controls = []
    for pairs in strings:
        for pair in pairs:
            targets, controls, anticontrols = [], [], []
            for k, bits in enumerate(zip(pair[0], pair[1])):
                if bits == ("0", "0"):
                    anticontrols.append(k)
                elif bits == ("1", "1"):
                    controls.append(k)
                elif bits == ("0", "1"):
                    targets.append(k)
            targets_and_controls.append([targets, controls, anticontrols])

    circuit = Circuit(nqubits, **kwargs)
    for targets, controls, anticontrols in targets_and_controls:
        gate_list = []
        if len(anticontrols) > 0:
            gate_list.append(gates.X(qubit) for qubit in anticontrols)
        gate_list.append(
            gates.RY(targets[0], 0.0).controlled_by(*(controls + anticontrols))
        )
        if len(anticontrols) > 0:
            gate_list.append(gates.X(qubit) for qubit in anticontrols)
        circuit.add(gate_list)

    angles = _generate_rbs_angles(data, "tree", dims, backend=backend)
    circuit.set_parameters(2 * angles)

    return circuit


def _binary_encoder_hyperspherical(
    data: ArrayLike,
    nqubits: int,
    complex_data: bool,
    codewords: Optional[List[int]] = None,
    keep_antictrls: bool = False,
    backend: Optional[Backend] = None,
    **kwargs,
) -> Circuit:
    backend = _check_backend(backend)

    dims = len(data)

    if codewords is None:
        codewords = _binary_codewords(dims, backend=backend)

    circuit = Circuit(nqubits, **kwargs)
    if complex_data:
        circuit += _monotonic_hw_encoder_complex(
            codewords,
            data[codewords],
            nqubits,
            backend=backend,
            keep_antictrls=keep_antictrls,
            **kwargs,
        )
    else:
        circuit += _monotonic_hw_encoder_real(
            codewords,
            data[codewords],
            nqubits,
            backend=backend,
            keep_antictrls=keep_antictrls,
            **kwargs,
        )

    return circuit


def _ehrlich_algorithm(
    initial_string: ArrayLike, return_indices: bool = True
) -> List[str] | Tuple[List[str], List[List[int]]]:
    """Return list of bitstrings with mininal Hamming distance between consecutive strings.

    Based on the Gray code called Ehrlich algorithm. For more details, please see Ref. [1].

    Args:
        initial_string (ndarray): initial bitstring as an :math:`1`-dimensional array
            of size :math:`n`. All ones in the bitstring need to be consecutive.
            For instance, for :math:`n = 6` and :math:`k = 2`, the bistrings
            :math:`000011` and :math:`001100` are examples of acceptable inputs.
            In contrast, :math:`001001` is not an acceptable input.
        return_indices (bool, optional): if ``True``, returns the list of indices of
            qubits that act like controls and targets of the circuit to be created.
            Defaults to ``True``.

    Returns:
        list or tuple(list, list): If ``return_indices=False``, returns list containing
        sequence of bistrings in the order generated by the Gray code.
        If ``return_indices=True`` returns tuple with the aforementioned list and the list
        of control anf target qubits of gates to be implemented based on the sequence of
        bitstrings.

    References:
        1. R. M. S. Farias, T. O. Maciel, G. Camilo, R. Lin, S. Ramos-Calderer, and L. Aolita,
        *Quantum encoder for fixed-Hamming-weight subspaces*
        `Phys. Rev. Applied 23, 044014 (2025) <https://doi.org/10.1103/PhysRevApplied.23.044014>`_.
    """
    k = np.unique(initial_string, return_counts=True)
    if len(k[1]) == 1:  # pragma: no cover
        return ["".join([str(item) for item in initial_string])]

    k = k[1][1]
    n = len(initial_string)
    n_choose_k = int(binom(n, k))

    markers = _get_markers(initial_string, last_run=False)
    string = initial_string
    strings = ["".join(str(elem) for elem in string[::-1])]
    controls_and_targets = []
    for _ in range(n_choose_k - 1):
        string, markers, c_and_t = _get_next_bistring(string, markers, k)
        strings.append("".join(str(elem) for elem in string[::-1]))
        controls_and_targets.append(c_and_t)

    if return_indices:
        return strings, controls_and_targets

    return strings


def _ehrlich_codewords_up_to_k(
    up2k: int,
    reversed_list: bool = False,
    nqubits: Optional[int] = None,
    backend: Optional[Backend] = None,
):
    """
    Yield all bitstrings with monotonically changing Hamming weight from 0..up2k (or the reverse),
    such that consecutive strings have Hamming distance ≤ 2. Codewords for each weight are produced
    by `_ehrlich_algorithm(initial_string, False)` (assumed to return an iterable of bitstrings).

    Args:
        up2k (int): Length of the bitstrings and the maximum Hamming weight.
        reversed_list (bool, optional): If ``True``, generate from weight ``up2k`` down to :math:`0`.
            Otherwise, from :math:`0` up to ``up2k``.
        nqubits (int, optional): Number of qubits to be considered. If None, assumes that the
            number of qubits is ``up2k``, which is useful for the binary encoding case.
            Defaults to None.
        backend (:class:`qibo.backends.abstract.Backend`, optional): backend to be used
            in the execution. If ``None``, it uses the current backend. Defaults to ``None``.

    Yields:
        str: Bitstrings produced by concatenating sequences from Ehrlich's algorithm.
    """

    n = up2k if nqubits is None else nqubits
    if up2k > n:
        raise_error(ValueError, "up2k must be <= nqubits.")

    if n == 0:
        yield ""
        return

    def ones_left(k: int):
        return "1" * k + "0" * (n - k)

    def ones_right(k: int):
        return "0" * (n - k) + "1" * k

    from qibo.quantum_info.utils import (
        hamming_distance,  # pylint: disable=import-outside-toplevel
    )

    # starting boundary (weight 0 or weight up2k)
    if reversed_list:
        last_emitted = ones_left(up2k)
    else:
        last_emitted = "0" * n

    yield last_emitted

    # define weight iteration (exclude starting boundary)
    if reversed_list:
        weights = range(up2k - 1, -1, -1)
    else:
        weights = range(1, up2k + 1)

    for k in weights:
        # skip boundary already yielded
        if (not reversed_list and k == 0) or (reversed_list and k == up2k):
            continue  # pragma:no cover

        left = ones_left(k)
        right = ones_right(k)

        if hamming_distance(last_emitted, left) <= hamming_distance(
            last_emitted, right
        ):
            initial = left
        else:
            initial = right

        k_seq = _ehrlich_algorithm(
            backend.cast([int(b) for b in initial[::-1]], dtype=backend.int8),
            False,
        )

        if reversed_list:
            yield from reversed(k_seq)
            last_emitted = k_seq[0]
        else:
            yield from k_seq
            last_emitted = k_seq[-1]


def _gate_params(
    bsi: List[int], bsip1: List[int], keep_antictrls: bool = False
) -> Tuple[List[int], ...]:

    one_ind_bsi = {i for i in range(len(bsi)) if (bsi[i] == 1)}
    one_ind_bsip1 = {i for i in range(len(bsip1)) if (bsip1[i] == 1)}

    ctrls = one_ind_bsi.intersection(one_ind_bsip1)

    actrls = []
    if keep_antictrls:
        zero_ind_bsi = {i for i in range(len(bsi)) if (bsi[i] == 0)}
        zero_ind_bsip1 = {i for i in range(len(bsip1)) if (bsip1[i] == 0)}
        actrls = zero_ind_bsi.intersection(zero_ind_bsip1)
        ctrls = ctrls.union(actrls)

    in_bits = one_ind_bsi.difference(ctrls)
    out_bits = one_ind_bsip1.difference(ctrls)

    return list(in_bits), list(out_bits), list(ctrls), list(actrls)


def _generate_rbs_angles(
    data: ArrayLike,
    architecture: str,
    nqubits: Optional[int] = None,
    backend: Optional[Backend] = None,
) -> List[float]:
    """Generate list of angles for RBS gates based on ``architecture``.

    Args:
        data (ndarray, optional): :math:`1`-dimensional array of data to be loaded.
        architecture(str, optional): circuit architecture used for the unary loader.
            If ``diagonal``, uses a ladder-like structure.
            If ``tree``, uses a binary-tree-based structure.
            Defaults to ``tree``.
        nqubits (int): Number of qubits. To be used then ``architecture="tree"``.

    Returns:
        list: List of phases for RBS gates.
    """
    backend = _check_backend(backend)

    if architecture == "diagonal":
        phases = [
            backend.arctan2(backend.vector_norm(data[k + 1 :]), data[k])
            for k in range(len(data) - 2)
        ]
        phases.append(backend.arctan2(data[-1], data[-2]))

    if architecture == "tree":
        if nqubits is None:  # pragma: no cover
            raise_error(
                TypeError,
                '``nqubits`` must be specified when ``architecture=="tree"``.',
            )

        j_max = int(nqubits / 2)

        r_array = np.zeros(nqubits - 1, dtype=float)
        phases = np.zeros(nqubits - 1, dtype=float)
        for j in range(1, j_max + 1):
            r_array[j_max + j - 2] = math.sqrt(
                data[2 * j - 1] ** 2 + data[2 * j - 2] ** 2
            )
            theta = math.acos(data[2 * j - 2] / r_array[j_max + j - 2])
            if data[2 * j - 1] < 0.0:
                theta = 2 * math.pi - theta
            phases[j_max + j - 2] = theta

        for j in range(j_max - 1, 0, -1):
            r_array[j - 1] = math.sqrt(r_array[2 * j] ** 2 + r_array[2 * j - 1] ** 2)
            phases[j - 1] = math.acos(r_array[2 * j - 1] / r_array[j - 1])

    phases = backend.cast(phases, dtype=phases[0].dtype)

    return phases


def _generate_rbs_pairs(
    nqubits: int, architecture: str, **kwargs
) -> Tuple[Circuit, List[List[int]]]:
    """Generate list of indexes representing the RBS connections.

    Creates circuit with all RBS initialised with 0.0 phase.

    Args:
        nqubits (int): number of qubits.
        architecture(str, optional): circuit architecture used for the unary loader.
            If ``diagonal``, uses a ladder-like structure.
            If ``tree``, uses a binary-tree-based structure.
            Defaults to ``tree``.
        kwargs (dict, optional): Additional arguments used to initialize a Circuit object.
            For details, see the documentation of :class:`qibo.models.circuit.Circuit`.

    Returns:
        (:class:`qibo.models.circuit.Circuit`, list): Circuit composed of
        :class:`qibo.gates.gates.RBS` and list of indexes of target qubits per depth.
    """

    if architecture == "diagonal":
        pairs_rbs = np.arange(nqubits)
        pairs_rbs = [[pair] for pair in zip(pairs_rbs[:-1], pairs_rbs[1:])]

    if architecture == "tree":
        pairs_rbs = [[(0, int(nqubits / 2))]]
        indexes = list(pairs_rbs[0][0])
        for depth in range(2, int(math.log2(nqubits)) + 1):
            pairs_rbs_per_depth = [
                [(index, index + int(nqubits / 2**depth)) for index in indexes]
            ]
            pairs_rbs += pairs_rbs_per_depth
            indexes = list(np.array(pairs_rbs_per_depth).flatten())

    pairs_rbs = [
        [(nqubits - 1 - a, nqubits - 1 - b) for a, b in row] for row in pairs_rbs
    ]

    circuit = Circuit(nqubits, **kwargs)
    for row in pairs_rbs:
        for pair in row:
            circuit.add(gates.RBS(*pair, 0.0, trainable=True))

    return circuit, pairs_rbs


def _get_gate(
    qubits_in: List[int],
    qubits_out: List[int],
    controls: List[int],
    theta: float,
    phi: float,
    complex_data: bool = False,
) -> List[Gate]:
    """Return gate(s) necessary to encode a complex amplitude in a given computational basis state,
    given the computational basis state used to encode the previous amplitude.

    Information about computational basis states in question are contained
    in the indexing of ``qubits_in``, ``qubits_out``, and ``controls``.

    Args:
        qubits_in (list): list of qubits with ``in`` label.
        qubits_out (list): list of qubits with ``out`` label.
        controls (list): list of qubits that control the resulting gate.
        theta (float): first phase, used to encode the ``abs`` of amplitude.
        phi (float): second phase, used to encode the complex phase of amplitude.
        complex_data (bool): if ``True``, uses :class:`qibo.gates.U3` to as basis gate.
            If ``False``, uses :class:`qibo.gates.RY` as basis gate. Defaults to ``False``.

    Returns:
        List[:class:`qibo.gates.Gate`]: gate(s) to be added to circuit.

    References:
        1. R. M. S. Farias, T. O. Maciel, G. Camilo, R. Lin, S. Ramos-Calderer, and L. Aolita,
        *Quantum encoder for fixed-Hamming-weight subspaces*
        `Phys. Rev. Applied 23, 044014 (2025) <https://doi.org/10.1103/PhysRevApplied.23.044014>`_.
    """
    if len(qubits_in) == 0 and len(qubits_out) == 1:  # pragma: no cover
        gate_list = (
            gates.U3(*qubits_out, 2 * theta, 2 * phi, 0.0).controlled_by(*controls)
            if complex_data
            else gates.RY(*qubits_out, 2 * theta).controlled_by(*controls)
        )
        gate_list = [gate_list]
    elif len(qubits_in) == 1 and len(qubits_out) == 1:
        ## chooses best combination of complex RBS gate
        ## given number of controls and if data is real or complex
        gate_list = []
        gate = gates.RBS(*qubits_in, *qubits_out, theta)
        if len(controls) > 0:
            gate = gate.controlled_by(*controls)
        gate_list.append(gate)

        if complex_data:
            gate = [gates.RZ(*qubits_in, -phi), gates.RZ(*qubits_out, phi)]
            if len(controls) > 0:
                gate = [g.controlled_by(*controls) for g in gate]
            gate_list.extend(gate)

    else:  # pragma: no cover
        # Important for future sparse encoder
        gate_list = [
            gates.GeneralizedRBS(
                list(qubits_in), list(qubits_out), theta, phi
            ).controlled_by(*controls)
        ]

    return gate_list


def _get_gate_sparse(
    distance: int,
    difference: int,
    touched_qubits: Union[List[int], Tuple[int, ...]],
    complex_data: bool,
    controls: Union[List[int], Tuple[int, ...]],
    hw_0: int,
    hw_1: int,
    theta: float,
    phi: float,
    backend: Optional[Backend] = None,
) -> Gate:
    backend = _check_backend(backend)
    if distance == 1:
        qubit = int(backend.where(difference == 1)[0][0])
        if qubit not in touched_qubits:
            touched_qubits.append(qubit)
        gate = (
            gates.U3(qubit, 2 * theta, 2 * phi, 0.0).controlled_by(*controls)
            if complex_data
            else gates.RY(qubit, 2 * theta).controlled_by(*controls)
        )
    elif distance == 2 and hw_0 == hw_1:
        qubits = [
            int(np.where(difference == -1)[0][0]),
            int(np.where(difference == 1)[0][0]),
        ]
        touched_qubits += list(set(qubits) - set(touched_qubits))
        qubits_in = [int(qubits[0])]
        qubits_out = [int(qubits[1])]
        qubits = [np.where(difference == -1)[0][0], np.where(difference == 1)[0][0]]

        gate = _get_gate(
            qubits_in,
            qubits_out,
            controls,
            theta,
            phi,
            complex_data,
        )
    else:
        qubits = [np.where(difference == -1)[0], np.where(difference == 1)[0]]
        for row in qubits:
            row = [int(elem) for elem in row]
            touched_qubits += list(set(row) - set(touched_qubits))
        qubits_in = [int(qubit) for qubit in qubits[0]]
        qubits_out = [int(qubit) for qubit in qubits[1]]
        gate = gates.GeneralizedRBS(qubits_in, qubits_out, theta, -phi).controlled_by(
            *controls
        )

    return gate


def _get_int_type(x: int, backend: Optional[Backend] = None) -> DTypeLike:
    # Candidates in increasing size of memory usage

    backend = _check_backend(backend)
    int_types = (
        backend.int8,
        backend.int16,
        backend.int32,
        backend.int64,
    )
    for dt in int_types:
        if abs(x) <= backend.engine.iinfo(dt).max:
            return backend.engine.dtype(dt)

    raise_error(
        ValueError,
        f"``|x|`` must not be greater than {backend.engine.iinfo(backend.int64).max}.",
    )


def _get_markers(bitstring: ArrayLike, last_run: bool = False) -> Set[int]:
    """Subroutine of the Ehrlich algorithm."""
    nqubits = len(bitstring)
    markers = [len(bitstring) - 1]
    for ind, value in zip(range(nqubits - 2, -1, -1), bitstring[::-1][1:]):
        if value == bitstring[-1]:
            markers.append(ind)
        else:
            break

    markers = set(markers)

    if not last_run:
        markers = set(range(nqubits)) - markers

    return markers


def _get_next_bistring(
    bitstring: ArrayLike, markers: Set[int], hamming_weight: int
) -> Tuple[ArrayLike, Set[int], List[List[int]]]:
    """Subroutine of the Ehrlich algorithm."""
    if len(markers) == 0:  # pragma: no cover
        return bitstring

    new_bitstring = np.copy(bitstring)

    nqubits = len(new_bitstring)

    indexes = np.argsort(bitstring)
    zeros, ones = np.sort(indexes[:-hamming_weight]), np.sort(indexes[-hamming_weight:])

    max_index = max(markers)
    nearest_one = ones[ones > max_index]
    nearest_one = None if len(nearest_one) == 0 else nearest_one[0]
    if new_bitstring[max_index] == 0 and nearest_one is not None:
        new_bitstring[max_index] = 1
        new_bitstring[nearest_one] = 0
    else:
        farthest_zero = zeros[zeros > max_index]
        if nearest_one is not None:
            farthest_zero = farthest_zero[farthest_zero < nearest_one]
        farthest_zero = farthest_zero[-1]
        new_bitstring[max_index] = 0
        new_bitstring[farthest_zero] = 1

    markers.remove(max_index)
    last_run = _get_markers(new_bitstring, last_run=True)
    markers = markers | (set(range(max_index + 1, nqubits)) - last_run)

    new_ones = np.argsort(new_bitstring)[-hamming_weight:]

    controls = list({int(elem) for elem in ones} & {int(elem) for elem in new_ones})
    difference = new_bitstring - bitstring
    qubits = [np.where(difference == -1)[0][0], np.where(difference == 1)[0][0]]

    return new_bitstring, markers, [qubits, controls]


def _get_phase_gate_correction(
    last_string: Union[ArrayLike, str], phase: float
) -> Gate:
    """Return final gate of HW-k circuits that encode complex data."""

    # to avoid circular import error
    from qibo.quantum_info.utils import (  # pylint: disable=import-outside-toplevel
        hamming_weight,
    )

    if isinstance(last_string, str):
        last_string = np.asarray(list(last_string), dtype=int)

    last_weight = hamming_weight(last_string)
    last_ones = np.argsort(last_string)
    last_zero = int(last_ones[0])
    last_controls = [int(qubit) for qubit in last_ones[-last_weight:]]

    # adding an RZ gate to correct the phase of the last amplitude encoded
    return gates.RZ(last_zero, 2 * phase).controlled_by(*last_controls)


def _get_phase_gate_correction_sparse(
    last_string: Union[ArrayLike, str],
    second_to_last_string: Union[ArrayLike, str],
    nqubits: int,
    last_data: complex,
    second_to_last_data: complex,
    circuit: Circuit,
    phis: ArrayLike,
):
    from qibo.quantum_info.utils import (  # pylint: disable=import-outside-toplevel
        hamming_weight,
    )

    hw_0 = hamming_weight(second_to_last_string)
    hw_1 = hamming_weight(last_string)
    if hw_1 == nqubits and hw_0 == nqubits - 1:
        phi = _angle_mod_two_pi(np.angle(last_data) - np.angle(second_to_last_data))
        lamb = _angle_mod_two_pi(
            -(np.angle(second_to_last_data) + np.angle(last_data))
            + 2 * np.sum(phis[:-2])
        )
        _gate = circuit.queue[-1]
        gate = gates.U3(
            *_gate.target_qubits, _gate.init_kwargs["theta"], phi, lamb
        ).controlled_by(*_gate.control_qubits)

        new_queue = circuit.queue[:-1] + [gate]

        return new_queue

    if hw_1 == nqubits:  # pragma: no cover
        first_one = np.argsort(second_to_last_string)[0]
        other_ones = list(set(range(nqubits)) ^ {first_one})
        gate = gates.RZ(first_one, 2 * phis[-1]).controlled_by(*other_ones)
    else:
        gate = _get_phase_gate_correction(last_string, phis[-1])

    return gate


def _monotonic_hw_encoder_complex(
    codewords: List[int],
    data: ArrayLike,
    nqubits: int,
    keep_antictrls: bool = False,
    backend: Optional[Backend] = None,
    **kwargs,
) -> Circuit:
    """Implements Algorithm 4 from  Ref. [1].

    Args:
        codewords (int): list of codewords. Assumed ordered such that their hamming-weights are
            non-decreasing.
        data (complex): data to be encoded. assumed to have the same length as the list of codewords
        nqubits (int, optional): total number of qubits in the system.
            To be used when :math:`b_j` are integers. If :math:`b_j` are strings and
            ``nqubits`` is ``None``, defaults to the length of the strings :math:`b_{j}`.
            Defaults to ``None``.
        backend (:class:`qibo.backends.abstract.Backend`, optional): backend to be used
            in the execution. If ``None``, it uses the current backend. Defaults to ``None``.
        keep_antictrls (bool, optional): If ``True``, we don't simplify the anti-controls
            when placing the RBS gates. For details, see [1].
        kwargs (dict, optional): Additional arguments used to initialize a Circuit object.
            For details, see the documentation of :class:`qibo.models.circuit.Circuit`.

    Returns:
        :class:`qibo.models.circuit.Circuit`: Circuit that loads :math:`\\mathbf{x}`.

    References:
        1. R. M. S. Farias, T. O. Maciel, G. Camilo, R. Lin, S. Ramos-Calderer, and L. Aolita,
        *Quantum encoder for fixed-Hamming-weight subspaces*
        `Phys. Rev. Applied 23, 044014 (2025) <https://doi.org/10.1103/PhysRevApplied.23.044014>`_.
    """
    dims = len(data)

    # the angles phi wil be computed on the fly,
    # so we don't have to keep them in memory
    def phis(x, k):
        cumsum = 0.0
        for i in range(k + 1):
            val = (-backend.angle(x[i]) + cumsum) % (2.0 * np.pi)
            cumsum += val
        return val

    circuit = Circuit(nqubits, **kwargs)

    bsi = [int(b) for b in format(codewords[0], "0{}b".format(nqubits))]
    for i in range(1, dims - 1):
        bsip1 = [int(b) for b in format(codewords[i], "0{}b".format(nqubits))]

        in_bits, out_bits, ctrls, actrls = _gate_params(bsi, bsip1, keep_antictrls)

        theta = backend.arctan2(backend.vector_norm(abs(data[i:]), 2), abs(data[i - 1]))

        if keep_antictrls:
            circuit.add([gates.X(ac) for ac in actrls])

        if len(in_bits) + len(out_bits) == 1:
            targ_gate = out_bits if len(out_bits) > 0 else in_bits
            sign_angle = 1.0 if len(out_bits) > 0 else -1.0
            circuit.add(
                gates.RY(*targ_gate, 2.0 * sign_angle * theta).controlled_by(*ctrls)
            )
            circuit.add(
                gates.RZ(
                    *targ_gate, 2.0 * sign_angle * phis(data, i - 1)
                ).controlled_by(*ctrls)
            )
        else:
            circuit.add(
                gates.GeneralizedRBS(
                    in_bits, out_bits, theta, -phis(data, i - 1)
                ).controlled_by(*ctrls)
            )
        if keep_antictrls:
            circuit.add([gates.X(ac) for ac in actrls])

        bsi = bsip1
    bsip1 = [int(b) for b in format(codewords[dims - 1], "0{}b".format(nqubits))]

    in_bits, out_bits, ctrls, actrls = _gate_params(bsi, bsip1, keep_antictrls)

    theta = backend.arctan2(abs(data[-1]), abs(data[-2]))

    if keep_antictrls:
        circuit.add([gates.X(ac) for ac in actrls])

    if len(in_bits) + len(out_bits) == 1:
        phil = (0.5 * (backend.angle(data[-1]) - backend.angle(data[-2]))) % (
            2.0 * np.pi
        )
        lambdal = (
            0.5 * (-backend.angle(data[-1]) + backend.angle(data[-2]))
            + phis(data, dims - 2)
        ) % (2.0 * np.pi)

        targ_gate = out_bits if len(out_bits) > 0 else in_bits
        sign_angle = 1.0 if len(out_bits) > 0 else -1.0
        circuit.add(
            gates.U3(
                *targ_gate,
                2.0 * sign_angle * theta,
                2.0 * sign_angle * phil,
                2.0 * sign_angle * lambdal,
            ).controlled_by(*ctrls)
        )
        if keep_antictrls:
            circuit.add([gates.X(ac) for ac in actrls])
    else:
        circuit.add(
            gates.GeneralizedRBS(
                in_bits, out_bits, theta, -phis(data, dims - 2)
            ).controlled_by(*ctrls)
        )

        if keep_antictrls:
            circuit.add([gates.X(ac) for ac in actrls])

        ctrls = [i for i in range(len(bsip1)) if (bsip1[i] == 1)]
        in_bits = [i for i in range(len(bsip1)) if (bsip1[i] == 0)]

        circuit.add(gates.X(in_bits[0]).controlled_by(*ctrls))
        if keep_antictrls:
            circuit.add([gates.X(ac) for ac in in_bits[1:]])
            circuit.add(
                gates.U1(in_bits[0], -phis(data, dims - 1)).controlled_by(
                    *ctrls + in_bits[1:]
                )
            )
            circuit.add([gates.X(ac) for ac in in_bits[1:]])
        else:
            circuit.add(
                gates.U1(in_bits[0], -phis(data, dims - 1)).controlled_by(*ctrls)
            )
        circuit.add(gates.X(in_bits[0]).controlled_by(*ctrls))

    return circuit


def _monotonic_hw_encoder_real(
    codewords: List[int],
    data: ArrayLike,
    nqubits: int,
    keep_antictrls: bool = False,
    backend: Optional[Backend] = None,
    **kwargs,
) -> Circuit:
    """Implements Algorithm 3 from Ref. [1].

    Args:
        codewords (int): list of codewords. Assumed ordered such that their hamming-weights are
            non-decreasing.
        data (float): data to be encoded. assumed to have the same length as the list of codewords
        nqubits (int, optional): total number of qubits in the system.
            To be used when :math:`b_j` are integers. If :math:`b_j` are strings and
            ``nqubits`` is ``None``, defaults to the length of the strings :math:`b_{j}`.
            Defaults to ``None``.
        backend (:class:`qibo.backends.abstract.Backend`, optional): backend to be used
            in the execution. If ``None``, it uses the current backend. Defaults to ``None``.
        keep_antictrls (bool, optional): If ``True``, we don't simplify the anti-controls
            when placing the RBS gates. For details, see [1].
        kwargs (dict, optional): Additional arguments used to initialize a Circuit object.
            For details, see the documentation of :class:`qibo.models.circuit.Circuit`.

    Returns:
        :class:`qibo.models.circuit.Circuit`: Circuit that loads :math:`\\mathbf{x}`.

    References:
        1. R. M. S. Farias, T. O. Maciel, G. Camilo, R. Lin, S. Ramos-Calderer, and L. Aolita,
        *Quantum encoder for fixed-Hamming-weight subspaces*
        `Phys. Rev. Applied 23, 044014 (2025) <https://doi.org/10.1103/PhysRevApplied.23.044014>`_.
    """
    dims = len(data)

    circuit = Circuit(nqubits, **kwargs)

    bsi = [int(b) for b in format(codewords[0], "0{}b".format(nqubits))]
    for i in range(1, dims - 1):
        bsip1 = [int(b) for b in format(codewords[i], "0{}b".format(nqubits))]

        in_bits, out_bits, ctrls, actrls = _gate_params(bsi, bsip1, keep_antictrls)

        theta = backend.arctan2(backend.vector_norm(data[i:], 2), data[i - 1])

        if keep_antictrls:
            circuit.add([gates.X(ac) for ac in actrls])

        targ_bits = len(in_bits) + len(out_bits)
        if targ_bits == 1:
            targ_gate = out_bits if len(out_bits) > 0 else in_bits
            sign_angle = 1.0 if len(out_bits) > 0 else -1.0
            circuit.add(
                gates.RY(*targ_gate, 2.0 * sign_angle * theta).controlled_by(*ctrls)
            )
        elif targ_bits == 2:
            circuit.add(gates.RBS(*in_bits, *out_bits, theta).controlled_by(*ctrls))
        else:
            circuit.add(
                gates.GeneralizedRBS(
                    tuple(in_bits), tuple(out_bits), theta=theta
                ).controlled_by(*ctrls)
            )

        if keep_antictrls:
            circuit.add([gates.X(ac) for ac in actrls])

        bsi = bsip1
    bsip1 = [int(b) for b in format(codewords[dims - 1], "0{}b".format(nqubits))]

    in_bits, out_bits, ctrls, actrls = _gate_params(bsi, bsip1, keep_antictrls)

    theta = backend.arctan2(data[-1], data[-2])

    if keep_antictrls:
        circuit.add([gates.X(ac) for ac in actrls])

    targ_bits = len(in_bits) + len(out_bits)
    if targ_bits == 1:
        targ_gate = out_bits if len(out_bits) > 0 else in_bits
        sign_angle = 1.0 if len(out_bits) > 0 else -1.0
        circuit.add(
            gates.RY(*targ_gate, 2.0 * sign_angle * theta).controlled_by(*ctrls)
        )

    elif targ_bits == 2:
        circuit.add(gates.RBS(*in_bits, *out_bits, theta).controlled_by(*ctrls))
    else:
        circuit.add(
            gates.GeneralizedRBS(
                tuple(in_bits), tuple(out_bits), theta=theta
            ).controlled_by(*ctrls)
        )

    if keep_antictrls:
        circuit.add([gates.X(ac) for ac in actrls])

    return circuit


def _next_nearest_layer(
    nqubits: int,
    gate: Gate,
    parameters: Union[List[int], Tuple[int, ...]],
    closed_boundary: bool,
    **kwargs,
) -> Circuit:
    """Create entangling layer with next-nearest-neighbour connectivity."""
    circuit = Circuit(nqubits, **kwargs)
    circuit.add(
        _parametrized_two_qubit_gate(gate, qubit, qubit + 2, parameters)
        for qubit in range(nqubits - 2)
    )

    if closed_boundary:
        circuit.add(_parametrized_two_qubit_gate(gate, nqubits - 1, 0, parameters))

    return circuit


def _non_trivial_layers(
    nqubits: int,
    architecture: str = "pyramid",
    entangling_gate: Union[str, Gate] = "RBS",
    closed_boundary: bool = False,
    **kwargs,
) -> Circuit:
    """Create more intricate entangling layers of different shapes.

    Args:
        nqubits (int): number of qubits.
        architecture (str, optional): Architecture of the entangling layer.
            In alphabetical order, options are ``"next_nearest"``, ``"pyramid"``,
            ``"v"``, and ``"x"``. The ``"x"`` architecture is only defined for
            an even number of qubits. Defaults to ``"pyramid"``.
        entangling_gate (str or :class:`qibo.gates.Gate`, optional): Two-qubit gate to be used
            in the entangling layer. If ``entangling_gate`` is a parametrized gate,
            all phases are initialized as :math:`0.0`. Defaults to  ``"CNOT"``.
        closed_boundary (bool, optional): If ``True`` and ``architecture="next_nearest"``,
            adds a closed-boundary condition to the entangling layer. Defaults to ``False``.
        kwargs (dict, optional): Additional arguments used to initialize a Circuit object.
            For details, see the documentation of :class:`qibo.models.circuit.Circuit`.

    Returns:
        :class:`qibo.models.circuit.Circuit`: Circuit containing layer of two-qubit gates.
    """

    gate = (
        getattr(gates, entangling_gate)
        if isinstance(entangling_gate, str)
        else entangling_gate
    )

    parameters = list(signature(gate).parameters)
    parameters = (0.0,) * (len(parameters) - 3) if len(parameters) > 2 else None

    if architecture == "next_nearest":
        return _next_nearest_layer(nqubits, gate, parameters, closed_boundary, **kwargs)

    if architecture == "v":
        return _v_layer(nqubits, gate, parameters, **kwargs)

    if architecture == "x":
        return _x_layer(nqubits, gate, parameters, **kwargs)

    return _pyramid_layer(nqubits, gate, parameters, **kwargs)


def _parametrized_two_qubit_gate(
    gate: Gate,
    q0: int,
    q1: int,
    params: Optional[Union[List[int], Tuple[float, ...]]] = None,
) -> Gate:
    """Return two-qubit gate initialized with or without phases."""
    if params is not None:
        return gate(q0, q1, *params)

    return gate(q0, q1)


def _perm_column_ops(
    indices: List[int],
    n: int,
    backend: Optional[Backend] = None,
):
    """Return (ell, gate_list) performing duplicate‑col removal + compaction."""
    backend = _check_backend(backend)
    # flatten the (x_0, x_1),...(x_{2m-2}, x_{2m-1})
    # We construct a matrix composed of
    # xj to track the changes in xj where xj,k represents the k-th bit and the
    # bits are arranged from the least significant bit to the most significant bit
    indices = list(sum(indices, ()))
    A = []
    for x in indices:
        bits = []
        for k in range(n):
            bits.append((x >> k) & 1)
        A.append(bits)
    A = backend.cast(A, dtype=backend.int8)
    ncols = A.shape[1]
    # initialize the list of gates
    qgates = []

    # number of non-zero columns
    ell = 0
    flag = backend.zeros(n, dtype=int)
    for idxj in range(ncols):
        if any(elem != 0 for elem in A[:, idxj]):
            ell += 1
            flag[idxj] = 1

            # look for columns that are equal to A[:,idxj]
            for idxk in range(idxj + 1, ncols):
                if backend.array_equal(A[:, idxj], A[:, idxk]):
                    qgates.append(gates.CNOT(n - idxj - 1, n - idxk - 1))
                    # this should transform the k-th column into an all-zero column
                    A[:, idxk] = 0

    # Now, we need to swap the ell non-zero columns to the first ell columns
    for idxk in range(ell, ncols):
        if not backend.array_equal(A[:, idxk], backend.zeros_like(A[:, idxk])):
            for k in range(len(flag)):
                if flag[k] == 0:
                    flag[k] = 1
                    flag[idxk] = 0

                    qgates.append(gates.SWAP(n - idxk - 1, n - k - 1))

                    bits = A[:, idxk].copy()
                    A[:, idxk] = A[:, k]
                    A[:, k] = bits
                    break

    return ell, qgates, A


def _perm_pair_flip_ops(
    n: int, m: int, backend: Optional[Backend] = None
) -> List[Gate]:
    """Implement σ_{i,2} as X fan‑in + MCX + X fan‑out."""
    backend = _check_backend(backend)
    # let us flip the first qubit when the last {int(n-math.log2(2*m))} qubits are all in the state |0⟩
    prefix = int(backend.ceil(backend.log2(2 * m)))
    x_qubits, controls = range(prefix, n), range(n - prefix)
    qgates = [gates.X(n - q - 1) for q in x_qubits]
    qgates.append(gates.X(n - 1).controlled_by(*controls))  # flip qubit 0
    qgates.extend(gates.X(n - q - 1) for q in x_qubits)

    return qgates


def _perm_row_ops(
    A: ArrayLike, ell: int, m: int, n: int, backend: Optional[Backend] = None
) -> List[Gate]:
    """Return gates that reduce all rows after row0 to target form."""
    backend = _check_backend(backend)

    log2m = int(backend.log2(2 * m))
    atilde = backend.cast(
        [[(x >> k) & 1 for k in range(n)] for x in range(2 * m)], dtype=int
    )

    qgates = []
    nrows = A.shape[0]
    ncols = A.shape[1]
    # Start with the first row (indexed as row 0)
    for k in range(ncols):
        # If we find a0,k = 1 for any 0 <= k <= n−1
        if A[0, k] == 1:
            qgates.append(gates.X(n - k - 1))
            A[:, k] = (A[:, k] + 1) % 2

    for j in range(1, nrows):
        flag = False
        for k in range(log2m, ncols):
            if A[j, k] != 0:
                flag = True
                break

        if not flag:
            # There is no element b_{j},k != 0 for k > {log2m-1}"
            ctrls = [n - l - 1 for l in range(ncols) if A[j, l] != 0]
            qgates.append(gates.X(n - log2m - 1).controlled_by(*ctrls))
            ctrls = [l for l in range(ncols) if A[j, l] == 1]

            # check whether the gate is applied on other rows
            for l in range(j, nrows):
                if all(elem == 1 for elem in A[l, ctrls]):
                    A[l, log2m] = (A[l, log2m] + 1) % 2

        # There is always an element b_{j},k != 0 for k > {log2m-1}
        for k in range(log2m, ncols):
            if A[j, k] != 0:
                # Element b_{j},{k} != 0, {k} > {log2m-1}"
                for kprime in range(ell):
                    # There is a typo in the paper
                    # b_{j},{kprime} should be different from Ã_{j},{kprime} not Ã_{j},{k}
                    if kprime != k and A[j, kprime] != atilde[j, kprime]:
                        qgates.append(gates.CNOT(n - k - 1, n - kprime - 1))
                        # check whether the gate is applied on other rows
                        for l in range(nrows):
                            if A[l, k] == 1:
                                A[l, kprime] = (A[l, kprime] + 1) % 2

                # Let us clean the element b_{j},{k}

                # There is another typo in the paper
                # the control qubits for this gate correspond to the non-zero elements in row j of matrix A, not Ã
                ctrls = [n - l - 1 for l in range(k) if A[j, l] != 0]
                qgates.append(gates.X(n - k - 1).controlled_by(*ctrls))
                ctrls = [l for l in range(k) if A[j, l] != 0]

                # check whether the gate is applied on other rows
                for l in range(nrows):
                    if all(elem == 1 for elem in A[l, ctrls]):
                        A[l, k] = (A[l, k] + 1) % 2

    return qgates


def _pyramid_layer(
    nqubits: int, gate: Gate, parameters: Union[List[int], Tuple[int, ...]], **kwargs
) -> Circuit:
    """Create entangling layer in triangular shape."""
    _, pairs_gates = _generate_rbs_pairs(nqubits, architecture="diagonal")
    pairs_gates = pairs_gates[::-1]

    circuit = Circuit(nqubits, **kwargs)
    circuit.add(
        _parametrized_two_qubit_gate(gate, pair[0][1], pair[0][0], parameters)
        for pair in pairs_gates
    )
    circuit.add(
        _parametrized_two_qubit_gate(gate, pair[0][1], pair[0][0], parameters)
        for k in range(1, len(pairs_gates))
        for pair in pairs_gates[:-k]
    )

    return circuit


def _sort_data_sparse(
    data: ArrayLike, nqubits: int, backend: Backend
) -> Tuple[ArrayLike, ArrayLike]:
    from qibo.quantum_info.utils import (  # pylint: disable=import-outside-toplevel
        hamming_weight,
    )

    # TODO: Fix this mess with qibo native data types
    try:
        test_dtype = bool("int" in str(data[0][0].dtype))
    except AttributeError:
        test_dtype = bool("int" in str(type(data[0][0])))

    _data = [(f"{row[0]:0{nqubits}b}", row[1]) for row in data] if test_dtype else data

    _data = sorted(_data, key=lambda x: hamming_weight(x[0]))

    bitstrings_sorted, data_sorted = zip(*_data)
    bitstrings_sorted = [
        np.array(list(string)).astype(int) for string in bitstrings_sorted
    ]

    bitstrings_sorted = backend.cast(bitstrings_sorted, dtype=backend.int8)
    data_sorted = backend.cast(data_sorted, dtype=data_sorted[0].dtype)

    return data_sorted, bitstrings_sorted


def _sparse_encoder_farias(
    data: ArrayLike, nqubits: int, backend: Optional[Backend] = None, **kwargs
) -> Circuit:
    """Create circuit that encodes :math:`1`-dimensional data in a subset of amplitudes
    of the computational basis.

    Consider a sparse-access model, where for a data vector
    :math:`\\mathbf{x} \\in \\mathbb{C}^{d}`, with :math:`d = 2^{n}` and
    :math:`s` non-zero amplitudes, one has access to the data vector
    :math:`\\mathbf{y}` of the form

    .. math::
        \\mathbf{y} = \\left\\{ (b_{1}, x_{1}), \\, \\dots, \\, (b_{s}, x_{s}) \\right\\} \\, ,


    where :math:`\\{x_{j}\\}_{j\\in[s]}` are the non-zero components of :math:`\\mathbf{x}`
    and :math:`\\{b_{j}\\}_{j\\in[s]}` is the set of addresses associated with these values.
    Then, this function generates a quantum circuit  :math:`s\\text{-}\\mathrm{Load}` that encodes
    :math:`\\mathbf{x}` in the amplitudes of an :math:`n`-qubit quantum state as

    .. math::
        s\\text{-}\\mathrm{Load}(\\mathbf{y}) \\, \\ket{0}^{\\otimes \\, n} = \\sum_{j\\in[s]} \\,
            \\frac{x_{j}}{\\|\\mathbf{x}\\|_{2}} \\, \\ket{b_{j}} \\, ,

    where :math:`\\|\\cdot\\|_{2}` is the l2-norm.

    Resulting circuit parametrizes ``data`` in hyperspherical coordinates
    in the :math:`(2^{n} - 1)`-unit sphere.


    Args:
        data (ndarray or list or zip): sequence of tuples of the form :math:`(b_{j}, x_{j})`.
            The addresses :math:`b_{j}` can be either integers or in bitstring
            format of size :math:`n`.
        nqubits (int, optional): total number of qubits in the system.
            To be used when :math:`b_j` are integers. If :math:`b_j` are strings and
            ``nqubits`` is ``None``, defaults to the length of the strings :math:`b_{j}`.
            Defaults to ``None``.
        backend (:class:`qibo.backends.abstract.Backend`, optional): backend to be used
            in the execution. If ``None``, it uses the current backend. Defaults to ``None``.
        kwargs (dict, optional): Additional arguments used to initialize a Circuit object.
            For details, see the documentation of :class:`qibo.models.circuit.Circuit`.

    Returns:
        :class:`qibo.models.circuit.Circuit`: Circuit that loads sparse :math:`\\mathbf{x}`.

    References:
        1. R. M. S. Farias, T. O. Maciel, G. Camilo, R. Lin, S. Ramos-Calderer, and L. Aolita,
        *Quantum encoder for fixed-Hamming-weight subspaces*,
        `Phys. Rev. Applied 23, 044014 (2025) <https://doi.org/10.1103/PhysRevApplied.23.044014>`_.

        2. `Hyperpherical coordinates <https://en.wikipedia.org/wiki/N-sphere>`_.
    """
    from qibo.models.encodings import comp_basis_encoder  # pylint: disable=C0415
    from qibo.quantum_info.utils import (  # pylint: disable=import-outside-toplevel
        hamming_distance,
        hamming_weight,
    )

    backend = _check_backend(backend)

    if isinstance(data, zip):
        data = list(data)

    # TODO: Fix this mess with qibo native dtypes
    try:
        type_test = bool("int" in str(data[0][0].dtype))
    except AttributeError:
        type_test = bool("int" in str(type(data[0][0])))

    if type_test and nqubits is None:
        raise_error(
            ValueError,
            "``nqubits`` must be specified when computational basis states are "
            + "indidated by integers.",
        )

    if isinstance(data[0][0], str) and nqubits is None:
        nqubits = len(data[0][0])

    _data_test = data[0][1]
    _data_test = (
        _data_test.dtype if "array" in str(type(_data_test)) else type(_data_test)
    )

    complex_data = bool("complex" in str(_data_test))

    # sort data by HW of the bitstrings
    data_sorted, bitstrings_sorted = _sort_data_sparse(data, nqubits, backend)
    # calculate phases
    _data_sorted = backend.abs(data_sorted) if complex_data else data_sorted
    thetas = _generate_rbs_angles(
        _data_sorted, architecture="diagonal", backend=backend
    )
    phis = backend.zeros(len(thetas) + 1, dtype=float)
    if complex_data:
        phis[0] = _angle_mod_two_pi(-backend.angle(data_sorted[0]))
        for k in range(1, len(phis)):
            phis[k] = _angle_mod_two_pi(
                -backend.angle(data_sorted[k]) + backend.sum(phis[:k])
            )
    phis = backend.cast(phis, dtype=phis[0].dtype)

    # marking qubits that have suffered the action of a gate
    initial_string = [int(bit) for bit in bitstrings_sorted[0]]
    circuit = comp_basis_encoder(initial_string, nqubits=nqubits, **kwargs)
    touched_qubits = list(np.nonzero(initial_string)[0])

    for b_1, b_0, theta, phi in zip(
        bitstrings_sorted[1:], bitstrings_sorted[:-1], thetas, phis
    ):
        hw_0, hw_1 = hamming_weight(b_0), hamming_weight(b_1)
        distance = hamming_distance(b_1, b_0)
        difference = b_1 - b_0

        ones, new_ones = (
            list(backend.argsort(b_0)[-hw_0:]),
            list(backend.argsort(b_1)[-hw_1:]),
        )
        ones, new_ones = {int(elem) for elem in ones}, {int(elem) for elem in new_ones}
        controls = (set(ones) & set(new_ones)) & set(touched_qubits)

        gate = _get_gate_sparse(
            distance,
            difference,
            touched_qubits,
            complex_data,
            controls,
            hw_0,
            hw_1,
            theta,
            phi,
            backend=backend,
        )
        circuit.add(gate)

    if complex_data:
        hw_0 = hamming_weight(bitstrings_sorted[-2])
        hw_1 = hamming_weight(bitstrings_sorted[-1])
        correction = _get_phase_gate_correction_sparse(
            bitstrings_sorted[-1],
            bitstrings_sorted[-2],
            nqubits,
            data_sorted[-1],
            data_sorted[-2],
            circuit,
            phis,
        )
        if hw_1 == nqubits and hw_0 == nqubits - 1:
            circuit.queue = correction
        else:
            circuit.add(correction)

    return circuit


def _sparse_encoder_li(
    data: ArrayLike, nqubits: int, backend: Optional[Backend] = None, **kwargs
) -> Circuit:
    """Create circuit that encodes :math:`1`-dimensional data in a subset of amplitudes
    of the computational basis.

    Consider a sparse-access model, where for a data vector
    :math:`\\mathbf{x} \\in \\mathbb{C}^{d}`, with :math:`d = 2^{n}` and
    :math:`s` non-zero amplitudes, one has access to the data vector
    :math:`\\mathbf{y}` of the form

    .. math::
        \\mathbf{y} = \\left\\{ (b_{1}, x_{1}), \\, \\dots, \\, (b_{s}, x_{s}) \\right\\} \\, ,


    where :math:`\\{x_{j}\\}_{j\\in[s]}` are the non-zero components of :math:`\\mathbf{x}`
    and :math:`\\{b_{j}\\}_{j\\in[s]}` is the set of addresses associated with these values.
    Then, this function generates a quantum circuit  :math:`s\\text{-}\\mathrm{Load}` that encodes
    :math:`\\mathbf{x}` in the amplitudes of an :math:`n`-qubit quantum state as

    .. math::
        s\\text{-}\\mathrm{Load}(\\mathbf{y}) \\, \\ket{0}^{\\otimes \\, n} = \\sum_{j\\in[s]} \\,
            \\frac{x_{j}}{\\|\\mathbf{x}\\|_{2}} \\, \\ket{b_{j}} \\, ,

    where :math:`\\|\\cdot\\|_{2}` is the l2-norm.

    The resulting circuit parametrizes ``data`` in hyperspherical coordinates
    in the :math:`(2^{n} - 1)`-unit sphere.


    Args:
        data (ndarray or list or zip): sequence of tuples of the form :math:`(b_{j}, x_{j})`.
            The addresses :math:`b_{j}` can be either integers or in bitstring
            format of size :math:`n`.
        nqubits (int, optional): total number of qubits in the system.
            To be used when :math:`b_j` are integers. If :math:`b_j` are strings and
            ``nqubits`` is ``None``, defaults to the length of the strings :math:`b_{j}`.
            Defaults to ``None``.
        backend (:class:`qibo.backends.abstract.Backend`, optional): backend to be used
            in the execution. If ``None``, it uses the current backend. Defaults to ``None``.
        kwargs (dict, optional): Additional arguments used to initialize a Circuit object.
            For details, see the documentation of :class:`qibo.models.circuit.Circuit`.

    Returns:
        :class:`qibo.models.circuit.Circuit`: Circuit that loads sparse :math:`\\mathbf{x}`.

    References:
        1. L. Li, and J. Luo,
        *Nearly Optimal Circuit Size for Sparse Quantum State Preparation*
        `arXiv:2406.16142 (2024) <https://doi.org/10.48550/arXiv.2406.16142>`_.

        2. `Hyperpherical coordinates <https://en.wikipedia.org/wiki/N-sphere>`_.
    """
    from qibo.models.encodings import permutation_synthesis  # pylint: disable=C0415

    backend = _check_backend(backend)

    # TODO: Fix this mess with qibo native data types
    try:
        test_dtype = bool("int" in str(data[0][0].dtype))
    except AttributeError:
        test_dtype = bool("int" in str(type(data[0][0])))

    bitstrings_sorted, data_sorted = zip(
        *([(f"{row[0]:0{nqubits}b}", row[1]) for row in data] if test_dtype else data)
    )
    bitstrings_sorted = backend.cast(
        [int("".join(map(str, string)), 2) for string in bitstrings_sorted],
        dtype=_get_int_type(2**nqubits, backend=backend),
    )

    data_sorted = backend.cast(data_sorted, dtype=data_sorted[0].dtype)

    dim = len(data_sorted)
    sigma = backend.arange(2**nqubits)

    flag = backend.zeros(dim, dtype=backend.int8)
    indexes = list(
        backend.to_numpy(bitstrings_sorted[bitstrings_sorted < dim]).astype(int)
    )
    flag[indexes] = 1

    data_binary = backend.zeros(dim, dtype=data_sorted.dtype)
    for bi_int, xi in zip(bitstrings_sorted, data_sorted):
        bi_int = int(bi_int)
        if bi_int >= dim:
            for k in range(dim):
                if flag[k] == 0:
                    flag[k] = 1
                    sigma[[bi_int, k]] = [k, bi_int]
                    data_binary[k] = xi
                    break
        else:
            data_binary[bi_int] = xi

    sigma = [int(elem) for elem in sigma]

    # binary enconder on \sum_i = xi |sigma^{-1}(b_i)>
    circuit_binary = _binary_encoder_hyperspherical(
        data_binary,
        nqubits=nqubits,
        complex_data=bool("complex" in str(data_binary.dtype)),
        backend=backend,
        **kwargs,
    )
    circuit_permutation = permutation_synthesis(sigma, **kwargs)

    return circuit_binary + circuit_permutation


def _up_to_k_encoder_hyperspherical(
    data: ArrayLike,
    nqubits: int,
    up_to_k: int,
    complex_data: bool,
    codewords: Optional[List[int]] = None,
    keep_antictrls: bool = False,
    backend: Optional[Backend] = None,
    **kwargs,
) -> Circuit:
    backend = _check_backend(backend)

    dims = len(data)
    expected_dim = sum(binom(nqubits, l) for l in range(up_to_k + 1))
    if dims != expected_dim:
        raise_error(
            ValueError,
            f"Data dimension should be {expected_dim}, passed data has dims {dims}.",
        )

    if codewords is None:
        codewords = list(
            _ehrlich_codewords_up_to_k(up_to_k, False, nqubits, backend=backend)
        )
        codewords_lex = sorted(codewords)
        codewords_reorder = backend.cast(
            [codewords_lex.index(x) for x in codewords],
            dtype=_get_int_type(dims, backend=backend),
        )
        codewords = backend.cast(
            [int(bin_cw, 2) for bin_cw in codewords],
            dtype=_get_int_type(dims, backend=backend),
        )
    else:
        codewords = backend.cast(codewords, dtype=_get_int_type(dims, backend=backend))
        codewords_reorder = backend.cast(
            codewords, copy=True, dtype=_get_int_type(dims, backend=backend)
        )

    circuit = Circuit(nqubits, **kwargs)
    if complex_data:
        circuit += _monotonic_hw_encoder_complex(
            codewords,
            data[codewords_reorder],
            nqubits,
            backend=backend,
            keep_antictrls=keep_antictrls,
            **kwargs,
        )
    else:
        circuit += _monotonic_hw_encoder_real(
            codewords,
            data[codewords_reorder],
            nqubits,
            backend=backend,
            keep_antictrls=keep_antictrls,
            **kwargs,
        )

    return circuit


def _v_layer(
    nqubits: int, gate: Gate, parameters: Union[List[int], Tuple[int, ...]], **kwargs
) -> Circuit:
    """Create entangling layer in V shape."""
    _, pairs_gates = _generate_rbs_pairs(nqubits, architecture="diagonal")
    pairs_gates = pairs_gates[::-1]

    circuit = Circuit(nqubits, **kwargs)
    circuit.add(
        _parametrized_two_qubit_gate(gate, pair[0][1], pair[0][0], parameters)
        for pair in pairs_gates
    )
    circuit.add(
        _parametrized_two_qubit_gate(gate, pair[0][1], pair[0][0], parameters)
        for pair in pairs_gates[::-1][1:]
    )

    return circuit


def _x_layer(
    nqubits: int, gate: Gate, parameters: Union[List[int], Tuple[int, ...]], **kwargs
) -> Circuit:
    """Create entangling layer in X shape."""
    _, pairs_gates = _generate_rbs_pairs(nqubits, architecture="diagonal")
    pairs_gates = pairs_gates[::-1]

    middle = int(np.floor(len(pairs_gates) / 2))
    pairs_1 = pairs_gates[:middle]
    pairs_2 = pairs_gates[-middle:]

    circuit = Circuit(nqubits, **kwargs)

    for first, second in zip(pairs_1, pairs_2[::-1]):
        circuit.add(
            _parametrized_two_qubit_gate(gate, first[0][1], first[0][0], parameters)
        )
        circuit.add(
            _parametrized_two_qubit_gate(gate, second[0][1], second[0][0], parameters)
        )

    circuit.add(
        _parametrized_two_qubit_gate(
            gate,
            pairs_gates[middle][0][1],
            pairs_gates[middle][0][0],
            parameters,
        )
    )

    for first, second in zip(pairs_1[::-1], pairs_2):
        circuit.add(
            _parametrized_two_qubit_gate(gate, first[0][1], first[0][0], parameters)
        )
        circuit.add(
            _parametrized_two_qubit_gate(gate, second[0][1], second[0][0], parameters)
        )

    return circuit
