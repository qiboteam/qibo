"""Utility functions that support the Clifford submodule."""

from functools import reduce
from itertools import product

import numpy as np

from qibo import Circuit, gates
from qibo.config import raise_error


def _one_qubit_paulis_string_product(pauli_1: str, pauli_2: str):
    """Calculate the product of two single-qubit Paulis represented as strings.

    Args:
        pauli_1 (str): First Pauli operator.
        pauli_2 (str): Second Pauli operator.

    Returns:
        (str): Product of the two Pauli operators.
    """
    products = {
        "XY": "iZ",
        "YZ": "iX",
        "ZX": "iY",
        "YX": "-iZ",
        "ZY": "-iX",
        "XZ": "iY",
        "XX": "I",
        "ZZ": "I",
        "YY": "I",
        "XI": "X",
        "IX": "X",
        "YI": "Y",
        "IY": "Y",
        "ZI": "Z",
        "IZ": "Z",
    }
    prod = products[
        "".join([p.replace("i", "").replace("-", "") for p in (pauli_1, pauli_2)])
    ]
    # calculate the phase
    sign = len([True for p in (pauli_1, pauli_2, prod) if "-" in p])
    n_i = len([True for p in (pauli_1, pauli_2, prod) if "i" in p])
    sign = "-" if sign % 2 == 1 else ""
    if n_i == 0:
        i = ""
    elif n_i == 1:
        i = "i"
    elif n_i == 2:
        i = ""
        sign = "-" if sign == "" else ""
    elif n_i == 3:
        i = "i"
        sign = "-" if sign == "" else ""
    return "".join(
        [sign, i, prod.replace("i", "").replace("-", "")]  # pylint: disable=E0606
    )


def _string_product(operators: list):
    """Calculates the tensor product of a list of operators represented as strings.

    Args:
        operators (list): list of operators.

    Returns:
        (str): String representing the tensor product of the operators.
    """
    # calculate global sign
    phases = len([True for op in operators if "-" in op])
    i = len([True for op in operators if "i" in op])
    # remove the - signs and the i
    operators = "|".join(operators).replace("-", "").replace("i", "").split("|")

    prod = []
    for op in zip(*operators):
        op = [o for o in op if o != "I"]
        if len(op) == 0:
            tmp = "I"
        elif len(op) > 1:
            tmp = reduce(_one_qubit_paulis_string_product, op)
        else:
            tmp = op[0]
        # append signs coming from products
        if tmp[0] == "-":
            phases += 1
        # count i coming from products
        if "i" in tmp:
            i += 1
        prod.append(tmp.replace("i", "").replace("-", ""))
    result = "".join(prod)

    # product of the i-s
    if i % 4 == 1 or i % 4 == 3:
        result = f"i{result}"
    if i % 4 == 2 or i % 4 == 3:
        phases += 1

    phases = "-" if phases % 2 == 1 else ""

    return f"{phases}{result}"


def _decomposition_AG04(clifford):
    """Returns a Clifford object decomposed into a circuit based on Aaronson-Gottesman method.

    Args:
        clifford (:class:`qibo.quantum_info.clifford.Clifford`): Clifford object.

    Returns:
        :class:`qibo.models.circuit.Circuit`: Clifford circuit.

    References:
        1. S. Aaronson, D. Gottesman, *Improved Simulation of Stabilizer Circuits*,
           Phys. Rev. A 70, 052328 (2004).
           `arXiv:quant-ph/0406196 <https://arxiv.org/abs/quant-ph/0406196>`_
    """
    nqubits = clifford.nqubits

    circuit = Circuit(nqubits)
    if clifford._backend.platform == "cupy":  # pragma: no cover
        raise_error(
            NotImplementedError,
            "``AG04`` algorithm currently not supported with the ``cupy`` engine, please use the ``BM20`` algorithm instead or switch clifford engine.",
        )

    clifford_copy = clifford.copy(deep=True)

    if nqubits == 1:
        return _single_qubit_clifford_decomposition(clifford_copy.symplectic_matrix)

    for k in range(nqubits):
        # put a 1 one into position by permuting and using Hadamards(i,i)
        _set_qubit_x_to_true(clifford_copy, circuit, k)
        # make all entries in row i except ith equal to 0
        # by using phase gate and CNOTS
        _set_row_x_to_zero(clifford_copy, circuit, k)
        # treat Zs
        _set_row_z_to_zero(clifford_copy, circuit, k)

    for k in range(nqubits):
        if clifford_copy.symplectic_matrix[:nqubits, -1][k]:
            clifford._backend.engine.Z(clifford_copy.symplectic_matrix, k, nqubits)
            circuit.add(gates.Z(k))
        if clifford_copy.symplectic_matrix[nqubits:-1, -1][k]:
            clifford._backend.engine.X(clifford_copy.symplectic_matrix, k, nqubits)
            circuit.add(gates.X(k))

    return circuit.invert()


def _decomposition_BM20(clifford):
    """Optimal CNOT-cost decomposition of a Clifford operator on :math:`n \\in \\{2, 3 \\}`
    into a circuit based on Bravyi-Maslov method.

    Args:
        clifford (:class:`qibo.quantum_info.clifford.Clifford`): Clifford object.

    Returns:
        :class:`qibo.models.circuit.Circuit`: Clifford circuit.

    References:
        1. S. Bravyi, D. Maslov, *Hadamard-free circuits expose the structure of the Clifford group*,
           `arXiv:2003.09412 [quant-ph] <https://arxiv.org/abs/2003.09412>`_.
    """
    nqubits = clifford.nqubits
    clifford_copy = clifford.copy(deep=True)

    if nqubits > 3:
        raise_error(
            ValueError, "This method can only be implemented for ``nqubits <= 3``."
        )

    if nqubits == 1:
        return _single_qubit_clifford_decomposition(clifford_copy.symplectic_matrix)

    inverse_circuit = Circuit(nqubits)

    cnot_cost = _cnot_cost(clifford_copy)

    while cnot_cost > 0:
        clifford_copy, inverse_circuit, cnot_cost = _reduce_cost(
            clifford_copy, inverse_circuit, cnot_cost
        )

    last_row = clifford_copy.engine.cast([False] * 3, dtype=bool)
    circuit = Circuit(nqubits)
    for qubit in range(nqubits):
        position = [qubit, qubit + nqubits]
        single_qubit_circuit = _single_qubit_clifford_decomposition(
            clifford_copy.engine.np.append(
                clifford_copy.symplectic_matrix[position][:, position + [-1]], last_row
            ).reshape(3, 3)
        )
        if len(single_qubit_circuit.queue) > 0:
            for gate in single_qubit_circuit.queue:
                gate.init_args = [qubit]
                gate.target_qubits = (qubit,)
                circuit.queue.extend([gate])

    if len(inverse_circuit.queue) > 0:
        circuit.queue.extend(inverse_circuit.invert().queue)

    return circuit


def _single_qubit_clifford_decomposition(symplectic_matrix):
    """Decompose symplectic matrix of a single-qubit Clifford into a Clifford circuit.

    Args:
        symplectic_matrix (ndarray): Symplectic matrix to be decomposed.

    Returns:
        :class:`qibo.models.circuit.Circuit`: Clifford circuit.
    """
    circuit = Circuit(nqubits=1)

    destabilizer_phase, stabilizer_phase = symplectic_matrix[:-1, -1]
    if destabilizer_phase and not stabilizer_phase:
        circuit.add(gates.Z(0))
    elif not destabilizer_phase and stabilizer_phase:
        circuit.add(gates.X(0))
    elif destabilizer_phase and stabilizer_phase:
        circuit.add(gates.Y(0))

    destabilizer_x, destabilizer_z = symplectic_matrix[0, 0], symplectic_matrix[0, 1]
    stabilizer_x, stabilizer_z = symplectic_matrix[1, 0], symplectic_matrix[1, 1]

    if stabilizer_z and not stabilizer_x:
        if destabilizer_z:
            circuit.add(gates.S(0))
    elif not stabilizer_z and stabilizer_x:
        if destabilizer_x:
            circuit.add(gates.SDG(0))
        circuit.add(gates.H(0))
    else:
        if not destabilizer_z:
            circuit.add(gates.S(0))
        circuit.add(gates.H(0))
        circuit.add(gates.S(0))

    return circuit


def _set_qubit_x_to_true(clifford, circuit: Circuit, qubit: int):
    """Set a :math:`X`-destabilizer to ``True``.

    This is done by permuting columns ``l > qubit`` or, if necessary, applying a Hadamard.

    Args:
        clifford (:class:`qibo.quantum_info.clifford.Clifford`): Clifford object.
        circuit (:class:`qibo.models.circuit.Circuit`): circuit object.
        qubit (int): index of the qubit to operate on.
    """
    nqubits = clifford.nqubits

    x = clifford.destabilizers(symplectic=True)
    x, z = x[:, :nqubits][qubit], x[:, nqubits:-1][qubit]

    if x[qubit]:
        return

    for k in range(qubit + 1, nqubits):
        if x[k]:
            clifford._backend.engine.SWAP(clifford.symplectic_matrix, k, qubit, nqubits)
            circuit.add(gates.SWAP(k, qubit))
            return

    for k in range(qubit, nqubits):
        if z[k]:
            clifford._backend.engine.H(clifford.symplectic_matrix, k, nqubits)
            circuit.add(gates.H(k))
            if k != qubit:
                clifford._backend.engine.SWAP(
                    clifford.symplectic_matrix, k, qubit, nqubits
                )
                circuit.add(gates.SWAP(k, qubit))
            return


def _set_row_x_to_zero(clifford, circuit: Circuit, qubit: int):
    """Set :math:`X`-destabilizer to ``False`` for all ``k > qubit``.

    This is done by applying CNOTs, assuming ``k <= N`` and ``clifford.symplectic_matrix[k][k]=1``.

    Args:
        clifford (:class:`qibo.quantum_info.clifford.Clifford`): Clifford object.
        circuit (:class:`qibo.models.circuit.Circuit`): circuit object.
        qubit (int): index of the qubit to operate on.
    """
    nqubits = clifford.nqubits

    x = clifford.destabilizers(symplectic=True)
    x, z = x[:, :nqubits][qubit], x[:, nqubits:-1][qubit]

    # Check X first
    for k in range(qubit + 1, nqubits):
        if x[k]:
            clifford._backend.engine.CNOT(clifford.symplectic_matrix, qubit, k, nqubits)
            circuit.add(gates.CNOT(qubit, k))

    if np.any(z[qubit:]):
        if not z[qubit]:
            # to treat Zs: make sure row.Z[k] to True
            clifford._backend.engine.S(clifford.symplectic_matrix, qubit, nqubits)
            circuit.add(gates.S(qubit))

        for k in range(qubit + 1, nqubits):
            if z[k]:
                clifford._backend.engine.CNOT(
                    clifford.symplectic_matrix, k, qubit, nqubits
                )
                circuit.add(gates.CNOT(k, qubit))

        clifford._backend.engine.S(clifford.symplectic_matrix, qubit, nqubits)
        circuit.add(gates.S(qubit))


def _set_row_z_to_zero(clifford, circuit: Circuit, qubit: int):
    """Set :math:`Z`-stabilizer to ``False`` for all ``i > qubit``.

    Implemented by applying (reverse) CNOTs.
    It assumes ``qubit < nqubits`` and that ``_set_row_x_to_zero`` has been called first.

    Args:
        clifford (:class:`qibo.quantum_info.clifford.Clifford`): Clifford object.
        circuit (:class:`qibo.models.circuit.Circuit`): circuit object.
        qubit (int): index of the qubit to operate on.
    """
    nqubits = clifford.nqubits

    x = clifford.stabilizers(symplectic=True)
    x, z = x[:, :nqubits][qubit], x[:, nqubits:-1][qubit]

    if np.any(z[qubit + 1 :]):
        for k in range(qubit + 1, nqubits):
            if z[k]:
                clifford._backend.engine.CNOT(
                    clifford.symplectic_matrix, k, qubit, nqubits
                )
                circuit.add(gates.CNOT(k, qubit))

    if np.any(x[qubit:]):
        clifford._backend.engine.H(clifford.symplectic_matrix, qubit, nqubits)
        circuit.add(gates.H(qubit))
        for k in range(qubit + 1, nqubits):
            if x[k]:
                clifford._backend.engine.CNOT(
                    clifford.symplectic_matrix, qubit, k, nqubits
                )
                circuit.add(gates.CNOT(qubit, k))
        if z[qubit]:
            clifford._backend.engine.S(clifford.symplectic_matrix, qubit, nqubits)
            circuit.add(gates.S(qubit))
        clifford._backend.engine.H(clifford.symplectic_matrix, qubit, nqubits)
        circuit.add(gates.H(qubit))


def _cnot_cost(clifford):
    """Returns the number of CNOT gates required for Clifford decomposition.

    Args:
        clifford (:class:`qibo.quantum_info.clifford.Clifford`): Clifford object.

    Returns:
        int: Number of CNOT gates required.
    """
    if clifford.nqubits > 3:
        raise_error(ValueError, "No Clifford CNOT cost function for ``nqubits > 3``.")

    if clifford.nqubits == 3:
        return _cnot_cost3(clifford)

    return _cnot_cost2(clifford)


def _rank_2(a: bool, b: bool, c: bool, d: bool):
    """Returns rank of 2x2 boolean matrix."""
    if (a & d) ^ (b & c):
        return 2
    if a or b or c or d:
        return 1
    return 0


def _cnot_cost2(clifford):
    """Returns CNOT cost of a two-qubit Clifford.

    Args:
        clifford (:class:`qibo.quantum_info.clifford.Clifford`): Clifford object.

    Returns:
        int: Number of CNOT gates required.
    """
    symplectic_matrix = clifford.symplectic_matrix[:-1, :-1]

    r00 = _rank_2(
        symplectic_matrix[0, 0],
        symplectic_matrix[0, 2],
        symplectic_matrix[2, 0],
        symplectic_matrix[2, 2],
    )
    r01 = _rank_2(
        symplectic_matrix[0, 1],
        symplectic_matrix[0, 3],
        symplectic_matrix[2, 1],
        symplectic_matrix[2, 3],
    )

    if r00 == 2:
        return r01

    return r01 + 1 - r00


def _cnot_cost3(clifford):  # pragma: no cover
    """Return CNOT cost of a 3-qubit clifford.

    Args:
        clifford (:class:`qibo.quantum_info.clifford.Clifford`): Clifford object.

    Returns:
        int: Number of CNOT gates required.
    """

    symplectic_matrix = clifford.symplectic_matrix[:-1, :-1]

    nqubits = 3

    R1 = np.zeros((nqubits, nqubits), dtype=int)
    R2 = np.zeros((nqubits, nqubits), dtype=int)
    for q1 in range(nqubits):
        for q2 in range(nqubits):
            R2[q1, q2] = _rank_2(
                symplectic_matrix[q1, q2],
                symplectic_matrix[q1, q2 + nqubits],
                symplectic_matrix[q1 + nqubits, q2],
                symplectic_matrix[q1 + nqubits, q2 + nqubits],
            )
            mask = np.zeros(2 * nqubits, dtype=int)
            mask = clifford.engine.cast(mask, dtype=mask.dtype)
            mask[[q2, q2 + nqubits]] = 1
            loc_y_x = np.array_equal(
                symplectic_matrix[q1, :] & mask, symplectic_matrix[q1, :]
            )
            loc_y_z = np.array_equal(
                symplectic_matrix[q1 + nqubits, :] & mask,
                symplectic_matrix[q1 + nqubits, :],
            )
            loc_y_y = np.array_equal(
                (symplectic_matrix[q1, :] ^ symplectic_matrix[q1 + nqubits, :]) & mask,
                (symplectic_matrix[q1, :] ^ symplectic_matrix[q1 + nqubits, :]),
            )
            R1[q1, q2] = 1 * (loc_y_x or loc_y_z or loc_y_y) + 1 * (
                loc_y_x and loc_y_z and loc_y_y
            )

    diag1 = np.sort(np.diag(R1)).tolist()
    diag2 = np.sort(np.diag(R2)).tolist()

    nz1 = np.count_nonzero(R1)
    nz2 = np.count_nonzero(R2)

    if diag1 == [2, 2, 2]:
        return 0

    if diag1 == [1, 1, 2]:
        return 1

    if (
        diag1 == [0, 1, 1]
        or (diag1 == [1, 1, 1] and nz2 < 9)
        or (diag1 == [0, 0, 2] and diag2 == [1, 1, 2])
    ):
        return 2

    if (
        (diag1 == [1, 1, 1] and nz2 == 9)
        or (
            diag1 == [0, 0, 1]
            and (nz1 == 1 or diag2 == [2, 2, 2] or (diag2 == [1, 1, 2] and nz2 < 9))
        )
        or (diag1 == [0, 0, 2] and diag2 == [0, 0, 2])
        or (diag2 == [1, 2, 2] and nz1 == 0)
    ):
        return 3

    if diag2 == [0, 0, 1] or (
        diag1 == [0, 0, 0]
        and (
            (diag2 == [1, 1, 1] and nz2 == 9 and nz1 == 3)
            or (diag2 == [0, 1, 1] and nz2 == 8 and nz1 == 2)
        )
    ):
        return 5

    if nz1 == 3 and nz2 == 3:
        return 6

    return 4


def _reduce_cost(clifford, inverse_circuit: Circuit, cost: int):  # pragma: no cover
    """Step that tries to reduce the two-qubit cost of a Clifford circuit.

    Args:
        clifford (:class:`qibo.quantum_info.clifford.Clifford`): Clifford object.
        circuit (:class:`qibo.models.circuit.Circuit`): circuit object.
        cost (int): initial cost.
    """
    nqubits = clifford.nqubits

    for control in range(nqubits):
        for target in range(control + 1, nqubits):
            for n0, n1 in product(range(3), repeat=2):
                reduced = clifford.copy(deep=True)

                matrix = reduced.symplectic_matrix
                matrix = reduced._backend._clifford_pre_execution_reshape(matrix)

                for qubit, n in [(control, n0), (target, n1)]:
                    if n == 1:
                        matrix = reduced._backend.engine.SDG(matrix, qubit, nqubits)
                        matrix = reduced._backend.engine.H(matrix, qubit, nqubits)
                    elif n == 2:
                        matrix = reduced._backend.engine.SDG(matrix, qubit, nqubits)
                        matrix = reduced._backend.engine.H(matrix, qubit, nqubits)
                        matrix = reduced._backend.engine.SDG(matrix, qubit, nqubits)
                        matrix = reduced._backend.engine.H(matrix, qubit, nqubits)
                matrix = reduced._backend.engine.CNOT(matrix, control, target, nqubits)
                reduced.symplectic_matrix = (
                    reduced._backend._clifford_post_execution_reshape(matrix, nqubits)
                )
                new_cost = _cnot_cost(reduced)

                if new_cost == cost - 1:
                    for qubit, n in [(control, n0), (target, n1)]:
                        if n == 1:
                            inverse_circuit.add(gates.SDG(qubit))
                            inverse_circuit.add(gates.H(qubit))
                        elif n == 2:
                            inverse_circuit.add(gates.H(qubit))
                            inverse_circuit.add(gates.S(qubit))
                    inverse_circuit.add(gates.CNOT(control, target))

                    return reduced, inverse_circuit, new_cost

    raise_error(RuntimeError, "Failed to reduce CNOT cost.")
