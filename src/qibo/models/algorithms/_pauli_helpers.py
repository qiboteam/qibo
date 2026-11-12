from itertools import pairwise

from qibo import gates
from qibo.backends import Backend, _check_backend
from qibo.config import log, raise_error
from qibo.hamiltonians import Hamiltonian, SymbolicHamiltonian


def _pauli_rotation(pauli: str, angle: float) -> list[gates.Gate]:
    """Creates the gates of the rotation :math:`e^{-i \\theta P / 2}` of a Pauli string :math:`P`.

    The Pauli matrices of the string are rotated to :math:`Z` with Clifford gates, then a ladder
    of CNOT gates writes the parity of the qubits in the last of them, which has a
    :class:`qibo.gates.RZ` gate. The identity is a :class:`qibo.gates.GlobalPhase` gate.

    Args:
        pauli (str): Pauli string, with one letter from ``"IXYZ"`` for each qubit. The :math:`j`-th
            letter is the Pauli matrix applied to the :math:`j`-th qubit.
        angle (float): angle :math:`\\theta` of the rotation.

    Returns:
        list: gates of the rotation.
    """
    targets = [qubit for qubit, letter in enumerate(pauli) if letter != "I"]

    # the identity is a global phase
    if len(targets) == 0:
        return [gates.GlobalPhase(-angle / 2)]

    # basis change that maps the Pauli string to Z in each of the targets, and a ladder
    # of CNOTs that writes its parity in the last target
    before, after = [], []
    for qubit in targets:
        if pauli[qubit] == "X":
            before.append(gates.H(qubit))
            after.append(gates.H(qubit))
        elif pauli[qubit] == "Y":
            before.extend([gates.SDG(qubit), gates.H(qubit)])
            after.extend([gates.H(qubit), gates.S(qubit)])
    pairs = list(pairwise(targets))

    rotation = []
    rotation.extend(before)
    rotation.extend(gates.CNOT(*pair) for pair in pairs)
    rotation.append(gates.RZ(targets[-1], theta=angle))
    rotation.extend(gates.CNOT(*pair) for pair in reversed(pairs))
    rotation.extend(after)

    return rotation


def _pauli_terms(
    operator: (
        Hamiltonian
        | SymbolicHamiltonian
        | list[tuple[str, complex]]
        | tuple[tuple[str, complex], ...]
    ),
    nqubits: int | None,
    label: str = "operator",
    real: bool = False,
    backend: Backend | None = None,
) -> list[tuple[str, complex]]:
    """Writes an operator as a sum of Pauli strings, :math:`\\sum_{k} c_{k} P_{k}`.

    A symbolic Hamiltonian made of Pauli symbols gives its strings directly. For any other
    Hamiltonian, they are computed from its matrix with :func:`qibo.quantum_info.pauli_basis`,
    which is inefficient in the number of qubits, and a warning is logged.

    Args:
        operator (:class:`qibo.hamiltonians.Hamiltonian`,
            :class:`qibo.hamiltonians.SymbolicHamiltonian`, list or tuple): Hamiltonian, or list
            or tuple of tuples ``(pauli, coefficient)``, with the Pauli string ``pauli`` and its
            complex ``coefficient``. The :math:`j`-th letter of the string, among ``"IXYZ"``, is
            the Pauli matrix applied to the :math:`j`-th qubit.
        nqubits (int, optional): number of qubits of ``operator``. If ``None``, it is taken from
            ``operator``.
        label (str, optional): name of the argument of the caller that is ``operator``, which is
            used in the error messages. Defaults to ``"operator"``.
        real (bool, optional): If ``True``, the coefficients must be real numbers, which are
            returned as floats. Defaults to ``False``.
        backend (:class:`qibo.backends.abstract.Backend`, optional): backend to be used in the
            calculation. If ``None``, it uses the current backend. Defaults to ``None``.

    Returns:
        list: tuples ``(pauli, coefficient)`` of the Pauli strings and their coefficients, with at
        least one non-zero coefficient.
    """
    backend = _check_backend(backend)

    if not isinstance(operator, (list, tuple, Hamiltonian, SymbolicHamiltonian)):
        raise_error(
            TypeError,
            f"{label} must be a list, a tuple, a Hamiltonian or a SymbolicHamiltonian, "
            + f"but it is type {type(operator)}.",
        )

    if isinstance(operator, (list, tuple)):
        for term in operator:
            if not (
                isinstance(term, (list, tuple))
                and len(term) == 2
                and isinstance(term[0], str)
            ):
                raise_error(
                    TypeError,
                    f"the elements of {label} must be tuples of a Pauli string and "
                    + f"a coefficient, but one is {term}.",
                )
            if nqubits is None:
                nqubits = len(term[0])
            if len(term[0]) != nqubits or not set(term[0]) <= set("IXYZ"):
                raise_error(
                    ValueError,
                    f"the Pauli string '{term[0]}' must have {nqubits} letters "
                    + "among 'I', 'X', 'Y' and 'Z'.",
                )
        terms = list(operator)
    else:
        if nqubits is not None and operator.nqubits != nqubits:
            raise_error(
                ValueError,
                f"{label} must have {nqubits} qubits, but it has {operator.nqubits}.",
            )
        nqubits = operator.nqubits

        if isinstance(operator, SymbolicHamiltonian) and all(
            set(name) <= set("XYZ") for name in operator.simple_terms[1]
        ):
            coefficients, names, qubits = operator.simple_terms
            terms = []
            for coefficient, name, targets in zip(coefficients, names, qubits):
                pauli = ["I"] * nqubits
                for letter, qubit in zip(name, targets):
                    pauli[qubit] = letter
                terms.append(("".join(pauli), coefficient))
            if operator.constant != 0.0:
                terms.append(("I" * nqubits, operator.constant))
        else:
            from qibo.quantum_info import pauli_basis  # circular import

            log.warning(
                "Calculating the Pauli decomposition of the matrix of a Hamiltonian. "
                "This operation is inefficient in the number of qubits."
            )
            # coefficients of the Pauli strings, whose index in base four has the Pauli
            # matrix I, X, Y, Z of the first qubit as its most significant digit
            basis = pauli_basis(nqubits, vectorize=True, order="row", backend=backend)
            matrix = backend.cast(operator.matrix, dtype=backend.complex128)
            coefficients = backend.matmul(
                backend.conj(basis), backend.reshape(matrix, (-1,))
            ) / (2**nqubits)
            magnitudes = backend.abs(coefficients)
            terms = [
                (
                    "".join(
                        "IXYZ"[(int(index) // 4 ** (nqubits - 1 - qubit)) % 4]
                        for qubit in range(nqubits)
                    ),
                    complex(coefficients[index]),
                )
                for index in backend.nonzero(
                    magnitudes > 1e-12 * backend.max(magnitudes)
                )[0]
            ]

    if not any(coefficient != 0 for _, coefficient in terms):
        raise_error(
            ValueError,
            f"{label} must have at least one Pauli string with a non-zero coefficient.",
        )

    if real:
        coefficients = backend.cast(
            [coefficient for _, coefficient in terms], dtype=backend.complex128
        )

        if backend.any(
            backend.abs(backend.imag(coefficients))
            > 1e-12 * backend.max(backend.abs(coefficients))
        ):
            raise_error(
                ValueError, f"the coefficients of {label} must be real numbers."
            )

        terms = [
            (pauli, float(coefficient))
            for (pauli, _), coefficient in zip(terms, backend.real(coefficients))
        ]

    return terms
