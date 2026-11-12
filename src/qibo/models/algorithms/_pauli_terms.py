from qibo.backends import Backend, _check_backend
from qibo.config import log, raise_error
from qibo.hamiltonians import Hamiltonian, SymbolicHamiltonian


def _pauli_terms(
    operator: (
        Hamiltonian
        | SymbolicHamiltonian
        | list[tuple[str, complex]]
        | tuple[tuple[str, complex], ...]
    ),
    nqubits: int | None,
    label: str = "operator",
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
        backend (:class:`qibo.backends.abstract.Backend`): backend to be used in the calculation.

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

    return terms
