"""Tools for parsing symbolic Hamiltonians."""
import numpy as np
import sympy


def symbolic_to_dict(symbolic_hamiltonian, symbol_map):
    """Transforms a symbolic Hamiltonian to a dictionary of targets and matrices.

    Helper method for ``from_symbolic``.
    Works for Hamiltonians with one and two qubit terms only.
    The two qubit terms should be sufficiently many so that every
    qubit appears as the first target at least once.

    Args:
        symbolic_hamiltonian: The full Hamiltonian written with symbols.
        symbol_map (dict): Dictionary that maps each symbol to a pair of
            (target, matrix).

    Returns:
        all_terms (dict): Dictionary that maps pairs of targets to the 4x4
                          matrix that acts on this pair in the given
                          Hamiltonian.
        constant (float): The overall constant term of the Hamiltonian.
    """
    symbolic_hamiltonian = sympy.expand(symbolic_hamiltonian)
    one_qubit_terms, two_qubit_terms = dict(), dict()
    first_targets = dict()
    constant = 0
    for term in symbolic_hamiltonian.args:
        if term.args:
            expression = term.args
        else:
            expression = (term,)

        symbols = [x for x in expression if x.is_symbol]
        numbers = [x for x in expression if not x.is_symbol]
        if numbers:
            if len(numbers) > 1:
                raise_error(ValueError, "Hamiltonian must be expanded "
                                        " before using this method.")
            const = float(numbers[0])
        else:
            const = 1

        if not symbols:
            constant += const
        elif len(symbols) == 1:
            target, matrix = symbol_map[symbols[0]]
            one_qubit_terms[target] = const * matrix
        elif len(symbols) == 2:
            target1, matrix1 = symbol_map[symbols[0]]
            target2, matrix2 = symbol_map[symbols[1]]
            if target1 in first_targets and target2 not in first_targets:
                target1, target2 = target2, target1
                matrix1, matrix2 = matrix2, matrix1
            two_qubit_terms[(target1, target2)] = const * np.kron(matrix1,
                                                                  matrix2)
            first_targets[target1] = (target1, target2)
        else:
            raise_error(ValueError, "Only one and two qubit terms are allowed.")

    all_terms = dict(two_qubit_terms)
    for target in one_qubit_terms.keys():
        if target not in first_targets:
            raise_error(ValueError, "Qubit {} has not been used as the "
                                    "first target.".format(target))
        pair = first_targets[target]
        eye = np.eye(2, dtype=one_qubit_terms[target].dtype)
        all_terms[pair] = (np.kron(one_qubit_terms[target], eye) +
                           two_qubit_terms[pair])
    return all_terms, constant
