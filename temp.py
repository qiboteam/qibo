@classmethod
def from_symbolic(cls, symbolic_hamiltonian, symbol_map, ground_state=None):
    """Creates a ``TrotterHamiltonian`` from a symbolic Hamiltonian.

    We refer to the :ref:`How to define custom Hamiltonians using symbols? <symbolicham-example>`
    example for more details.

    Args:
        symbolic_hamiltonian (sympy.Expr): The full Hamiltonian written
            with symbols.
        symbol_map (dict): Dictionary that maps each symbol that appears in
            the Hamiltonian to a pair of (target, matrix).
        ground_state (Callable): Optional callable with no arguments that
            returns the ground state of this ``TrotterHamiltonian``.
            See :class:`qibo.base.hamiltonians.TrotterHamiltonian` for more
            details.

    Returns:
        A :class:`qibo.base.hamiltonians.TrotterHamiltonian` object that
        implements the given symbolic Hamiltonian.
    """
    from qibo.hamiltonians import Hamiltonian
    terms, constant = _SymbolicHamiltonian(
      symbolic_hamiltonian, symbol_map).trotter_terms()
    hterms = cls._construct_terms(terms)
    return cls.from_dictionary(hterms, ground_state=ground_state) + constant

@staticmethod
def _split_terms(terms):
    """Splits a dictionary of terms to multiple parts.

    Each qubit should not appear in more that one terms in each
    part to ensure commutation relations in the definition of
    :class:`qibo.base.hamiltonians.TrotterHamiltonian`.

    Args:
        terms (dict): Dictionary that maps tuples of targets to the matrix
                      that acts on these on targets.

    Returns:
        List of dictionary parts to be used for the creation of a
        ``TrotterHamiltonian``. The parts are such that no qubit appears
        twice in each part.
    """
    groups, singles = [set()], [set()]
    for targets in terms.keys():
        flag = True
        t = set(targets)
        for g, s in zip(groups, singles):
            if not t & s:
                s |= t
                g.add(targets)
                flag = False
                break
        if flag:
            groups.append({targets})
            singles.append(t)
    return [{k: terms[k] for k in g} for g in groups]

@staticmethod
def _construct_terms(terms):
    # Avoid creating duplicate ``Hamiltonian`` objects for terms
    # to take better advantage of caching and increase performance
    from qibo.hamiltonians import Hamiltonian
    unique_matrices = []
    hterms = {}
    for targets, matrix in terms.items():
        flag = True
        for m, h in unique_matrices:
            if np.array_equal(matrix, m):
                ham = h
                flag = False
                break
        if flag:
            ham = Hamiltonian(len(targets), matrix, numpy=True)
            unique_matrices.append((matrix, ham))
        hterms[targets] = ham
    return hterms

def is_compatible(self, o):
    """Checks if a ``TrotterHamiltonian`` has the same part structure.

    ``TrotterHamiltonian``s with the same part structure can be add.

    Args:
        o: The second Hamiltonian to check.

    Returns:
        ``True`` if ``o`` has the same structure as ``self`` otherwise
        ``False``.
    """
    if isinstance(o, self.__class__):
        if len(self.parts) != len(o.parts):
            return False
        for part1, part2 in zip(self.parts, o.parts):
            if set(part1.keys()) != set(part2.keys()):
                return False
        return True
    return False

def make_compatible(self, o):
    """Makes given ``TrotterHamiltonian`` compatible to the current one.

    Args:
        o: The ``TrotterHamiltonian`` to make compatible to the current.
            Should be non-interacting (contain only one-qubit terms).

    Returns:
        A new :class:`qibo.base.hamiltonians.TrotterHamiltonian` object
        that is equivalent to ``o`` but has the same part structure as
        ``self``.
    """
    if not isinstance(o, self.__class__):
        raise TypeError("Only ``TrotterHamiltonians`` can be made "
                        "compatible but {} was given.".format(type(o)))
    if self.is_compatible(o):
        return o

    oterms = {}
    for part in o.parts:
        for t, m in part.items():
            if len(t) > 1:
                raise_error(NotImplementedError,
                            "Only non-interacting Hamiltonians can be "
                            "transformed using the ``make_compatible`` "
                            "method.")
            oterms[t[0]] = m

    normalizer = {}
    for part in self.parts:
        for targets in part.keys():
            if targets[0] in normalizer:
                normalizer[targets[0]] += 1
            else:
                normalizer[targets[0]] = 1
    if set(normalizer.keys()) != set(oterms.keys()):
        raise_error(ValueError, "Given non-interacting Hamiltonian cannot "
                                "be made compatible.")

    new_terms = {}
    for part in self.parts:
        for targets in part.keys():
            if targets[0] in oterms:
                m = oterms[targets[0]].matrix
                eye = np.eye(2 ** (len(targets) - 1), dtype=m.dtype)
                new_terms[targets] = np.kron(m, eye) / normalizer[targets[0]]

    new_terms = self._construct_terms(new_terms)
    new_parts = ({t: new_terms[t] for t in part.keys()}
                 for part in self.parts)
    return self.__class__(*new_parts, ground_state=o.ground_state_func)