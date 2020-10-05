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