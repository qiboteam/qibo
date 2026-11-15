from numbers import Integral

import numpy as np

from qibo.backends import Backend, _check_backend, _check_backend_and_local_state
from qibo.config import raise_error
from qibo.hamiltonians import Hamiltonian, SymbolicHamiltonian
from qibo.models.algorithms._pauli_helpers import _pauli_rotation, _pauli_terms
from qibo.models.circuit import Circuit


def RandomizedTrotter(
    hamiltonian: (
        Hamiltonian
        | SymbolicHamiltonian
        | list[tuple[str, float]]
        | tuple[tuple[str, float], ...]
    ),
    time: float,
    nsteps: int = 1,
    order: int = 1,
    seed: int | np.random.Generator | None = None,
    backend: Backend | None = None,
    **kwargs,
) -> Circuit:
    """Creates a random circuit of the randomized product formula of a Hamiltonian.

    Let :math:`H = \\sum_{j = 1}^{L} h_{j} P_{j}` be a Hamiltonian written as a sum of Pauli
    strings :math:`P_{j}` with real coefficients :math:`h_{j}`. The circuit approximates the
    evolution :math:`e^{-i H t}` during the time :math:`t` with :math:`r` segments, each one
    with a random order of the terms that is drawn independently of the other segments, as
    in Ref. [1]. Let :math:`\\lambda = -i t / r`.

    For the first order, the segment is one of the two formulas

    .. math::
        S_{1}(\\lambda) = \\prod_{j = 1}^{L} e^{\\lambda h_{j} P_{j}} \\quad \\text{or} \\quad
        S_{1}^{\\mathrm{rev}}(\\lambda) = \\prod_{j = L}^{1} e^{\\lambda h_{j} P_{j}} \\, ,

    which have the terms in the forward and in the reverse order, and it is chosen
    with a random bit that is :math:`0` or :math:`1` with probability :math:`1 / 2`. For an even
    order :math:`2k`, the segment is :math:`S_{2k}^{\\sigma}(\\lambda)`, where :math:`\\sigma` is a
    permutation of the :math:`L` terms drawn with the uniform distribution, and

    .. math::
        S_{2}^{\\sigma}(\\lambda) = \\prod_{j = 1}^{L} e^{\\frac{\\lambda}{2} h_{\\sigma(j)}
        P_{\\sigma(j)}} \\prod_{j = L}^{1} e^{\\frac{\\lambda}{2} h_{\\sigma(j)}
        P_{\\sigma(j)}} \\, ,

    .. math::
        S_{2k}^{\\sigma}(\\lambda) = S_{2k - 2}^{\\sigma}(p_{k} \\lambda)^{2} \\,
        S_{2k - 2}^{\\sigma}((1 - 4 p_{k}) \\lambda) \\, S_{2k - 2}^{\\sigma}(p_{k} \\lambda)^{2}
        \\, , \\quad p_{k} = \\frac{1}{4 - 4^{1 / (2k - 1)}} \\, ,

    in which the same permutation :math:`\\sigma` is used in all of the factors of the segment
    (Eq. 46 of Ref. [1]), as in the fractal decomposition of :func:`qibo.models.Trotter`. The
    circuit applies the rotation of the first of the terms of a product first, which is the same
    as reversing the products above, and the ensembles of the segments are the same, since both
    the pair of forward and reverse orders and the uniform permutations are symmetric.
    Adjacent rotations of the same Pauli string, at the boundaries of the segments, of the
    factors of the recursion and in the middle of the second order, are merged into one.

    Let :math:`\\mathcal{V}` be the channel of :math:`e^{-i H t}`, :math:`\\mathcal{S}`
    the average of the channels of the segment over its random choice, and
    :math:`\\Lambda = \\max_{j} \\| h_{j} P_{j} \\| = \\max_{j} |h_{j}|`. The error of the average
    of the circuits, which is the channel :math:`\\mathcal{S}^{r}`, is bounded in the diamond norm
    :math:`\\| \\cdot \\|_{\\diamond}` by the following results of Ref. [1], which are not
    valid for a single circuit. For the first order (Theorem 1 of Ref. [1], with the
    average of the forward and the reverse orders),

    .. math::
        \\| \\mathcal{V} - \\mathcal{S}^{r} \\|_{\\diamond} \\leq
        \\frac{(\\Lambda t L)^{4}}{r^{3}} e^{2 \\Lambda t L / r}
        + \\frac{2 (\\Lambda t L)^{3}}{3 r^{2}} e^{\\Lambda t L / r} \\, ,

    so that :math:`O((t L)^{1.5} / \\epsilon^{0.5})` segments, and
    :math:`O(t^{1.5} L^{2.5} / \\epsilon^{0.5})` exponentials of the terms, suffice for the error
    :math:`\\epsilon` if :math:`\\Lambda` is constant, against :math:`O(t^{2} L^{3} /
    \\epsilon)` of the deterministic formula. For the even order :math:`2k` (Theorem 2
    of Ref. [1], with the uniform average over the permutations),

    .. math::
        \\| \\mathcal{V} - \\mathcal{S}^{r} \\|_{\\diamond} \\leq
        4 \\frac{(2 \\cdot 5^{k - 1} \\Lambda t L)^{4k + 2}}{((2k + 1)!)^{2} \\, r^{4k + 1}}
        e^{4 \\cdot 5^{k - 1} \\Lambda t L / r}
        + 2 \\frac{(2 \\cdot 5^{k - 1} \\Lambda t)^{2k + 1} L^{2k}}{(2k - 1)! \\, r^{2k}}
        e^{2 \\cdot 5^{k - 1} \\Lambda t L / r} \\, .

    The bounds hold for the average of the random circuits and not for each one of them, so the
    error of the evolution has to be estimated by averaging over several circuits, each one
    drawn with its own random choices.

    .. note::
        Each call draws a new random circuit. An integer in ``seed`` makes the circuit
        reproducible, while a generator of random numbers, instead of an integer, has to be given
        in ``seed`` to draw different circuits in successive calls.

    Example:
        Evolve two qubits with a Hamiltonian of non-commuting terms. The distance in the spectral
        norm to the exact evolution decreases as the number of steps grows.

        .. testcode::

            import numpy as np
            from scipy.linalg import expm

            from qibo.models import RandomizedTrotter

            hamiltonian = [("XX", 1.0), ("ZI", 0.8), ("IZ", 0.6), ("YY", 0.4)]
            time = 1.0

            matrices = {
                "I": np.eye(2),
                "X": np.array([[0, 1], [1, 0]]),
                "Y": np.array([[0, -1j], [1j, 0]]),
                "Z": np.diag([1, -1]),
            }
            matrix = sum(
                coefficient * np.kron(matrices[pauli[0]], matrices[pauli[1]])
                for pauli, coefficient in hamiltonian
            )
            exact = expm(-1j * time * matrix)

            errors = []
            for nsteps in (1, 10, 100):
                circuit = RandomizedTrotter(hamiltonian, time, nsteps, order=2, seed=1)
                errors.append(np.linalg.norm(circuit.unitary() - exact, 2))

            print(errors[0] > errors[1] > errors[2])

        .. testoutput::

            True

    Args:
        hamiltonian (:class:`qibo.hamiltonians.Hamiltonian`,
            :class:`qibo.hamiltonians.SymbolicHamiltonian`, list or tuple): Hamiltonian, or list
            or tuple of tuples ``(pauli, coefficient)`` of its Pauli decomposition. The string
            ``pauli`` has one letter from ``"IXYZ"`` for each qubit, and its :math:`j`-th letter
            is the Pauli matrix applied to the :math:`j`-th qubit, *e.g.* ``"XIZ"`` is
            :math:`X_{0} Z_{2}`. The coefficients are real. A symbolic Hamiltonian made of Pauli
            symbols gives its strings directly. Otherwise, they are computed from the matrix of
            the Hamiltonian with :func:`qibo.quantum_info.pauli_basis`, which is inefficient in
            the number of qubits, and a warning is logged.
        time (float): evolution time :math:`t`.
        nsteps (int, optional): number of steps :math:`r`. Defaults to ``1``.
        order (int, optional): order of the product formula, which is one or an even integer.
            Defaults to ``1``.
        seed (int or :class:`numpy.random.Generator`, optional): seed or generator used to draw
            the random bits (first order) or the permutations (even orders). An integer seed starts
            the same sequence of draws in each call. If ``None``, it starts a generator with a
            random seed. Defaults to ``None``.
        backend (:class:`qibo.backends.abstract.Backend`, optional): backend to be
            used in the calculation. If ``None``, it uses the current backend.
            Defaults to ``None``.
        kwargs (dict, optional): Additional arguments used to initialize
            :class:`qibo.models.circuit.Circuit`, *e.g.* ``density_matrix=True``. For details,
            see the documentation of :class:`qibo.models.circuit.Circuit`.

    Returns:
        :class:`qibo.models.circuit.Circuit`: random circuit of the product formula. The Pauli
        rotations are made of Clifford gates and a single :class:`qibo.gates.RZ` gate, and the
        terms of the identity are :class:`qibo.gates.GlobalPhase` gates.

    References:
        1. A. M. Childs, A. Ostrander, and Y. Su, *Faster quantum simulation by randomization*,
        `Quantum 3, 182 (2019) <https://doi.org/10.22331/q-2019-09-02-182>`_.

        2. M. Suzuki, *General theory of fractal path integrals with applications to many-body
        theories and statistical physics*, `J. Math. Phys. 32, 400 (1991)
        <https://doi.org/10.1063/1.529425>`_.
    """
    backend, local_state = _check_backend_and_local_state(seed, backend)

    if isinstance(order, bool) or not isinstance(order, Integral) or order < 1:
        raise_error(
            ValueError, f"``order`` must be a positive integer, but it is {order}."
        )

    if order > 1 and order % 2 != 0:
        raise_error(
            ValueError, f"``order`` must be one or an even integer, but it is {order}."
        )

    if isinstance(nsteps, bool) or not isinstance(nsteps, Integral) or nsteps < 1:
        raise_error(
            ValueError, f"``nsteps`` must be a positive integer, but it is {nsteps}."
        )

    terms = _pauli_terms(hamiltonian, None, "hamiltonian", True, backend)
    nqubits = len(terms[0][0])

    # the terms with a zero coefficient have no rotation
    rotations = [
        (pauli, 2 * coefficient * time / nsteps)
        for pauli, coefficient in terms
        if coefficient != 0.0
    ]

    # adjacent rotations of the same Pauli string, at the boundaries of the steps and of the
    # recursion of the product formula, are merged
    merged = []
    for _ in range(nsteps):
        if order == 1:
            # forward or reverse order of the terms, with probability 1/2
            reverse = int(
                backend.random_choice(2, size=1, seed=local_state, dtype=backend.int64)[
                    0
                ]
            )
            step = rotations[::-1] if reverse else rotations
        else:
            # uniform permutation of the terms, the same one in the whole formula
            permutation = backend.random_choice(
                len(rotations),
                size=len(rotations),
                replace=False,
                seed=local_state,
                dtype=backend.int64,
            )
            step = _suzuki([rotations[int(index)] for index in permutation], order)
        for pauli, angle in step:
            if len(merged) > 0 and merged[-1][0] == pauli:
                merged[-1] = (pauli, merged[-1][1] + angle)
            else:
                merged.append((pauli, angle))

    circuit = Circuit(nqubits, **kwargs)
    for pauli, angle in merged:
        circuit.add(_pauli_rotation(pauli, angle))

    return circuit


def Trotter(
    hamiltonian: (
        Hamiltonian
        | SymbolicHamiltonian
        | list[tuple[str, float]]
        | tuple[tuple[str, float], ...]
    ),
    time: float,
    nsteps: int = 1,
    order: int = 1,
    backend: Backend = None,
    **kwargs,
) -> Circuit:
    """Creates the circuit of the Trotter-Suzuki product formula of a Hamiltonian.

    Let :math:`H = \\sum_{j} h_{j} P_{j}` be a Hamiltonian written as a sum of Pauli strings
    :math:`P_{j}` with real coefficients :math:`h_{j}`. The circuit approximates the evolution
    :math:`e^{-i H t}` during the time :math:`t` with :math:`r` steps. For the first order,

    .. math::
        e^{-i H t} \\approx \\left( \\prod_{j} e^{-i h_{j} P_{j} t / r} \\right)^{r} \\, ,

    and for the second order, which is symmetric,

    .. math::
        e^{-i H t} \\approx \\left( \\prod_{j} e^{-i h_{j} P_{j} t / 2 r} \\prod_{j}^{\\leftarrow}
        e^{-i h_{j} P_{j} t / 2 r} \\right)^{r} \\, ,

    where the arrow is the reversed order of the terms. In both cases, the Pauli rotations of a
    step are applied from the first term, which is the rightmost factor, to the last one. Each
    rotation is made of Clifford gates and a single :class:`qibo.gates.RZ` gate, and the terms of
    the identity are :class:`qibo.gates.GlobalPhase` gates.

    The symmetric formulas of the higher even orders :math:`2k` are built recursively with the
    fractal decomposition of Ref. [3], where :math:`S_{2k}(\\tau)` is the formula of a step of
    time :math:`\\tau`,

    .. math::
        S_{2k}(\\tau) = S_{2k - 2}(p \\tau)^{2} \\, S_{2k - 2}((1 - 4 p) \\tau) \\,
        S_{2k - 2}(p \\tau)^{2} \\, , \\quad p = \\frac{1}{4 - 4^{1 / (2k - 1)}} \\, ,

    and :math:`S_{2} (\\tau)` is the second order. One of the five factors has a negative time,
    since a decomposition with only positive times does not exist beyond the second order. An odd
    order other than one is not accepted, since the symmetric formulas have only even orders.

    The error is :math:`O(t^{p + 1} / r^{p})` for the order :math:`p`, with the commutators of the
    terms in the constant (Ref. [2]). Hence, it vanishes if the terms commute, and a higher order
    needs fewer steps for a small error, at the cost of more rotations in each one. Adjacent
    rotations of the same Pauli string, which are in the boundaries of the steps and of the
    recursion, are merged into one. For :math:`L` terms, a step of the order :math:`2k` has at
    most :math:`2 (L - 1) 5^{k - 1} + 1` rotations.

    Example:
        Evolve a qubit with :math:`H = X + Z`. The distance in the spectral norm to the exact
        evolution decreases as the number of steps grows, and faster for a higher order.

        .. testcode::

            import numpy as np
            from scipy.linalg import expm

            from qibo.models import Trotter

            hamiltonian = [("X", 1.0), ("Z", 1.0)]
            time = 1.0
            exact = expm(-1j * time * np.array([[1.0, 1.0], [1.0, -1.0]]))

            for order in (1, 2, 4):
                for nsteps in (1, 10, 100):
                    circuit = Trotter(hamiltonian, time, nsteps, order)
                    error = np.linalg.norm(circuit.unitary() - exact, 2)

                    print(order, nsteps, f"{error:.1e}")

        .. testoutput::

            1 1 8.0e-01
            1 10 7.0e-02
            1 100 7.0e-03
            2 1 3.1e-01
            2 10 2.7e-03
            2 100 2.7e-05
            4 1 1.3e-02
            4 10 1.1e-06
            4 100 1.1e-10

    Args:
        hamiltonian (:class:`qibo.hamiltonians.Hamiltonian`,
            :class:`qibo.hamiltonians.SymbolicHamiltonian`, list or tuple): Hamiltonian, or list
            or tuple of tuples ``(pauli, coefficient)`` of its Pauli decomposition. The string
            ``pauli`` has one letter from ``"IXYZ"`` for each qubit, and its :math:`j`-th letter
            is the Pauli matrix applied to the :math:`j`-th qubit, *e.g.* ``"XIZ"`` is
            :math:`X_{0} Z_{2}`. The coefficients are real. A symbolic Hamiltonian made of Pauli
            symbols gives its strings directly. Otherwise, they are computed from the matrix of
            the Hamiltonian with :func:`qibo.quantum_info.pauli_basis`, which is inefficient in
            the number of qubits, and a warning is logged.
        time (float): evolution time :math:`t`.
        nsteps (int, optional): number of steps :math:`r`. Defaults to ``1``.
        order (int, optional): order of the product formula, which is one or an even integer.
            Defaults to ``1``.
        backend (:class:`qibo.backends.abstract.Backend`, optional): backend to be
            used in the calculation. If ``None``, it uses the current backend.
            Defaults to ``None``.
        kwargs (dict, optional): Additional arguments used to initialize
            :class:`qibo.models.circuit.Circuit`, *e.g.* ``density_matrix=True``. For details,
            see the documentation of :class:`qibo.models.circuit.Circuit`.

    Returns:
        :class:`qibo.models.circuit.Circuit`: circuit of the product formula.

    References:
        1. S. Lloyd, *Universal quantum simulators*, `Science 273, 1073 (1996)
        <https://doi.org/10.1126/science.273.5278.1073>`_.

        2. A. M. Childs, Y. Su, M. C. Tran, N. Wiebe, and S. Zhu, *Theory of Trotter error with
        commutator scaling*, `Phys. Rev. X 11, 011020 (2021)
        <https://doi.org/10.1103/PhysRevX.11.011020>`_.

        3. M. Suzuki, *General theory of fractal path integrals with applications to many-body
        theories and statistical physics*, `J. Math. Phys. 32, 400 (1991)
        <https://doi.org/10.1063/1.529425>`_.
    """
    backend = _check_backend(backend)

    if isinstance(order, bool) or not isinstance(order, Integral) or order < 1:
        raise_error(
            ValueError, f"``order`` must be a positive integer, but it is {order}."
        )

    if order > 1 and order % 2 != 0:
        raise_error(
            ValueError, f"``order`` must be one or an even integer, but it is {order}."
        )

    if isinstance(nsteps, bool) or not isinstance(nsteps, Integral) or nsteps < 1:
        raise_error(
            ValueError, f"``nsteps`` must be a positive integer, but it is {nsteps}."
        )

    terms = _pauli_terms(hamiltonian, None, "hamiltonian", True, backend)
    nqubits = len(terms[0][0])

    # the terms with a zero coefficient have no rotation
    terms = [(pauli, coefficient) for pauli, coefficient in terms if coefficient != 0.0]

    rotations = [
        (pauli, 2 * coefficient * time / nsteps) for pauli, coefficient in terms
    ]
    step = _suzuki(rotations, order) if order > 1 else rotations

    # adjacent rotations of the same Pauli string, at the boundaries of the steps and of the
    # recursion of the product formula, are merged
    merged = []
    for _ in range(nsteps):
        for pauli, angle in step:
            if len(merged) > 0 and merged[-1][0] == pauli:
                merged[-1] = (pauli, merged[-1][1] + angle)
            else:
                merged.append((pauli, angle))

    circuit = Circuit(nqubits, **kwargs)
    for pauli, angle in merged:
        circuit.add(_pauli_rotation(pauli, angle))

    return circuit


def _suzuki(rotations: list[tuple[str, float]], order: int) -> list[tuple[str, float]]:
    """Creates the rotations of the symmetric product formula of an even order.

    The second order applies the rotations of all of the terms but the last one at half of their
    angles, the last one, and the first ones again in the reversed order. A higher order
    :math:`2k` applies five times the one of the order :math:`2k - 2`, whose angles are
    multiplied by :math:`p, p, 1 - 4 p, p, p`, where :math:`p = 1 / (4 - 4^{1 / (2k - 1)})`.

    Args:
        rotations (list): tuples ``(pauli, angle)`` of the Pauli string and the angle of the
            rotation of each term, in the order of the terms.
        order (int): even order of the product formula.

    Returns:
        list: tuples ``(pauli, angle)`` of the rotations of the product formula, in the order that
        they are applied.
    """
    if order == 2:
        half = [(pauli, angle / 2) for pauli, angle in rotations[:-1]]

        formula = []
        formula.extend(half)
        formula.append(rotations[-1])
        formula.extend(reversed(half))

        return formula

    fraction = 1 / (4 - 4 ** (1 / (order - 1)))

    formula = []
    for factor in (fraction, fraction, 1 - 4 * fraction, fraction, fraction):
        formula.extend(
            _suzuki([(pauli, factor * angle) for pauli, angle in rotations], order - 2)
        )

    return formula
