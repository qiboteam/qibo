from numbers import Integral

from qibo.backends import Backend, _check_backend
from qibo.config import raise_error
from qibo.hamiltonians import Hamiltonian, SymbolicHamiltonian
from qibo.models.algorithms._pauli_helpers import _pauli_rotation, _pauli_terms
from qibo.models.circuit import Circuit


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

    where the arrow is the reversed order of the terms, and the two rotations of the last term
    are merged into one. In both cases, the Pauli rotations of a step are applied from the
    first term, which is the rightmost factor, to the last one. Each rotation is made of
    Clifford gates and a single :class:`qibo.gates.RZ` gate, and the terms of the identity are
    :class:`qibo.gates.GlobalPhase` gates. The error is :math:`O(t^{p + 1} / r^{p})` for the
    order :math:`p`, with the commutators of the terms in the constant (Ref. [2]). Hence, it
    vanishes if the terms commute, and the second order needs fewer steps than the first one
    for a small error.

    Example:
        Evolve a qubit with :math:`H = X + Z`. The distance in the spectral norm to the exact
        evolution decreases as the number of steps grows, and faster for the second order.

        .. testcode::

            import numpy as np
            from scipy.linalg import expm

            from qibo.models import Trotter

            hamiltonian = [("X", 1.0), ("Z", 1.0)]
            time = 1.0
            exact = expm(-1j * time * np.array([[1.0, 1.0], [1.0, -1.0]]))

            for order in (1, 2):
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
        order (int, optional): order of the product formula, which is at least one. Only the first
            and the second orders are implemented. Defaults to ``1``.
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
    """
    backend = _check_backend(backend)

    if isinstance(order, bool) or not isinstance(order, Integral) or order < 1:
        raise_error(
            ValueError, f"``order`` must be a positive integer, but it is {order}."
        )

    if order > 2:
        raise_error(
            NotImplementedError,
            f"Only the first and second orders are implemented, but order is {order}.",
        )

    if isinstance(nsteps, bool) or not isinstance(nsteps, Integral) or nsteps < 1:
        raise_error(
            ValueError, f"``nsteps`` must be a positive integer, but it is {nsteps}."
        )

    terms = _pauli_terms(hamiltonian, None, "hamiltonian", True, backend)
    nqubits = len(terms[0][0])

    # the terms with a zero coefficient have no rotation
    terms = [(pauli, coefficient) for pauli, coefficient in terms if coefficient != 0.0]

    circuit = Circuit(nqubits, **kwargs)
    for _ in range(nsteps):
        rotations = [
            (pauli, 2 * coefficient * time / nsteps) for pauli, coefficient in terms
        ]

        # the second order is symmetric, and the two rotations of the last term are merged
        step = [
            (pauli, angle / 2 if order == 2 else angle)
            for pauli, angle in rotations[:-1]
        ]
        step.append(rotations[-1])
        if order == 2:
            step.extend((pauli, angle / 2) for pauli, angle in reversed(rotations[:-1]))

        for pauli, angle in step:
            circuit.add(_pauli_rotation(pauli, angle))

    return circuit
