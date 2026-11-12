import itertools
from numbers import Integral

import numpy as np

from qibo import gates
from qibo.backends import Backend, _check_backend_and_local_state
from qibo.config import raise_error
from qibo.hamiltonians import Hamiltonian, SymbolicHamiltonian
from qibo.models.algorithms._pauli_terms import _pauli_terms
from qibo.models.circuit import Circuit


def qdrift(
    hamiltonian: (
        Hamiltonian
        | SymbolicHamiltonian
        | list[tuple[str, float]]
        | tuple[tuple[str, float], ...]
    ),
    time: float,
    epsilon: float | None = None,
    nsteps: int | None = None,
    seed: int | np.random.Generator | None = None,
    backend: Backend = None,
    **kwargs,
) -> Circuit:
    """Creates a random circuit of the quantum stochastic drift protocol (qDRIFT).

    Let :math:`H = \\sum_{k} h_{k} P_{k}` be a Hamiltonian written as a sum of Pauli strings
    :math:`P_{k}` with real coefficients :math:`h_{k}`, and let :math:`\\lambda = \\sum_{k}
    |h_{k}|`. The circuit has :math:`N` steps. Each step draws an index :math:`k` with
    probability :math:`|h_{k}| / \\lambda`, independently of the other steps, and applies

    .. math::
        e^{-i \\tau \\, \\mathrm{sgn}(h_{k}) P_{k}}, \\quad \\tau = \\frac{\\lambda t}{N} \\, ,

    which is a Pauli rotation with the same angle for every term. The number of steps does not
    depend on the number of terms of the Hamiltonian. If :math:`N = \\lceil 2 \\lambda^{2} t^{2}
    / \\epsilon \\rceil`, the average of the random circuits is :math:`\\epsilon`-close, in the
    diamond norm, to :math:`e^{-i H t}`.

    .. note::
        The error :math:`\\epsilon` refers to the average of the circuits, not to a single
        circuit, whose error is of the order of :math:`\\sqrt{\\epsilon}`. Therefore, the
        expected values must be estimated with a new random circuit for each run. A generator of
        random numbers, instead of an integer, has to be given in ``seed`` to draw
        different circuits.

    Example:
        Average the expected value of :math:`Z` over random circuits that evolve
        :math:`\\ket{0}` with :math:`H = 0.6 X + 0.8 Z`, and compare it with the exact one.

        .. testcode::

            import math

            import numpy as np

            from qibo.hamiltonians import SymbolicHamiltonian
            from qibo.models import qdrift
            from qibo.symbols import X, Z

            hamiltonian = SymbolicHamiltonian(0.6 * X(0) + 0.8 * Z(0))
            observable = SymbolicHamiltonian(Z(0))
            time = 0.5

            rng = np.random.default_rng(7)
            values = []
            for _ in range(400):
                circuit = qdrift(hamiltonian, time, epsilon=0.05, seed=rng)
                values.append(observable.expectation_from_state(circuit().state()))

            # the exact value is 0.8^2 + 0.6^2 cos(2t), with 0.6 X + 0.8 Z = n . sigma
            exact = 0.64 + 0.36 * math.cos(2 * time)

            print(abs(np.mean(values) - exact) < 0.25)

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
        epsilon (float, optional): precision of the average of the circuits, which sets the
            number of steps :math:`N = \\lceil 2 \\lambda^{2} t^{2} / \\epsilon \\rceil`, with at
            least one step. Exactly one of ``epsilon`` and ``nsteps`` must be given.
            Defaults to ``None``.
        nsteps (int, optional): number of steps :math:`N` of the circuit. Exactly one of
            ``epsilon`` and ``nsteps`` must be given. Defaults to ``None``.
        seed (int or :class:`numpy.random.Generator`, optional): seed or generator used to draw the
            steps. An integer seed starts the same sequence of steps in each call. If ``None``,
            it starts a generator with a random seed. Defaults to ``None``.
        backend (:class:`qibo.backends.abstract.Backend`, optional): backend to be
            used in the calculation. If ``None``, it uses the current backend.
            Defaults to ``None``.
        kwargs (dict, optional): Additional arguments used to initialize
            :class:`qibo.models.circuit.Circuit`, *e.g.* ``density_matrix=True``. For details,
            see the documentation of :class:`qibo.models.circuit.Circuit`.

    Returns:
        :class:`qibo.models.circuit.Circuit`: random circuit of the protocol. The Pauli rotations
        are made of Clifford gates and a single :class:`qibo.gates.RZ` gate, and the terms of the
        identity are :class:`qibo.gates.GlobalPhase` gates.

    References:
        1. E. Campbell, *Random compiler for fast Hamiltonian simulation*,
        `Phys. Rev. Lett. 123, 070503 (2019) <https://doi.org/10.1103/PhysRevLett.123.070503>`_.
    """
    backend, local_state = _check_backend_and_local_state(seed, backend)

    if (epsilon is None) == (nsteps is None):
        raise_error(ValueError, "exactly one of epsilon and nsteps must be given.")

    if epsilon is not None and not epsilon > 0:
        raise_error(ValueError, f"epsilon must be positive, but it is {epsilon}.")

    if nsteps is not None and (
        isinstance(nsteps, bool) or not isinstance(nsteps, Integral) or nsteps < 1
    ):
        raise_error(
            ValueError, f"nsteps must be a positive integer, but it is {nsteps}."
        )

    terms = _pauli_terms(hamiltonian, None, backend)
    nqubits = len(terms[0][0])

    coefficients = backend.cast(
        [coefficient for _, coefficient in terms], dtype=backend.complex128
    )

    if backend.any(
        backend.abs(backend.imag(coefficients))
        > 1e-12 * backend.max(backend.abs(coefficients))
    ):
        raise_error(
            ValueError, "the coefficients of the Pauli strings must be real numbers."
        )

    coefficients = backend.real(coefficients)
    weights = backend.abs(coefficients)
    signs = backend.sign(coefficients)
    norm = backend.sum(weights)

    if nsteps is None:
        nsteps = max(1, int(backend.ceil(2 * (norm * time) ** 2 / epsilon)))
    tau = norm * time / nsteps

    indices = backend.random_choice(
        len(terms),
        size=nsteps,
        p=weights / norm,
        seed=local_state,
        dtype=backend.int64,
    )

    circuit = Circuit(nqubits, **kwargs)
    for index in indices:
        index = int(index)
        pauli = terms[index][0]
        angle = float(2 * tau * signs[index])
        targets = [qubit for qubit, letter in enumerate(pauli) if letter != "I"]

        # the identity is a global phase
        if len(targets) == 0:
            circuit.add(gates.GlobalPhase(-angle / 2))
            continue

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
        ladder = [gates.CNOT(*pair) for pair in itertools.pairwise(targets)]

        circuit.add(before)
        circuit.add(ladder)
        circuit.add(gates.RZ(targets[-1], theta=angle))
        circuit.add(ladder[::-1])
        circuit.add(after)

    return circuit
