"""Direct fidelity estimation with Pauli measurements."""

import math
from collections import Counter

from numpy.typing import ArrayLike

from qibo import Circuit
from qibo.backends import Backend, _check_backend
from qibo.config import raise_error
from qibo.quantum_info._superoperator_transformations import (
    _fast_walsh_hadamard_transform,
)
from qibo.tomography.abstract import Tomography


class DirectFidelityEstimation(Tomography):
    """Direct fidelity estimation of a pure state with Pauli measurements.

    The fidelity between the pure target state :math:`|\\psi\\rangle` of :math:`n` qubits and the
    state :math:`\\sigma` that is actually prepared is

    .. math::
        F = \\langle \\psi | \\sigma | \\psi \\rangle
        = \\sum_{P} \\chi_{\\psi}(P) \\, \\chi_{\\sigma}(P) \\, ,

    where the sum runs over the :math:`4^{n}` Pauli strings :math:`P`, and
    :math:`\\chi_{\\rho}(P) = \\text{tr}(P \\, \\rho) / \\sqrt{d}`, with :math:`d = 2^{n}`, is the
    characteristic function of :math:`\\rho`. Since :math:`\\text{tr}(\\psi^{2}) = 1`, the numbers
    :math:`\\chi_{\\psi}(P)^{2}` are a probability distribution, which is used to estimate
    :math:`F` without reconstructing :math:`\\sigma`, as follows [1, 2].

    1. :math:`\\ell = \\lceil 1 / (\\epsilon^{2} \\, \\delta) \\rceil` Pauli strings :math:`P_{i}`
       are drawn from the probability :math:`\\chi_{\\psi}(P)^{2}`, so that the strings that
       are the most informative about the target state are the most likely ones.
    2. Each :math:`P_{i}` is measured :math:`m_{i}` times on the prepared state, with

       .. math::
           m_{i} = \\left\\lceil \\frac{2 \\ln(2 / \\delta)}
           {\\langle \\psi | P_{i} | \\psi \\rangle^{2} \\, \\ell \\, \\epsilon^{2}}
           \\right\\rceil \\, ,

       which gives the outcomes :math:`A_{ij} \\in \\{+1, -1\\}`. The outcomes are the product
       of the eigenvalues of the qubits in the support of :math:`P_{i}`.
    3. The estimate is

       .. math::
           \\tilde{Y} = \\frac{1}{\\ell} \\sum_{i = 1}^{\\ell}
           \\frac{1}{m_{i} \\, \\langle \\psi | P_{i} | \\psi \\rangle}
           \\sum_{j = 1}^{m_{i}} A_{ij} \\, .

    For any :math:`\\sigma`, the fidelity satisfies
    :math:`|F - \\tilde{Y}| \\leq 2 \\epsilon` with probability at least :math:`1 - 2 \\delta`,
    independently of :math:`n`. The number of strings does not depend on the size of the system,
    and the expected number of shots is at most
    :math:`1 + 1 / (\\epsilon^{2} \\, \\delta) + 2 d \\ln(2 / \\delta) / \\epsilon^{2}`,
    which is a factor :math:`d` smaller than full tomography with Pauli measurements [1].

    The estimator is unbiased, but a single estimate can be outside the interval :math:`[0, 1]`.
    A string that is drawn :math:`c` times is measured by one circuit with :math:`c \\, m_{i}`
    shots, which is equivalent to :math:`c` independent measurements of :math:`m_{i}` shots. The
    number of circuits is then at most the number of strings with a non-zero probability.

    The target state is simulated classically to draw the Pauli strings, but the table of the
    :math:`4^{n}` probabilities :math:`\\chi_{\\psi}(P)^{2}` is never built. Writing a string as
    :math:`P \\propto X^{a} Z^{b}`, with :math:`a` and :math:`b` bit strings, the sampling has two
    stages. First, :math:`a` is drawn from its marginal probability

    .. math::
        p(a) = \\sum_{x} |\\psi(x)|^{2} \\, |\\psi(x \\oplus a)|^{2} \\, ,

    which is the XOR-autocorrelation of :math:`|\\psi(x)|^{2}`, so that two fast Walsh-Hadamard
    transforms give it for all :math:`a`. Then, :math:`b` is drawn from the probability, given
    :math:`a`, that is proportional to

    .. math::
        \\langle \\psi | X^{a} Z^{b} | \\psi \\rangle^{2}
        = \\left| \\sum_{x} \\psi^{*}(x \\oplus a) \\, \\psi(x) \\, (-1)^{b \\cdot x}
        \\right|^{2} \\, ,

    which is the square of a Walsh-Hadamard transform in :math:`b`, whose values are, up to the
    phase :math:`i^{a \\cdot b}` of the Pauli :math:`Y`, the expectation values that the estimator
    needs. The draws of :math:`a` are grouped, so that each different :math:`a` is processed once,
    and its transform is re-used by all the strings that have it. Only the expectation values of
    the strings that were drawn are kept. The memory grows as :math:`2^{n}`, like the state vector
    of the target, and the time as :math:`\\mathcal{O}(D \\, n \\, 2^{n})`, with
    :math:`D \\leq \\min(\\ell, 2^{n})` the number of different :math:`a`. The state that is
    prepared is only sampled, so it can have noise channels.

    Example:
        .. testcode::

            from qibo import Circuit, gates
            from qibo.tomography import DirectFidelityEstimation

            circuit = Circuit(3)
            circuit.add(gates.H(0))
            circuit.add(gates.CNOT(0, 1))
            circuit.add(gates.CNOT(1, 2))

            # fidelity of the GHZ state, which is one up to the statistical error
            fidelity = DirectFidelityEstimation()(circuit, epsilon=0.1, delta=0.1, seed=1)

            # fidelity of a noisy preparation with respect to the ideal GHZ state
            noisy = Circuit(3, density_matrix=True)
            noisy.add(circuit.queue)
            noisy.add(gates.DepolarizingChannel((0, 1, 2), 0.1))
            fidelity = DirectFidelityEstimation()(noisy, target=circuit, seed=1)

    References:
        1. S. T. Flammia and Y.-K. Liu,
           *Direct fidelity estimation from few Pauli measurements*,
           `Phys. Rev. Lett. 106, 230501 (2011) <https://doi.org/10.1103/PhysRevLett.106.230501>`_.
        2. M. P. da Silva, O. Landon-Cardinal and D. Poulin,
           *Practical characterization of quantum devices without tomography*,
           `Phys. Rev. Lett. 107, 210404 (2011) <https://doi.org/10.1103/PhysRevLett.107.210404>`_.
    """

    def __call__(
        self,
        circuit: Circuit,
        target: Circuit | ArrayLike | None = None,
        epsilon: float = 0.1,
        delta: float = 0.1,
        seed: int | None = None,
        backend: Backend = None,
    ) -> float:
        """Estimates the fidelity between the state prepared by ``circuit`` and ``target``.

        Args:
            circuit (:class:`qibo.models.circuit.Circuit`): circuit that prepares the state
                whose fidelity is estimated, without measurements. It can have noise channels
                and be a circuit of density matrices.
            target (:class:`qibo.models.circuit.Circuit` or ArrayLike, optional): pure state
                that ``circuit`` should prepare, as a state vector or as a circuit without
                measurements, noise channels, and density matrices. It is simulated by
                ``backend``. If ``None``, the target is the state prepared by ``circuit``
                itself, which is only meaningful if ``circuit`` is noiseless. Defaults to
                ``None``.
            epsilon (float, optional): additive error :math:`\\epsilon` of the estimate, in the
                interval :math:`(0, 1)`. The fidelity is within :math:`2 \\epsilon` of the
                estimate, with probability at least :math:`1 - 2 \\delta`. Defaults to
                :math:`0.1`.
            delta (float, optional): failure probability :math:`\\delta`, in the interval
                :math:`(0, 1)`. Defaults to :math:`0.1`.
            seed (int, optional): seed of the sampling of the Pauli strings and of their
                outcomes. If ``None``, the generator of the backend is not reseeded for the
                outcomes. Defaults to ``None``.
            backend (:class:`qibo.backends.abstract.Backend`, optional): backend
                to be used in the execution. If ``None``, it uses the current
                backend. Defaults to ``None``.

        Returns:
            float: estimate :math:`\\tilde{Y}` of the fidelity.
        """
        for name, value in (("epsilon", epsilon), ("delta", delta)):
            if (
                isinstance(value, bool)
                or not isinstance(value, int | float)
                or not 0 < value < 1
            ):
                raise_error(
                    ValueError,
                    f"``{name}`` must be a number in the interval (0, 1), "
                    + f"but it is {value}.",
                )

        backend = _check_backend(backend)

        nqubits = circuit.nqubits
        dim = 2**nqubits

        target = circuit if target is None else target
        if isinstance(target, Circuit):
            self._check_circuit(target)
            if target.repeated_execution:
                raise_error(
                    ValueError, "``target`` must be a circuit without noise channels."
                )
            target = backend.execute_circuit(target).state()

        state = backend.cast(target, dtype=backend.complex128)
        if (
            tuple(state.shape) != (dim,)
            or abs(float(backend.sum(backend.abs(state) ** 2)) - 1.0) > 1e-6
        ):
            raise_error(
                ValueError,
                f"``target`` must be a normalized state vector of {dim} elements, "
                + "or a circuit that prepares one.",
            )

        nsamples = math.ceil(1 / (epsilon**2 * delta))

        # a single generator for all the draws, since the same seed would give correlated
        # ones
        generator = None if seed is None else backend.default_rng(seed)

        # the Pauli strings are X^a Z^b, with a Y where both bits are one, so that each
        # qubit is I, X, Z, Y for a + 2 b = 0, 1, 2, 3. The first stage draws the a from
        # their marginal probability, which is the XOR-autocorrelation of the
        # probabilities of the target state, computed with Walsh-Hadamard transforms
        spectrum = _fast_walsh_hadamard_transform(
            backend.abs(state) ** 2, backend=backend
        )
        marginal = backend.maximum(
            _fast_walsh_hadamard_transform(spectrum**2, backend=backend), 0.0
        )
        patterns = Counter(
            backend.to_numpy(
                backend.random_choice(
                    dim,
                    size=nsamples,
                    p=marginal / backend.sum(marginal),
                    seed=generator,
                    dtype=backend.int64,
                )
            ).tolist()
        )

        # the second stage draws the b of each different a, from the transform in b of
        # the product of the state and its flip, which is also the expectation values of
        # the strings of that a, up to the phase of the number of Y. The draws of the
        # same a are made together, so that the transform is computed only once, and the
        # expectation values are kept for the strings that are drawn
        outcomes = backend.arange(dim)
        phases = backend.cast([1, 1j, -1, -1j], dtype=backend.complex128)
        tensor = backend.reshape(state, (2,) * nqubits)
        counts, expectations = Counter(), {}
        for pattern, npattern in sorted(patterns.items()):
            qubits = [q for q in range(nqubits) if pattern >> (nqubits - 1 - q) & 1]
            flipped = backend.reshape(backend.flip(tensor, axis=tuple(qubits)), (dim,))
            values = backend.real(
                phases[
                    sum((outcomes // 2 ** (nqubits - 1 - q)) % 2 for q in qubits) % 4
                ]
                * _fast_walsh_hadamard_transform(
                    backend.conj(flipped) * state, backend=backend
                )
            )
            weights = values**2

            for bits, count in Counter(
                backend.to_numpy(
                    backend.random_choice(
                        dim,
                        size=npattern,
                        p=weights / backend.sum(weights),
                        seed=generator,
                        dtype=backend.int64,
                    )
                ).tolist()
            ).items():
                observable = "".join(
                    "IXZY"[
                        (pattern >> (nqubits - 1 - q) & 1)
                        + 2 * (bits >> (nqubits - 1 - q) & 1)
                    ]
                    for q in range(nqubits)
                )
                counts[observable] = count
                expectations[observable] = float(values[bits])

        observables = sorted(counts)
        circuits = self.circuits(circuit, observables)

        if seed is not None:
            backend.set_seed(seed)

        estimate = 0.0
        for observable, tomography_circuit in zip(observables, circuits):
            expectation = expectations[observable]
            nrepetitions = math.ceil(
                2 * math.log(2 / delta) / (expectation**2 * nsamples * epsilon**2)
            )
            result = self.execute(
                [tomography_circuit], counts[observable] * nrepetitions, backend
            )[0]

            # the outcome of a Pauli string is the parity of the bits of its support
            support = int(
                "".join("0" if pauli == "I" else "1" for pauli in observable), 2
            )
            parities = sum(
                frequency * (-1) ** (outcome & support).bit_count()
                for outcome, frequency in result.frequencies(binary=False).items()
            )
            estimate += parities / (nrepetitions * expectation)

        return estimate / nsamples

    def circuits(self, circuit: Circuit, observables: list[str]) -> list[Circuit]:
        """Appends the basis changes and the measurements of each Pauli string to ``circuit``.

        Args:
            circuit (:class:`qibo.models.circuit.Circuit`): circuit preparing the
                state, without measurements.
            observables (list): Pauli strings to be measured, each one a ``str`` of
                ``"I"``, ``"X"``, ``"Y"`` and ``"Z"``, one per qubit, *e.g.* ``"XIZ"``.

        Returns:
            list: one circuit for each Pauli string, in the same order. Each circuit
            measures all qubits, after rotating the Pauli :math:`X` with
            :class:`qibo.gates.H`, and the Pauli :math:`Y` with :class:`qibo.gates.SDG`
            followed by :class:`qibo.gates.H`. The identity and the Pauli :math:`Z` need
            no rotation. The outcomes of the qubits of the identity are not used.
        """
        return self._pauli_circuits(circuit, observables)
