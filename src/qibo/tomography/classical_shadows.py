"""Classical shadows tomography with random Clifford measurements."""

from gc import collect

from numpy.typing import ArrayLike

from qibo import Circuit, gates
from qibo.backends import Backend, _check_backend
from qibo.config import raise_error
from qibo.hamiltonians.abstract import AbstractHamiltonian
from qibo.models.encodings import entangling_layer
from qibo.quantum_info._superoperator_transformations import (
    _operator_to_pauli_vectors_fht,
)
from qibo.quantum_info.basis import comp_basis_to_pauli
from qibo.quantum_info.random_ensembles import random_clifford
from qibo.quantum_info.utils import hadamard_transform
from qibo.tomography.abstract import Tomography

BASES = ("pauli", "computational")
"""Supported bases for the frame operator."""

METHODS = ("global-clifford", "local-clifford", "ultra-shallow")
"""Supported ensembles: :math:`\\text{Cl}(2^{n})`, :math:`\\text{Cl}(2)^{\\otimes n}`, and
random single-qubit Cliffords interleaved with layers of CNOT gates."""

PAULI_DIGITS = str.maketrans("IXYZ", "0123")
"""Maps a Pauli string to the base-:math:`4` digits of its position in the Pauli basis."""

SINGLE_QUBIT_CLIFFORDS = (
    "",
    "H",
    "S",
    "HS",
    "SH",
    "SS",
    "HSH",
    "HSS",
    "SHS",
    "SSH",
    "SSS",
    "HSHS",
    "HSSH",
    "HSSS",
    "SHSS",
    "SSHS",
    "HSHSS",
    "HSSHS",
    "SHSSH",
    "SHSSS",
    "SSHSS",
    "HSHSSH",
    "HSHSSS",
    "HSSHSS",
)
"""The :math:`24` elements of :math:`\\text{Cl}(2)` up to a global phase, as words in
the gates :class:`qibo.gates.H` and :class:`qibo.gates.S`, in order of application."""


class ClassicalShadow(Tomography):
    """Classical shadows protocol with random Clifford measurements.

    The state is rotated by a random unitary :math:`U` and measured in the
    computational basis, returning :math:`b`. Each snapshot
    :math:`U^{\\dagger} |b\\rangle\\langle b| U` is averaged and mapped through the
    inverse of the frame operator, i.e. the measurement channel of the ensemble
    of :math:`U`, which is one of the following ``method``.

    * ``"global-clifford"``: :math:`U \\in \\text{Cl}(2^{n})`.
    * ``"local-clifford"``: :math:`U \\in \\text{Cl}(2)^{\\otimes n}`.
    * ``"ultra-shallow"``: :math:`U` has ``depth + 1`` layers of random single-qubit
      Cliffords, with a layer of CNOT gates before each layer but the first.
      Its frame operator has no analytical form, and is sampled from the same
      circuits applied to the zero state.

    The frame operator is applied in the Pauli basis, where it is diagonal, or in the
    computational basis. Estimates can be combined by the median of means, and circuits
    are generated and executed in chunks to bound the memory used.

    Example:
        .. testcode::

            from qibo import Circuit, gates
            from qibo.tomography import ClassicalShadow

            circuit = Circuit(2)
            circuit.add(gates.H(0))
            circuit.add(gates.CNOT(0, 1))

            # expectation value of 0.5 ZZ + 0.5 XX in a Bell state
            shadow = ClassicalShadow()
            estimate = shadow(
                circuit, {"ZZ": 0.5, "XX": 0.5}, 100, "local-clifford", seed=42
            )

    References:
        1. H.-Y. Huang, R. Kueng and J. Preskill, *Predicting many properties of a
           quantum system from very few measurements*,
           `Nat. Phys. 16, 1050 (2020) <https://doi.org/10.1038/s41567-020-0932-7>`_.
        2. L. Innocenti *et al.*, *Shadow tomography on general measurement frames*,
           `PRX Quantum 4, 040328 (2023) <https://doi.org/10.1103/PRXQuantum.4.040328>`_.
        3. R. M. S. Farias, R. D. Peddinti, I. Roth and L. Aolita,
           *Robust ultra-shallow shadows*, `Quantum Sci. Technol.
           <https://doi.org/10.1088/2058-9565/adc14f>`_.
    """

    def __call__(
        self,
        circuit: Circuit,
        observable: (
            list[tuple[str, float]] | dict[str, float] | AbstractHamiltonian | ArrayLike
        ),
        nsamples: int,
        method: str = "global-clifford",
        basis: str = "pauli",
        nshots: int = 1,
        nbatches: int = 1,
        chunksize: int = 1000,
        depth: int = 1,
        ncalibration: int | None = None,
        seed: int | None = None,
        backend: Backend = None,
    ):
        """Estimates the expectation value of ``observable`` on the output of ``circuit``.

        Args:
            circuit (:class:`qibo.models.circuit.Circuit`): circuit preparing the
                state, without measurements.
            observable (list, dict, :class:`qibo.hamiltonians.abstract.AbstractHamiltonian`
                or ArrayLike): if ``basis="pauli"``, a list of tuples
                ``(pauli_string, coefficient)`` or a dictionary
                ``{pauli_string: coefficient}``, with ``pauli_string`` a ``str`` of
                ``"I"``, ``"X"``, ``"Y"`` and ``"Z"``, one per qubit, e.g. ``"XIZ"``.
                If ``basis="computational"``, a Hamiltonian or a matrix.
            nsamples (int): number of random unitaries.
            method (str, optional): ``"global-clifford"``, ``"local-clifford"``, or
                ``"ultra-shallow"``. Defaults to ``"global-clifford"``.
            basis (str, optional): ``"pauli"`` or ``"computational"``, the basis in
                which the frame is applied. Defaults to ``"pauli"``.
            nshots (int, optional): number of shots per random unitary.
                Defaults to :math:`1`.
            nbatches (int, optional): number of batches of the median-of-means
                estimator. The ``nsamples`` unitaries are split into ``nbatches``
                batches of (almost) equal size, and the median of the batch
                estimates is returned. If :math:`1`, it is the empirical mean.
                Defaults to :math:`1`.
            chunksize (int, optional): maximum number of circuits generated and
                executed at once, which bounds the memory used by the circuits.
                Defaults to :math:`1000`.
            depth (int, optional): number of entangling layers of the measurement
                circuits of ``"ultra-shallow"``, which has ``depth + 1`` layers of
                single-qubit Cliffords. It is ignored by the other methods.
                Defaults to :math:`1`.
            ncalibration (int, optional): number of circuits used to sample the frame
                operator of ``"ultra-shallow"``, which has no analytical form. It is
                ignored by the other methods. If ``None``, it is ``nsamples``.
                Defaults to ``None``.
            seed (int, optional): seed of the random unitaries. The calibration of
                ``"ultra-shallow"`` uses ``seed + 1``. Defaults to ``None``.
            backend (:class:`qibo.backends.abstract.Backend`, optional): backend
                to be used in the execution. If ``None``, it uses the current
                backend. Defaults to ``None``.

        Returns:
            float: estimated expectation value.
        """
        if basis not in BASES:
            raise_error(
                ValueError, f"``basis`` must be one of {BASES}, but it is {basis}."
            )

        if not isinstance(nbatches, int) or not 1 <= nbatches <= nsamples:
            raise_error(
                ValueError,
                f"``nbatches`` must be an int between 1 and ``nsamples`` = {nsamples}, "
                + f"but it is {nbatches}.",
            )

        if not isinstance(chunksize, int) or chunksize < 1:
            raise_error(
                ValueError,
                f"``chunksize`` must be a positive int, but it is {chunksize}.",
            )

        backend = _check_backend(backend)

        if basis == "pauli":
            estimates = self._expectation_pauli(
                circuit,
                observable,
                nsamples,
                method,
                nshots,
                nbatches,
                chunksize,
                depth,
                nsamples if ncalibration is None else ncalibration,
                seed,
                backend,
            )
        else:
            estimates = self._expectation_computational(
                circuit,
                observable,
                nsamples,
                method,
                nshots,
                nbatches,
                chunksize,
                depth,
                nsamples if ncalibration is None else ncalibration,
                seed,
                backend,
            )

        estimates = backend.sort(estimates)
        half = nbatches // 2

        if nbatches % 2 == 1:
            return estimates[half]

        return (estimates[half - 1] + estimates[half]) / 2

    def circuits(
        self,
        circuit: Circuit,
        nsamples: int,
        method: str = "global-clifford",
        depth: int = 1,
        seed: int | None = None,
        backend: Backend = None,
    ) -> list[Circuit]:
        """Appends random Clifford unitaries and measurements to ``circuit``.

        Args:
            circuit (:class:`qibo.models.circuit.Circuit`): circuit preparing the
                state, without measurements.
            nsamples (int): number of random unitaries, i.e. of circuits.
            method (str, optional): ``"global-clifford"`` samples from
                :math:`\\text{Cl}(2^{n})`; ``"local-clifford"`` samples from
                :math:`\\text{Cl}(2)^{\\otimes n}`; ``"ultra-shallow"`` alternates
                ``depth + 1`` layers of random single-qubit Cliffords with ``depth``
                layers of CNOT gates, built with the ``"shifted"``
                architecture of :func:`qibo.models.encodings.entangling_layer`, i.e.
                CNOTs on even pairs followed by odd pairs, so that ``depth=0`` is
                ``"local-clifford"``. Defaults to ``"global-clifford"``.
            depth (int, optional): number of entangling layers of ``"ultra-shallow"``.
                It is ignored by the other methods. Defaults to :math:`1`.
            seed (int, optional): seed of the random unitaries. Defaults to ``None``.
            backend (:class:`qibo.backends.abstract.Backend`, optional): backend
                to be used in the sampling. If ``None``, it uses the current
                backend. Defaults to ``None``.

        Returns:
            list: ``nsamples`` circuits, each followed by a measurement of all qubits.
        """
        if method not in METHODS:
            raise_error(
                ValueError, f"``method`` must be one of {METHODS}, but it is {method}."
            )

        if not isinstance(depth, int) or depth < 0:
            raise_error(
                ValueError,
                f"``depth`` must be a non-negative int, but it is {depth}.",
            )

        self._check_circuit(circuit)
        backend = _check_backend(backend)

        nqubits = circuit.nqubits
        nlayers = depth + 1 if method == "ultra-shallow" else 1

        if method == "global-clifford":
            # ``random_clifford`` reseeds the backend, so one seed per Clifford is drawn
            seeds = backend.random_integers(2**31 - 1, size=nsamples, seed=seed)
        else:
            indices = backend.random_integers(
                len(SINGLE_QUBIT_CLIFFORDS),
                size=(nsamples, nlayers, nqubits),
                seed=seed,
            )
            entangler = entangling_layer(nqubits, "shifted")

        circuits = []
        for sample in range(nsamples):
            shadow_circuit = circuit.copy()
            if method == "global-clifford":
                clifford = random_clifford(
                    nqubits, seed=int(seeds[sample]), backend=backend
                )
                shadow_circuit.add(clifford.on_qubits(*range(nqubits)))
            else:
                for layer in range(nlayers):
                    if layer > 0:
                        shadow_circuit.add(entangler.on_qubits(*range(nqubits)))
                    for qubit, index in enumerate(indices[sample, layer]):
                        shadow_circuit.add(
                            [
                                getattr(gates, name)(qubit)
                                for name in SINGLE_QUBIT_CLIFFORDS[int(index)]
                            ]
                        )
            shadow_circuit.add(gates.M(*range(nqubits)))
            circuits.append(shadow_circuit)

        return circuits

    def frame_operator(
        self,
        nqubits: int,
        method: str = "global-clifford",
        inverse: bool = False,
        basis: str = "pauli",
        depth: int = 1,
        nsamples: int = 1000,
        chunksize: int = 1000,
        seed: int | None = None,
        backend: Backend = None,
    ) -> ArrayLike:
        """Frame operator, i.e. the measurement channel, of the Clifford ensemble.

        In the Pauli basis, the frame is diagonal with eigenvalue :math:`1` for the
        identity. Any other Pauli string of weight :math:`w` has eigenvalue
        :math:`1 / (2^{n} + 1)` if ``method="global-clifford"``, and
        :math:`3^{-w}` if ``method="local-clifford"``. If ``method="ultra-shallow"``,
        the eigenvalue :math:`f_{k}` depends only on the support
        :math:`k \\in \\{0, 1\\}^{n}` of the Pauli string, and it is sampled: it is the
        average of :math:`\\langle z | g Z^{k} g^{\\dagger} | z \\rangle` over
        ``nsamples`` random circuits :math:`g` applied to the zero state, with
        outcomes :math:`z`, where :math:`Z^{k} = \\bigotimes_{l} Z^{k_{l}}`.

        .. note::
            If ``basis="pauli"``, only the diagonal of the frame is returned,
            as a real-valued vector of size :math:`4^{n}`, with Pauli strings
            ordered as in :func:`qibo.quantum_info.pauli_basis`.

        Args:
            nqubits (int): number of qubits.
            method (str, optional): ``"global-clifford"``, ``"local-clifford"``, or
                ``"ultra-shallow"``. Defaults to ``"global-clifford"``.
            inverse (bool, optional): if ``True``, returns the inverse frame. For
                ``"ultra-shallow"``, it raises an error if a sampled eigenvalue is not
                positive. Defaults to ``False``.
            basis (str, optional): ``"pauli"`` or ``"computational"``. If
                ``"computational"``, the full :math:`4^{n} \\times 4^{n}` matrix acting
                on the row-vectorized density matrix is returned.
                Defaults to ``"pauli"``.
            depth (int, optional): number of entangling layers of ``"ultra-shallow"``.
                It is ignored by the other methods. Defaults to :math:`1`.
            nsamples (int, optional): number of circuits used to sample the frame of
                ``"ultra-shallow"``. It is ignored by the other methods. Defaults to
                :math:`1000`.
            chunksize (int, optional): maximum number of circuits executed at once
                when sampling. Defaults to :math:`1000`.
            seed (int, optional): seed of the sampling. Defaults to ``None``.
            backend (:class:`qibo.backends.abstract.Backend`, optional): backend
                to be used in the calculation. If ``None``, it uses the current
                backend. Defaults to ``None``.

        Returns:
            ArrayLike: frame operator, or its inverse.
        """
        if method not in METHODS:
            raise_error(
                ValueError, f"``method`` must be one of {METHODS}, but it is {method}."
            )

        if basis not in BASES:
            raise_error(
                ValueError, f"``basis`` must be one of {BASES}, but it is {basis}."
            )

        backend = _check_backend(backend)

        if method == "global-clifford":
            eigenvalues = [1.0]
            eigenvalues.extend([1 / (2**nqubits + 1)] * (4**nqubits - 1))
            eigenvalues = backend.cast(eigenvalues, dtype=backend.float64)
        elif method == "local-clifford":
            single_qubit = backend.cast([1, 1 / 3, 1 / 3, 1 / 3], dtype=backend.float64)
            eigenvalues = single_qubit
            for _ in range(nqubits - 1):
                eigenvalues = backend.kron(eigenvalues, single_qubit)
        else:
            # the zero state is the only one needed, since the frame is Pauli diagonal
            snapshot = next(
                self._snapshots(
                    Circuit(nqubits),
                    nsamples,
                    method,
                    1,
                    1,
                    chunksize,
                    depth,
                    seed,
                    backend,
                )
            )
            # eigenvalue of each support: Walsh-Hadamard transform of the probabilities,
            # whose scale is fixed by the eigenvalue of the identity, which is one
            eigenvalues = hadamard_transform(
                backend.real(backend.diag(snapshot)), backend=backend
            )
            eigenvalues = eigenvalues / eigenvalues[0]

            # lifts the eigenvalues of the supports to those of all Pauli strings
            ones = backend.cast([1, 1, 1, 1], dtype=backend.int64)
            non_identity = backend.cast([0, 1, 1, 1], dtype=backend.int64)
            supports = backend.cast([0], dtype=backend.int64)
            for _ in range(nqubits):
                supports = backend.kron(2 * supports, ones) + backend.kron(
                    backend.ones(len(supports), dtype=backend.int64), non_identity
                )
            eigenvalues = eigenvalues[supports]

        if inverse:
            if backend.any(eigenvalues <= 0):
                raise_error(
                    ValueError,
                    "The sampled frame is not invertible, increase ``nsamples``.",
                )
            eigenvalues = 1 / eigenvalues

        if basis == "pauli":
            return eigenvalues

        change_of_basis = comp_basis_to_pauli(nqubits, normalize=True, backend=backend)
        eigenvalues = backend.cast(eigenvalues, dtype=backend.complex128)

        return (
            backend.conj(change_of_basis).T
            @ backend.diag(eigenvalues)
            @ change_of_basis
        )

    def _expectation_computational(
        self,
        circuit: Circuit,
        observable: AbstractHamiltonian | ArrayLike,
        nsamples: int,
        method: str,
        nshots: int,
        nbatches: int,
        chunksize: int,
        depth: int,
        ncalibration: int,
        seed: int | None,
        backend: Backend = None,
    ):
        """Expectation value with the inverse frame applied in the computational basis.

        Args:
            circuit (:class:`qibo.models.circuit.Circuit`): circuit preparing the
                state, without measurements.
            observable (:class:`qibo.hamiltonians.abstract.AbstractHamiltonian` or
                ArrayLike): observable, either as a Hamiltonian or as a matrix.
            nsamples (int): number of random unitaries.
            method (str): one of ``METHODS``.
            nshots (int): number of shots per random unitary.
            nbatches (int): number of batches.
            chunksize (int): maximum number of circuits executed at once.
            depth (int): number of entangling layers of ``"ultra-shallow"``.
            ncalibration (int): number of circuits used to sample the frame.
            seed (int or None): seed of the random unitaries.
            backend (:class:`qibo.backends.abstract.Backend`, optional): backend
                to be used in the execution. If ``None``, it uses the current
                backend. Defaults to ``None``.

        Returns:
            ArrayLike: estimated expectation value of each batch.
        """
        backend = _check_backend(backend)

        nqubits = circuit.nqubits
        dim = 2**nqubits

        frame = self.frame_operator(
            nqubits,
            method,
            True,
            "computational",
            depth,
            ncalibration,
            chunksize,
            # independent of the seed of the shadows, to avoid correlations
            None if seed is None else seed + 1,
            backend,
        )

        if not isinstance(observable, AbstractHamiltonian):
            observable = backend.cast(observable)

        estimates = []
        for snapshot in self._snapshots(
            circuit,
            nsamples,
            method,
            nshots,
            nbatches,
            chunksize,
            depth,
            seed,
            backend,
        ):
            state = backend.reshape(
                frame @ backend.reshape(snapshot, (-1,)), (dim, dim)
            )
            if isinstance(observable, AbstractHamiltonian):
                estimates.append(observable.expectation_from_state(state))
            else:
                estimates.append(backend.expectation_value(observable, state, False))

        return backend.cast(estimates, dtype=backend.float64)

    def _expectation_pauli(
        self,
        circuit: Circuit,
        observable: list[tuple[str, float]] | dict[str, float],
        nsamples: int,
        method: str,
        nshots: int,
        nbatches: int,
        chunksize: int,
        depth: int,
        ncalibration: int,
        seed: int | None,
        backend: Backend = None,
    ):
        """Expectation value with the inverse frame applied in the Pauli basis.

        Since the frame is diagonal in this basis, the estimate is the sum over the
        Pauli strings :math:`P` of the observable of the product of its coefficient,
        the inverse-frame eigenvalue of :math:`P`, and the coefficient of the
        averaged snapshot along :math:`P`. The latter are obtained for all Pauli
        strings with the fast Hadamard transform.

        Args:
            circuit (:class:`qibo.models.circuit.Circuit`): circuit preparing the
                state, without measurements.
            observable (list or dict): Pauli strings and their coefficients.
            nsamples (int): number of random unitaries.
            method (str): one of ``METHODS``.
            nshots (int): number of shots per random unitary.
            nbatches (int): number of batches.
            chunksize (int): maximum number of circuits executed at once.
            depth (int): number of entangling layers of ``"ultra-shallow"``.
            ncalibration (int): number of circuits used to sample the frame.
            seed (int or None): seed of the random unitaries.
            backend (:class:`qibo.backends.abstract.Backend`, optional): backend
                to be used in the execution. If ``None``, it uses the current
                backend. Defaults to ``None``.

        Returns:
            ArrayLike: estimated expectation value of each batch.
        """
        backend = _check_backend(backend)

        nqubits = circuit.nqubits

        terms = (
            list(observable.items())
            if isinstance(observable, dict)
            else list(observable)
        )
        for string, _ in terms:
            if len(string) != nqubits or set(string) - set("IXYZ"):
                raise_error(
                    ValueError,
                    f"Pauli string {string} is not a string of {nqubits} characters "
                    + "among ``I``, ``X``, ``Y`` and ``Z``.",
                )

        inverse_frame = self.frame_operator(
            nqubits,
            method,
            True,
            "pauli",
            depth,
            ncalibration,
            chunksize,
            # independent of the seed of the shadows, to avoid correlations
            None if seed is None else seed + 1,
            backend,
        )

        indices = backend.cast(
            [int(string.translate(PAULI_DIGITS), 4) for string, _ in terms],
            dtype=backend.int64,
        )
        coefficients = backend.cast(
            [coefficient for _, coefficient in terms], dtype=backend.float64
        )
        # observable coefficients times inverse-frame eigenvalues of its strings
        weights = coefficients * inverse_frame[indices]

        estimates = []
        for snapshot in self._snapshots(
            circuit,
            nsamples,
            method,
            nshots,
            nbatches,
            chunksize,
            depth,
            seed,
            backend,
        ):
            # real Pauli coefficients tr(P snapshot), for all Pauli strings P
            state = backend.real(
                _operator_to_pauli_vectors_fht(
                    snapshot, nqubits, 2**nqubits, backend=backend
                )
            )
            estimates.append(backend.sum(weights * state[indices]))

        return backend.cast(estimates, dtype=backend.float64)

    def _snapshots(
        self,
        circuit: Circuit,
        nsamples: int,
        method: str,
        nshots: int,
        nbatches: int,
        chunksize: int,
        depth: int,
        seed: int | None,
        backend: Backend = None,
    ):
        """Yields the averaged snapshot of each batch, one batch at a time.

        Circuits are generated and executed in chunks of at most ``chunksize``
        that never cross a batch, so only one chunk and one batch average are
        kept in memory. One seed per chunk is drawn from ``seed``, and it also seeds
        the sampling of the outcomes.

        Args:
            circuit (:class:`qibo.models.circuit.Circuit`): circuit preparing the
                state, without measurements.
            nsamples (int): number of random unitaries.
            method (str): one of ``METHODS``.
            nshots (int): number of shots per random unitary.
            nbatches (int): number of batches. The first ``nsamples % nbatches``
                batches have one more unitary than the others.
            chunksize (int): maximum number of circuits executed at once.
            depth (int): number of entangling layers of ``"ultra-shallow"``.
            seed (int or None): seed of the random unitaries.
            backend (:class:`qibo.backends.abstract.Backend`, optional): backend
                to be used in the execution. If ``None``, it uses the current
                backend. Defaults to ``None``.

        Yields:
            ArrayLike: :math:`2^{n} \\times 2^{n}` average of
            :math:`U^{\\dagger} |b\\rangle\\langle b| U` over the random unitaries
            :math:`U` and outcomes :math:`b` of a batch.
        """
        backend = _check_backend(backend)

        nqubits = circuit.nqubits
        dim = 2**nqubits

        sizes = [
            nsamples // nbatches + (batch < nsamples % nbatches)
            for batch in range(nbatches)
        ]
        nchunks = sum(-(-size // chunksize) for size in sizes)
        seeds = backend.random_integers(2**31 - 1, size=nchunks, seed=seed)

        chunk = 0
        for size in sizes:
            snapshot = backend.zeros((dim, dim), dtype=backend.complex128)
            for start in range(0, size, chunksize):
                # seeds the sampling of the outcomes, which uses the backend generator
                backend.set_seed(int(seeds[chunk]))
                circuits = self.circuits(
                    circuit,
                    min(chunksize, size - start),
                    method,
                    depth,
                    int(seeds[chunk]),
                    backend,
                )
                results = self.execute(circuits, nshots, backend)
                chunk += 1

                for shadow_circuit, result in zip(circuits, results):
                    # the random unitary is what was appended after ``circuit``,
                    # without the final measurement
                    unitary_circuit = Circuit(nqubits)
                    unitary_circuit.add(shadow_circuit.queue[len(circuit.queue) : -1])
                    # rows of the unitary indexed by the outcomes are <b|U
                    rows = unitary_circuit.unitary(backend)[
                        result.samples(binary=False)
                    ]
                    snapshot = snapshot + backend.conj(rows).T @ rows

                # circuits hold reference cycles, so they are freed explicitly
                # instead of waiting for the next automatic collection
                del circuits, results
                collect()

            yield snapshot / (size * nshots)
