"""Vanilla quantum state tomography with Pauli measurements and linear inversion."""

from itertools import product

from numpy.typing import ArrayLike

from qibo import Circuit, gates
from qibo.backends import Backend, _check_backend
from qibo.config import raise_error
from qibo.quantum_info._superoperator_transformations import (
    _pauli_vectors_to_operator_fht,
)
from qibo.quantum_info.utils import hadamard_transform
from qibo.tomography.abstract import Tomography


class StateTomography(Tomography):
    """Vanilla quantum state tomography by linear inversion of Pauli measurements.

    The state :math:`\\rho` of :math:`n` qubits is expanded in the Pauli basis,

    .. math::
        \\rho = \\frac{1}{2^{n}} \\sum_{P} \\text{tr}(P \\, \\rho) \\, P \\, ,

    where the sum runs over the :math:`4^{n}` Pauli strings :math:`P`. Each qubit is measured
    in one of the bases :math:`X`, :math:`Y`, or :math:`Z`, which gives :math:`3^{n}`
    measurement settings. The expectation value of a Pauli string with :math:`w` non-identity
    factors is estimated from the outcomes of any setting that measures those factors in the
    right bases, so there are :math:`3^{n - w}` such settings, whose estimates are averaged.
    The estimate of the state, :math:`\\hat{\\rho}`, is obtained by inserting the estimated
    expectation values in the expansion above [1, 2].

    The estimate is Hermitian and has unit trace, and it is unbiased, but it is **not**
    positive semidefinite in general, since the statistical errors can make some of its
    eigenvalues negative. The number of circuits, :math:`3^{n}`, and the cost of the
    reconstruction grow exponentially with :math:`n`, so this protocol is meant for few qubits.

    Example:
        .. testcode::

            from qibo import Circuit, gates
            from qibo.tomography import StateTomography

            circuit = Circuit(2)
            circuit.add(gates.H(0))
            circuit.add(gates.CNOT(0, 1))

            # reconstruction of the Bell state from 9 circuits of 1000 shots
            estimate = StateTomography()(circuit, nshots=1000, seed=1)

    References:
        1. M. A. Nielsen and I. L. Chuang, *Quantum Computation and Quantum Information*,
           Cambridge University Press (2010), Sec. 8.4.2.
        2. D. F. V. James, P. G. Kwiat, W. J. Munro and A. G. White,
           *Measurement of qubits*,
           `Phys. Rev. A 64, 052312 (2001) <https://doi.org/10.1103/PhysRevA.64.052312>`_.
    """

    def __call__(
        self,
        circuit: Circuit,
        nshots: int = 1000,
        seed: int | None = None,
        backend: Backend = None,
    ) -> ArrayLike:
        """Estimates the density matrix of the state prepared by ``circuit``.

        Args:
            circuit (:class:`qibo.models.circuit.Circuit`): circuit preparing the
                state, without measurements.
            nshots (int, optional): number of shots per measurement setting.
                Defaults to :math:`1000`.
            seed (int, optional): seed of the sampling of the outcomes. If ``None``, the
                generator of the backend is not reseeded. Defaults to ``None``.
            backend (:class:`qibo.backends.abstract.Backend`, optional): backend
                to be used in the execution. If ``None``, it uses the current
                backend. Defaults to ``None``.

        Returns:
            ArrayLike: :math:`2^{n} \\times 2^{n}` estimated density matrix.
        """
        if isinstance(nshots, bool) or not isinstance(nshots, int) or nshots < 1:
            raise_error(
                ValueError,
                f"``nshots`` must be a positive int, but it is {nshots}.",
            )

        backend = _check_backend(backend)

        circuits = self.circuits(circuit)

        if seed is not None:
            backend.set_seed(seed)

        results = self.execute(circuits, nshots, backend)

        nqubits = circuit.nqubits
        dim = 2**nqubits

        # supports of the Pauli strings, as the bits of the Walsh-Hadamard transform,
        # and the weights of the qubits in the base-4 position of a Pauli string
        supports = list(product((0, 1), repeat=nqubits))
        weights = [4 ** (nqubits - 1 - qubit) for qubit in range(nqubits)]

        coefficients = backend.zeros(4**nqubits, dtype=backend.float64)
        counts = backend.zeros(4**nqubits, dtype=backend.float64)
        for setting, result in zip(product("XYZ", repeat=nqubits), results):
            probabilities = [0.0] * dim
            for outcome, frequency in result.frequencies(binary=False).items():
                probabilities[outcome] = frequency / nshots

            # the Walsh-Hadamard transform of the probabilities, which the Hadamard
            # transform divides by 2^n, has for each support the expectation value of
            # the Pauli string with the measured Pauli on the qubits of the support
            # and the identity on the others
            expectations = dim * backend.real(
                hadamard_transform(
                    backend.cast(probabilities, dtype=backend.float64), backend=backend
                )
            )

            # base-4 digits of the Pauli strings, with I, X, Y, Z as 0, 1, 2, 3
            digits = ["XYZ".index(pauli) + 1 for pauli in setting]
            indices = backend.cast(
                [
                    sum(
                        digit * bit * weight
                        for digit, bit, weight in zip(digits, bits, weights)
                    )
                    for bits in supports
                ],
                dtype=backend.int64,
            )

            coefficients[indices] += expectations
            counts[indices] += 1.0

        # each Pauli string is averaged over all the settings that measure it
        state = _pauli_vectors_to_operator_fht(
            coefficients / counts, nqubits, dim, backend=backend
        )

        return state / dim

    def circuits(self, circuit: Circuit) -> list[Circuit]:
        """Appends the basis changes and the measurements of each setting to ``circuit``.

        Args:
            circuit (:class:`qibo.models.circuit.Circuit`): circuit preparing the
                state, without measurements.

        Returns:
            list: :math:`3^{n}` circuits, one for each setting, in the order of
            ``itertools.product("XYZ", repeat=n)``, i.e. the Pauli of the first qubit
            changes the slowest. Each circuit measures all qubits, after rotating the
            Pauli :math:`X` with :class:`qibo.gates.H`, and the Pauli :math:`Y` with
            :class:`qibo.gates.SDG` followed by :class:`qibo.gates.H`.
        """
        self._check_circuit(circuit)

        nqubits = circuit.nqubits

        circuits = []
        for setting in product("XYZ", repeat=nqubits):
            tomography_circuit = circuit.copy()
            for qubit, pauli in enumerate(setting):
                if pauli == "Y":
                    tomography_circuit.add(gates.SDG(qubit))
                if pauli != "Z":
                    tomography_circuit.add(gates.H(qubit))
            tomography_circuit.add(gates.M(*range(nqubits)))
            circuits.append(tomography_circuit)

        return circuits
