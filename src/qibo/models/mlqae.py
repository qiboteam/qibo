import math
from collections.abc import Callable
from numbers import Integral

from qibo.backends import Backend, _check_backend
from qibo.config import raise_error
from qibo.models.circuit import Circuit
from qibo.models.qaa import QAA


def MLQAE(
    algorithm: Circuit,
    oracle: Circuit,
    check: Callable[[str], bool],
    nlevels: int = 5,
    nshots: int = 100,
    backend: Backend = None,
) -> float:
    """Estimates an amplitude with the Maximum-likelihood Quantum Amplitude Estimation.

    Let :math:`a` be the probability of measuring a good state in
    :math:`\\mathcal{A}|0\\rangle`, where :math:`\\mathcal{A}` is the circuit ``algorithm``,
    :math:`|0\\rangle` is the all-zero state, and the good states are marked by the oracle. Let
    :math:`\\theta_{a} \\in [0, \\, \\pi / 2]` be such that :math:`a = \\sin^{2}(\\theta_{a})`.
    After :math:`m` applications of the operator :math:`\\mathbf{Q}` of
    :func:`qibo.models.qaa.amplification_operator`, the probability of measuring a good state is
    :math:`\\sin^{2}((2m + 1)\\theta_{a})`. Differently from :func:`qibo.models.qae.QAE`, there
    are no counting qubits nor controlled operations. Instead, the circuit
    :math:`\\mathbf{Q}^{m}\\mathcal{A}` is measured ``nshots`` times for each :math:`m` in
    :math:`\\{0, \\, 1, \\, 2, \\, 4, \\, \\dots, \\, 2^{L - 1}\\}`, where :math:`L` is ``nlevels``.
    If :math:`h_{m}` of the outcomes are good for a given :math:`m`, the estimate is
    :math:`\\sin^{2}(\\hat{\\theta})`, where :math:`\\hat{\\theta}` maximizes the likelihood

    .. math::
        \\prod_{m} \\sin^{2}\\left((2m + 1)\\theta\\right)^{h_{m}} \\,
        \\cos^{2}\\left((2m + 1)\\theta\\right)^{N - h_{m}} \\, ,

    with :math:`N` being ``nshots``. The maximum is found with a grid search on
    :math:`[0, \\, \\pi / 2]`, which is refined around the best point.
    The number of applications of :math:`\\mathcal{A}` and :math:`\\mathcal{A}^{-1}` is
    :math:`N \\sum_{m} (2m + 1)`. With this exponential schedule of :math:`m`,
    the error is reported to approach the Heisenberg limit, which is inversely
    proportional to that number (Ref. [1]).

    Example:
        .. testcode::

            import math

            from qibo import Circuit, gates
            from qibo.models import MLQAE

            # state cos(theta) |0> + sin(theta) |1> with theta = 0.5, and good state |1>
            algorithm = Circuit(1)
            algorithm.add(gates.RY(0, theta=2 * 0.5))
            oracle = Circuit(1)
            oracle.add(gates.Z(0))

            estimate = MLQAE(algorithm, oracle, check=lambda bitstring: bitstring == "1")

            # the exact amplitude is sin(0.5)^2, which is approximately 0.23
            print(round(estimate, 1))

        .. testoutput::

            0.2

    Args:
        algorithm (:class:`qibo.models.circuit.Circuit`): circuit :math:`\\mathcal{A}` without
            measurements.
        oracle (:class:`qibo.models.circuit.Circuit`): circuit :math:`\\mathbf{S}_{\\chi}` that
            multiplies the good states by :math:`-1`, without measurements and with the same
            number of qubits as ``algorithm``. See
            :func:`qibo.models.qaa.amplification_operator`.
        check (Callable): function that receives a measured bitstring, *e.g.* ``"101"``, and
            returns ``True`` if the state is good.
        nlevels (int, optional): number :math:`L` of levels of the schedule of the numbers of
            iterations, which are :math:`0` and :math:`2^{k}`, for :math:`k = 0, \\dots, L - 1`.
            Defaults to :math:`5`.
        nshots (int, optional): number of shots of each circuit. Defaults to :math:`100`.
        backend (:class:`qibo.backends.abstract.Backend`, optional): backend to be used in the
            execution of the circuits and in the maximization. If ``None``, it uses the current
            backend. Defaults to ``None``.

    Returns:
        float: estimate of :math:`a`, which belongs to the interval :math:`[0, \\, 1]`.

    References:
        1. Y. Suzuki, S. Uno, R. Raymond, T. Tanaka, T. Onodera, and N. Yamamoto,
        *Amplitude estimation without phase estimation*,
        `Quantum Inf. Process. 19, 75 (2020) <https://doi.org/10.1007/s11128-019-2565-2>`_.

        2. G. Brassard, P. Høyer, M. Mosca, and A. Tapp, *Quantum Amplitude Amplification and
        Estimation*, `Contemporary Mathematics 305, 53 (2002)
        <https://doi.org/10.1090/conm/305/05215>`_.
    """
    backend = _check_backend(backend)

    if not callable(check):
        raise_error(TypeError, f"check must be callable, but it is {type(check)}.")

    for name, value in (("nlevels", nlevels), ("nshots", nshots)):
        if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
            raise_error(
                ValueError, f"{name} must be a positive integer, but it is {value}."
            )

    # numbers of iterations, and number of good outcomes of each circuit
    iterations = [0]
    iterations.extend(2**level for level in range(nlevels))

    hits = []
    for niterations in iterations:
        circuit = QAA(algorithm, oracle, niterations=niterations, backend=backend)
        frequencies = backend.execute_circuit(circuit, nshots=nshots).frequencies()
        hits.append(
            sum(count for bitstring, count in frequencies.items() if check(bitstring))
        )

    hits = backend.cast(hits, dtype=backend.float64)
    multipliers = backend.cast(
        [2 * niterations + 1 for niterations in iterations], dtype=backend.float64
    )

    # the grid has several points per period of the fastest oscillation of the likelihood
    ngrid = max(1000, 20 * (2 * iterations[-1] + 1))
    points = backend.arange(ngrid)
    low, high = 0.0, math.pi / 2
    for _ in range(3):
        angles = low + (high - low) * points / (ngrid - 1)
        probabilities = backend.sin(backend.outer(angles, multipliers)) ** 2

        # the small constant avoids the logarithm of zero
        loglikelihood = backend.sum(
            hits * backend.log(probabilities + 1e-300)
            + (nshots - hits) * backend.log(1 - probabilities + 1e-300),
            axis=1,
        )

        best = int(backend.argmax(loglikelihood))
        step = (high - low) / (ngrid - 1)
        low = max(float(angles[best]) - step, 0.0)
        high = min(float(angles[best]) + step, math.pi / 2)

    return float(backend.sin(angles[best]) ** 2)
