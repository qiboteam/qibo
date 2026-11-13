import math
from collections.abc import Sequence
from numbers import Integral, Real

from numpy.typing import ArrayLike

from qibo import gates
from qibo.backends import Backend, _check_backend
from qibo.config import raise_error
from qibo.models.algorithms.qae import QAE
from qibo.models.circuit import Circuit


class QMC:
    """Quantum algorithms of Ref. [1] to estimate the mean of the output of an algorithm.

    Let :math:`\\mathcal{A}` be a circuit that acts on the all-zero state of :math:`n` qubits,
    and let :math:`k \\leq n` of its qubits be measured, with outcome :math:`x` that has
    probability :math:`p(x)`. The algorithm outputs the real value :math:`v(x)`, which is
    one of the ``values``. The estimated mean is :math:`\\mathbb{E}[v] = \\sum_{x} p(x) v(x)`.
    All of the algorithms use the amplitude :math:`a = \\sum_{x} p(x) f(x)` of the good state of
    a circuit that prepares :math:`\\mathcal{A} \\ket{0}`, and a flag qubit that is rotated by
    :math:`\\ket{x} \\ket{0} \\mapsto \\ket{x} (\\sqrt{1 - f(x)} \\ket{0} +
    \\sqrt{f(x)} \\ket{1})` if :math:`x` is measured, with :math:`0 \\leq f(x) \\leq 1`, and it is
    estimated with :func:`qibo.models.QAE`. The median of several estimates is used to improve
    the probability of success, with the powering lemma of Ref. [1]. Depending on the arguments
    of :meth:`execute`, the algorithms are:

    - **bounded** (Algorithm 1): if :math:`0 \\leq v(x) \\leq 1`, with :math:`f = v` and a
      number :math:`t` of uses of :math:`\\mathbf{Q}`, the error is
      :math:`O(\\sqrt{\\mathbb{E}[v] / t} + 1 / t)`. It is used by the others.
    - **non-negative** (Algorithm 2): if :math:`v(x) \\geq 0`, :math:`\\mathbb{E}[v]` is split
      into the means of :math:`v` where it is smaller than :math:`1`, and in each interval
      :math:`[2^{\\ell - 1}, 2^{\\ell})` for :math:`\\ell = 1, \\dots,
      \\lceil \\log_{2}(1 / \\epsilon) \\rceil`, which are rescaled to be bounded. The error is
      :math:`\\epsilon (\\sqrt{\\mathbb{E}[v^{2}]} + 1)^{2}` with probability :math:`4 / 5`, and
      it uses :math:`\\mathbf{Q}` :math:`\\tilde{O}(1 / \\epsilon)` times.
    - **bounded variance** (Algorithm 3): if the variance of :math:`v` is at most
      :math:`\\sigma^{2}`, the values are divided by :math:`\\sigma`, a sample :math:`\\tilde{m}`
      is subtracted, and Algorithm 2 is applied to its negative and positive parts. The error is
      :math:`\\epsilon` with probability :math:`2 / 3`, and it uses :math:`\\mathbf{Q}`
      :math:`\\tilde{O}(\\sigma / \\epsilon)` times, while :math:`\\Omega(\\sigma^{2} /
      \\epsilon^{2})` samples are needed classically.
    - **bounded relative variance** (Algorithm 4): if :math:`v(x) \\geq 0` and
      :math:`\\mathrm{Var}(v) / \\mathbb{E}[v]^{2} \\leq B`, the mean :math:`\\tilde{m}` of
      :math:`\\lceil 32 B \\rceil` samples is used to apply Algorithm 2 to :math:`v /
      \\tilde{m}`. The relative error is :math:`\\epsilon` with probability :math:`3 / 4`. It
      assumes :math:`\\mathbb{E}[v] > 0`, and the estimate is zero if all of the samples are zero.

    The circuit of the last amplitude estimation is stored in the attribute ``circuit``, which
    is ``None`` before :meth:`execute`. The rotations are controlled by each of the
    :math:`2^{k}` outcomes, so the circuits are only efficient for a small :math:`k`.

    Example:
        .. testcode::

            import math

            from qibo import Circuit, gates
            from qibo.models import QMC

            # the outcome 1 has probability sin(theta)^2, with theta = 0.3, and it has value 3
            theta = 0.3
            algorithm = Circuit(1)
            algorithm.add(gates.RY(0, theta=2 * theta))

            qmc = QMC(algorithm, values=[0, 3])
            estimate = qmc.execute(epsilon=0.05, delta=0.01)

            print(abs(estimate - 3 * math.sin(theta) ** 2) < 0.15)

        .. testoutput::

            True

    Args:
        algorithm (:class:`qibo.models.circuit.Circuit`): circuit :math:`\\mathcal{A}` without
            measurements.
        values (ArrayLike): :math:`1`-dimensional array with the :math:`2^{k}` real values
            :math:`v(x)` output for each outcome :math:`x` of the measured qubits.
        qubits (Sequence[int], optional): measured qubits of ``algorithm``, where the first one
            is the most significant bit of :math:`x`. If ``None``, they are all of the qubits of
            ``algorithm``. Defaults to ``None``.

    References:
        1. A. Montanaro, *Quantum speedup of Monte Carlo methods*,
        `Proc. R. Soc. A 471, 20150301 (2015) <https://doi.org/10.1098/rspa.2015.0301>`_.
    """

    def __init__(
        self,
        algorithm: Circuit,
        values: ArrayLike,
        qubits: Sequence[int] | None = None,
    ):
        if not isinstance(algorithm, Circuit):
            raise_error(
                TypeError,
                f"algorithm must be a Circuit, but it is type {type(algorithm)}.",
            )

        nqubits = algorithm.nqubits
        qubits = list(range(nqubits)) if qubits is None else list(qubits)
        if (
            len(qubits) == 0
            or len(set(qubits)) != len(qubits)
            or any(
                isinstance(qubit, bool)
                or not isinstance(qubit, Integral)
                or not 0 <= qubit < nqubits
                for qubit in qubits
            )
        ):
            raise_error(
                ValueError,
                f"qubits must be distinct integers in [0, {nqubits}), but they are {qubits}.",
            )

        values = [float(value) for value in values]
        if len(values) != 2 ** len(qubits) or any(
            not math.isfinite(value) for value in values
        ):
            raise_error(
                ValueError,
                f"values must be {2 ** len(qubits)} finite numbers, one for each outcome.",
            )

        self.algorithm = algorithm
        self.values = values
        self.qubits = [int(qubit) for qubit in qubits]
        self.circuit = None

    def execute(
        self,
        epsilon: float,
        sigma: float | None = None,
        relative_variance: float | None = None,
        delta: float | None = None,
        constant: float = 1.0,
        backend: Backend | None = None,
    ) -> float:
        """Estimates the mean of the output of the algorithm.

        If ``sigma`` is given, it uses the algorithm for a bounded variance. If
        ``relative_variance`` is given, it uses the algorithm for a bounded relative variance,
        with non-negative values. If none is given, it uses the algorithm for non-negative values.

        Args:
            epsilon (float): accuracy :math:`\\epsilon`, which is the absolute error if
                ``sigma`` is given, the relative error if ``relative_variance`` is given, and the
                error :math:`\\epsilon (\\sqrt{\\mathbb{E}[v^{2}]} + 1)^{2}` otherwise. It
                has to be smaller than :math:`4 \\sigma` if ``sigma`` is given, than
                :math:`3 (2 \\sqrt{B} + 1)^{2} / 4` if ``relative_variance`` :math:`= B` is
                given, and than :math:`1 / 2` otherwise.
            sigma (float, optional): bound :math:`\\sigma` of the standard deviation of the
                output. Defaults to ``None``.
            relative_variance (float, optional): bound :math:`B \\geq 1` of the variance divided
                by the square of the mean. Defaults to ``None``.
            delta (float, optional): If given, the algorithm is repeated and the median is used,
                so that the probability of failure is at most ``delta``. Otherwise, the
                algorithm is executed once, with the probability of success of Ref. [1].
                Defaults to ``None``.
            constant (float, optional): constant :math:`D` of Algorithm 2 of Ref. [1], which
                has to be large enough to prove the error, and sets the number of uses of
                :math:`\\mathbf{Q}`, :math:`t_{0} = D \\sqrt{\\log_{2}(1 / \\epsilon)} /
                \\epsilon`. Defaults to :math:`1`.
            backend (:class:`qibo.backends.abstract.Backend`, optional): backend to be used in
                the execution of the circuits. If ``None``, it uses the current backend.
                Defaults to ``None``.

        Returns:
            float: estimate of the mean of the output.
        """
        backend = _check_backend(backend)

        for name, number, upper in (
            ("epsilon", epsilon, math.inf),
            ("sigma", sigma, math.inf),
            ("relative_variance", relative_variance, math.inf),
            ("constant", constant, math.inf),
            ("delta", delta, 1.0),
        ):
            if number is not None and (
                isinstance(number, bool)
                or not isinstance(number, Real)
                or not 0 < number < upper
            ):
                raise_error(
                    ValueError,
                    f"{name} must be a positive number"
                    + (" smaller than 1" if upper == 1.0 else "")
                    + f", but it is {number}.",
                )

        if sigma is not None and relative_variance is not None:
            raise_error(
                ValueError, "sigma and relative_variance cannot be given together."
            )

        if relative_variance is not None and relative_variance < 1:
            raise_error(
                ValueError,
                f"relative_variance must be at least 1, but it is {relative_variance}.",
            )

        # the accuracy of Algorithm 2, which is called by the others, has to be smaller than 1/2
        if sigma is not None:
            limit = 4 * sigma
        elif relative_variance is not None:
            limit = 3 * (2 * math.sqrt(relative_variance) + 1) ** 2 / 4
        else:
            limit = 1 / 2

        if epsilon >= limit:
            raise_error(
                ValueError,
                f"epsilon must be smaller than {limit} for the given arguments, "
                + f"but it is {epsilon}.",
            )

        if sigma is None and any(value < 0 for value in self.values):
            raise_error(
                ValueError,
                "The values must be non-negative if sigma is not given.",
            )

        # the repetitions of Algorithm 2 only use independent shots of the circuits
        if sigma is None and relative_variance is None:
            failure = 1 / 5 if delta is None else delta

            return self._positive(self.values, epsilon, failure, constant, backend)

        # probability of success of each algorithm, and of the median of its repetitions
        success = 2 / 3 if sigma is not None else 3 / 4
        estimates = []
        for _ in range(_repetitions(success, delta)):
            if sigma is not None:
                # the mean of the normalized values is the sample plus the means of the positive
                # and negative parts of the difference with it, which are scaled to be small
                scaled = [value / sigma for value in self.values]
                sample = scaled[self._sample(1, backend)[0]]
                difference = [value - sample for value in scaled]
                negative = [-value / 4 if value < 0 else 0.0 for value in difference]
                positive = [value / 4 if value >= 0 else 0.0 for value in difference]

                accuracy = epsilon / (32 * sigma)
                minus = self._positive(negative, accuracy, 1 / 9, constant, backend)
                plus = self._positive(positive, accuracy, 1 / 9, constant, backend)
                estimates.append(sigma * (sample - 4 * minus + 4 * plus))
            else:
                # the mean of the samples normalizes the values to have a mean close to 1
                outcomes = self._sample(math.ceil(32 * relative_variance), backend)
                mean = sum(self.values[outcome] for outcome in outcomes) / len(outcomes)

                accuracy = (
                    2 * epsilon / (3 * (2 * math.sqrt(relative_variance) + 1) ** 2)
                )
                scaled = [value / mean if mean > 0 else 0.0 for value in self.values]
                estimates.append(
                    mean * self._positive(scaled, accuracy, 1 / 8, constant, backend)
                )

        return sorted(estimates)[len(estimates) // 2]

    def _bounded(
        self,
        values: list[float],
        nuses: int,
        delta: float,
        nruns: int,
        backend: Backend,
    ) -> list[float]:
        """Algorithm 1 of Ref. [1], for values between 0 and 1, which are not scaled.

        The circuit is simulated once, and the independent shots of the estimation are used for
        the repetitions of the algorithm, in order to take the medians.

        Args:
            values (list[float]): values :math:`f(x)` in :math:`[0, 1]` for each outcome.
            nuses (int): number :math:`t` of uses of :math:`\\mathbf{Q}`, at least 2, which is
                rounded up to a number of counting qubits, with :math:`2^{t_{c}} - 1 \\geq t`.
            delta (float): probability of failure of each run of the algorithm.
            nruns (int): number of independent runs of the algorithm.
            backend (:class:`qibo.backends.abstract.Backend`): backend to be used in the
                execution of the circuits.

        Returns:
            list[float]: estimates of :math:`\\sum_{x} p(x) f(x)`, one for each run.
        """
        # all of the values are zero, so that the amplitude is zero
        if all(value == 0 for value in values):
            return [0.0] * nruns

        nqubits = self.algorithm.nqubits
        flag = nqubits

        # rotation of the flag, controlled by the measured qubits, with amplitude sqrt(f(x))
        angles = [
            2 * float(backend.arcsin(backend.sqrt(min(value, 1.0)))) for value in values
        ]
        circuit = Circuit(nqubits + 1)
        circuit.add(self.algorithm.on_qubits(*range(nqubits)))
        circuit.add(
            gates.Multiplexor(
                self.qubits, flag, backend.cast(angles, dtype=backend.float64)
            )
        )

        # the flag in the state 1 is the good state
        oracle = Circuit(nqubits + 1)
        oracle.add(gates.Z(flag))

        ncounting = max(1, math.ceil(math.log2(nuses + 1)))
        self.circuit = QAE(circuit, oracle, ncounting)

        nrepetitions = _repetitions(8 / math.pi**2, delta)
        shots = backend.execute_circuit(
            self.circuit, nshots=nruns * nrepetitions
        ).samples(binary=False)
        estimates = [
            math.sin(math.pi * int(shot) / 2**ncounting) ** 2 for shot in shots
        ]

        return [
            sorted(estimates[run * nrepetitions : (run + 1) * nrepetitions])[
                nrepetitions // 2
            ]
            for run in range(nruns)
        ]

    def _positive(
        self,
        values: list[float],
        epsilon: float,
        delta: float,
        constant: float,
        backend: Backend,
    ) -> float:
        """Algorithm 2 of Ref. [1], for non-negative values.

        Args:
            values (list[float]): non-negative values :math:`v(x)` for each outcome.
            epsilon (float): accuracy, smaller than :math:`1 / 2`.
            delta (float): probability of failure.
            constant (float): constant :math:`D` of the algorithm.
            backend (:class:`qibo.backends.abstract.Backend`): backend to be used in the
                execution of the circuits.

        Returns:
            float: estimate of :math:`\\sum_{x} p(x) v(x)`.
        """
        ninterval = math.ceil(math.log2(1 / epsilon))
        nuses = max(
            math.ceil(constant * math.sqrt(math.log2(1 / epsilon)) / epsilon), 2
        )

        # values below 1, and the values in the intervals [2^(l - 1), 2^l), rescaled by 2^l, with
        # their weights 2^l and the probabilities of failure of their estimations
        pieces = [(1, [value if 0 <= value < 1 else 0.0 for value in values], 1 / 10)]
        for exponent in range(1, ninterval + 1):
            interval = [
                value / 2**exponent
                if 2 ** (exponent - 1) <= value < 2**exponent
                else 0.0
                for value in values
            ]
            pieces.append((2**exponent, interval, 1 / (10 * ninterval)))

        # each of the runs of the algorithm adds one run of the estimation of each piece
        nruns = _repetitions(4 / 5, delta)
        totals = [0.0] * nruns
        for weight, interval, failure in pieces:
            runs = self._bounded(interval, nuses, failure, nruns, backend)
            totals = [total + weight * run for total, run in zip(totals, runs)]

        return sorted(totals)[nruns // 2]

    def _sample(self, nshots: int, backend: Backend) -> list[int]:
        """Samples the outcomes of the measured qubits of the algorithm.

        Args:
            nshots (int): number of samples.
            backend (:class:`qibo.backends.abstract.Backend`): backend to be used in the
                execution of the circuit.

        Returns:
            list[int]: integers :math:`x` of the outcomes.
        """
        circuit = Circuit(self.algorithm.nqubits)
        circuit.add(self.algorithm.on_qubits(*range(self.algorithm.nqubits)))
        circuit.add(gates.M(*self.qubits))

        frequencies = backend.execute_circuit(circuit, nshots=nshots).frequencies(
            binary=True
        )

        return [
            int(bitstring, 2)
            for bitstring, count in frequencies.items()
            for _ in range(count)
        ]


def _repetitions(success: float, delta: float | None) -> int:
    """Number of repetitions of an algorithm to have a median that fails with probability delta.

    Following the powering lemma of Ref. [1] and Hoeffding's inequality, the median of
    :math:`N \\geq \\ln(1 / \\delta) / (2 (p - 1/2)^{2})` repetitions with probability of
    success :math:`p > 1/2` fails with probability at most :math:`\\delta`.

    Args:
        success (float): probability :math:`p` of success of the algorithm.
        delta (float, optional): probability of failure. If ``None``, or if it is not smaller
            than :math:`1 - p`, one repetition is used.

    Returns:
        int: odd number of repetitions.
    """
    if delta is None or delta >= 1 - success:
        return 1

    repetitions = math.ceil(math.log(1 / delta) / (2 * (success - 1 / 2) ** 2))

    return repetitions + 1 - repetitions % 2
