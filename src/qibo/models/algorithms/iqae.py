import numpy as np
import scipy.stats

from qibo import gates
from qibo.config import raise_error
from qibo.models.circuit import Circuit


class IQAE:
    """Iterative Quantum Amplitude Estimation (IQAE).

    Let :math:`\\mathcal{A}` be a circuit on :math:`n + 1` qubits, such that

    .. math::
        \\mathcal{A} \\ket{0}_{n} \\ket{0} = \\sqrt{1 - a} \\, \\ket{\\psi_{0}}_{n} \\ket{0}
        + \\sqrt{a} \\, \\ket{\\psi_{1}}_{n} \\ket{1},

    where :math:`\\ket{\\psi_{0}}_{n}` and :math:`\\ket{\\psi_{1}}_{n}` are normalized states
    and :math:`a \\in [0, 1]` is the unknown probability of measuring the last qubit in the state
    :math:`\\ket{1}`. Let :math:`\\mathbf{Q}` be the operator of
    :func:`qibo.models.amplification_operator` for :math:`\\mathcal{A}` and for the oracle that
    multiplies by :math:`-1` the states with the last qubit in the state :math:`\\ket{1}`,
    *e.g.* a :class:`qibo.gates.Z` gate on the last qubit. For :math:`a = \\sin^{2}(\\theta_{a})`
    and any integer :math:`k \\geq 0`,

    .. math::
        \\mathbf{Q}^{k} \\mathcal{A} \\ket{0}_{n} \\ket{0}
        = \\cos((2k + 1) \\theta_{a}) \\ket{\\psi_{0}}_{n} \\ket{0}
        + \\sin((2k + 1) \\theta_{a}) \\ket{\\psi_{1}}_{n} \\ket{1},

    so the probability of measuring the last qubit in the state :math:`\\ket{1}` is
    :math:`\\sin^{2}((2k + 1) \\theta_{a})`. In contrast with :func:`qibo.models.QAE`, IQAE does
    not use quantum phase estimation, hence it needs neither counting qubits nor controlled
    powers of :math:`\\mathbf{Q}`. It only applies :math:`\\mathbf{Q}` to
    :math:`\\mathcal{A}\\ket{0}` and measures the last qubit. Starting from the interval
    :math:`[\\theta_{l}, \\theta_{u}] = [0, \\pi / 2]` for :math:`\\theta_{a}`, each iteration
    :math:`i`

    1. finds the largest power :math:`k_{i}` such that :math:`K_{i} = 4 k_{i} + 2 \\geq 2 K_{i-1}`
       and the interval :math:`K_{i} [\\theta_{l}, \\theta_{u}]`, modulo :math:`2 \\pi`, lies
       completely in :math:`[0, \\pi]` or in :math:`[\\pi, 2 \\pi]`. If there is none, it keeps
       :math:`k_{i-1}`;
    2. measures the last qubit of :math:`\\mathbf{Q}^{k_{i}} \\mathcal{A} \\ket{0}` ``n_shots``
       times, or fewer times in the last iterations, where more shots would not improve the
       result (the "no-overshooting" condition). With all of the shots of the iterations that
       share :math:`k_{i}`, it computes a confidence interval of
       :math:`\\sin^{2}((2 k_{i} + 1) \\theta_{a}) = (1 - \\cos(K_{i} \\theta_{a})) / 2`;
    3. inverts the cosine, which is unambiguous in each half of the circle, and shrinks
       :math:`[\\theta_{l}, \\theta_{u}]`.

    The algorithm stops when :math:`\\theta_{u} - \\theta_{l} \\leq 2 \\epsilon`, where
    :math:`\\epsilon` is ``epsilon``, and it returns the interval
    :math:`[a_{l}, a_{u}] = [\\sin^{2}(\\theta_{l}), \\sin^{2}(\\theta_{u})]`. The significance
    level of each confidence interval is the one of the algorithm, :math:`\\alpha`, divided by the
    maximum number :math:`\\lceil \\log_{2}(\\pi / (8 \\epsilon)) \\rceil` of rounds, a round being
    the set of iterations with the same :math:`k_{i}`. Then, :math:`a_{u} - a_{l} \\leq 2 \\epsilon`
    and :math:`a` lies in :math:`[a_{l}, a_{u}]` with probability at least :math:`1 - \\alpha`.
    Thus, the estimation :math:`(a_{l} + a_{u}) / 2` differs from :math:`a` by at most
    :math:`\\epsilon` with that probability. If ``n_shots`` is not larger than

    .. math::
        N_{\\max}(\\epsilon, \\alpha) = \\left\\lceil \\frac{32}{(1 - 2 \\sin(\\pi / 14))^{2}}
        \\log\\left(\\frac{2}{\\alpha} \\log_{2}\\left(\\frac{\\pi}{4 \\epsilon}\\right)\\right)
        \\right\\rceil,

    the algorithm ends after at most
    :math:`\\lceil \\log_{2}(\\pi / (8 \\epsilon)) \\rceil` rounds and
    :math:`\\mathbf{Q}` is applied fewer than
    :math:`(50 / \\epsilon) \\log((2 / \\alpha) \\log_{2}(\\pi / (4 \\epsilon)))` times. This is the
    quadratic speedup over the :math:`O(1 / \\epsilon^{2})` samples of classical Monte Carlo
    simulation, up to a double-logarithmic factor.

    .. note::
        For the default values of ``epsilon`` and ``alpha``, :math:`N_{\\max} = 590`, which is
        smaller than the default ``n_shots``.

    The circuit ``circuit_q`` can act on more qubits than ``circuit_a``, *e.g.* on auxiliary
    qubits, as long as it leaves them in the state :math:`\\ket{0}`.

    Example:
        .. testcode::

            import math

            from qibo import Circuit, gates
            from qibo.models import IQAE, amplification_operator

            # A applies a Hadamard gate on the first qubit and, if it is in the state 1, a
            # rotation of angle 2 on the last qubit, so a = sin(1)^2 / 2
            algorithm = Circuit(2)
            algorithm.add(gates.H(0))
            algorithm.add(gates.CRY(0, 1, theta=2.0))

            # the oracle multiplies by -1 the states with the last qubit in the state 1
            oracle = Circuit(2)
            oracle.add(gates.Z(1))
            operator = amplification_operator(algorithm, oracle)

            iqae = IQAE(algorithm, operator, alpha=0.05, epsilon=0.01)
            result = iqae.execute()

            print(abs(result.estimation - math.sin(1.0) ** 2 / 2) <= 0.01)

        .. testoutput::

            True

    Args:
        circuit_a (:class:`qibo.models.circuit.Circuit`): circuit :math:`\\mathcal{A}` without
            measurements. Its last qubit is the one that is measured.
        circuit_q (:class:`qibo.models.circuit.Circuit`): circuit of the operator
            :math:`\\mathbf{Q}`, which has to act on at least as many qubits as ``circuit_a``.
        alpha (float, optional): significance level :math:`\\alpha \\in (0, 1)`, *i.e.* the
            probability that the interval returned does not contain :math:`a` is at most
            ``alpha``. Defaults to :math:`0.05`.
        epsilon (float, optional): target accuracy :math:`\\epsilon \\in (0, 0.5]` of the
            estimation of :math:`a`. Defaults to :math:`0.005`.
        n_shots (int, optional): number of shots of each iteration. Defaults to :math:`1024`.
        method (str, optional): method used to compute the confidence intervals of each
            iteration, either ``"chernoff"`` for the Chernoff-Hoeffding intervals or ``"beta"``
            for the Clopper-Pearson intervals. Defaults to ``"chernoff"``.

    References:
        1. D. Grinko, J. Gacon, C. Zoufal, and S. Woerner, *Iterative quantum amplitude
        estimation*, `npj Quantum Inf. 7, 52 (2021) <https://doi.org/10.1038/s41534-021-00379-1>`_,
        `arXiv:1912.05559 <https://arxiv.org/abs/1912.05559>`_.

        2. G. Brassard, P. Høyer, M. Mosca, and A. Tapp, *Quantum Amplitude Amplification and
        Estimation*, `Contemporary Mathematics 305, 53 (2002)
        <https://doi.org/10.1090/conm/305/05215>`_.

        3. C. J. Clopper and E. S. Pearson, *The use of confidence or fiducial limits illustrated
        in the case of the binomial*, Biometrika 26, 404 (1934).

        4. W. Hoeffding, *Probability inequalities for sums of bounded random variables*,
        J. Am. Stat. Assoc. 58, 13 (1963).
    """

    def __init__(
        self,
        circuit_a,
        circuit_q,
        alpha=0.05,
        epsilon=0.005,
        n_shots=1024,
        method="chernoff",
    ):
        self.circuit_a = circuit_a
        self.circuit_q = circuit_q
        if circuit_a.nqubits > circuit_q.nqubits:
            raise_error(
                ValueError,
                "The number of qubits for Q must be greater or equal than the number"
                "of qubits of A.",
            )
        if not isinstance(n_shots, int):
            raise_error(
                ValueError,
                "The number of shots must be an integer number.",
            )
        # validate ranges of input arguments
        if not 0 < epsilon <= 0.5:
            raise_error(ValueError, f"Epsilon must be in (0, 0.5], but is {epsilon}.")

        if not 0 < alpha < 1:
            raise_error(
                ValueError,
                f"The confidence level alpha must be in (0, 1), but is {alpha}.",
            )

        if method not in {"chernoff", "beta"}:
            raise_error(
                ValueError,
                f"The confidence interval method must be chernoff or beta, but is {method}.",
            )

        self.alpha = alpha
        self.epsilon = epsilon
        self.n_shots = n_shots
        self.method = method

    def construct_qae_circuit(self, k):
        """Creates the circuit that applies :math:`\\mathbf{Q}^{k}` to
        :math:`\\mathcal{A}\\ket{0}` and measures the last qubit of ``circuit_a``.

        Args:
            k (int): number :math:`k \\geq 0` of applications of ``circuit_q``.

        Returns:
            :class:`qibo.models.circuit.Circuit`: circuit with as many qubits as ``circuit_q``.
        """
        initialization_circuit_a = self.circuit_a
        amplification_circuit_q = self.circuit_q

        qc = Circuit(amplification_circuit_q.nqubits)

        qc.add(
            initialization_circuit_a.on_qubits(
                *range(0, initialization_circuit_a.nqubits, 1)
            )
        )
        for _ in range(k):
            qc = qc + amplification_circuit_q
        qc.add(gates.M(initialization_circuit_a.nqubits - 1))
        return qc

    def clopper_pearson(self, count, n, alpha):
        """Calculates the Clopper-Pearson confidence interval of the probability :math:`a`.

        Args:
            count (int): number of successes, *i.e.* of measurements of the state
                :math:`\\ket{1}`.
            n (int): total number of trials.
            alpha (float): significance level, *i.e.* the probability that the interval does not
                contain :math:`a` is at most ``alpha``.

        Returns:
            tuple: lower and upper bounds of the interval. A bound that cannot be computed, which
            happens if ``count`` is :math:`0` or ``n``, is replaced by :math:`0` or :math:`1`.
        """
        beta_prob_function = scipy.stats.beta.ppf
        a_min = beta_prob_function(alpha / 2, count, n - count + 1)
        a_max = beta_prob_function(1 - alpha / 2, count + 1, n - count)
        if np.isnan(a_min):
            a_min = 0
        if np.isnan(a_max):
            a_max = 1
        return a_min, a_max

    def h_calc_CP(self, n_successes, n_total_shots, upper_bound_t):
        """Calculates half of the width of the Clopper-Pearson interval of :math:`K \\theta_{a}`.

        The Clopper-Pearson interval :math:`[a_{\\min}, a_{\\max}]` of the probability :math:`a`
        has significance level ``alpha`` / ``upper_bound_t``, and the result is
        :math:`|\\arccos(1 - 2 a_{\\max}) - \\arccos(1 - 2 a_{\\min})| / 2`.

        Args:
            n_successes (int): number of successes.
            n_total_shots (int): total number of trials.
            upper_bound_t (int): maximum number :math:`T` of rounds needed to achieve the target
                accuracy.

        Returns:
            float: half of the width of the interval of :math:`K \\theta_{a}`, in radians.
        """
        a_min, a_max = self.clopper_pearson(
            n_successes, n_total_shots, alpha=(self.alpha / upper_bound_t)
        )
        return np.abs(np.arccos(1 - 2 * a_max) - np.arccos(1 - 2 * a_min)) / 2

    def calc_L_range_CP(self, n_shots, upper_bound_t):
        """Calculates the maximum and minimum errors of an iteration with Clopper-Pearson intervals.

        They are the maximum and the minimum of :meth:`h_calc_CP` over the fractions of successes
        in :math:`[0, 1]`, divided by :math:`2 \\pi`. The maximum is the quantity
        :math:`L_{\\max}` of the no-overshooting condition.

        Args:
            n_shots (int): number of shots.
            upper_bound_t (int): maximum number :math:`T` of rounds needed to achieve the target
                accuracy.

        Returns:
            tuple: maximum and minimum errors that an iteration can return, in units of
            :math:`2 \\pi`.
        """
        x = np.linspace(0, np.pi, 10000)
        x_domain = [x <= 1.0 + 1 / 10 / n_shots]
        y = [
            self.h_calc_CP(int(t * n_shots), n_shots, upper_bound_t)
            for t in x[tuple(x_domain)]
        ]
        max_L = np.max(y) / (2 * np.pi)
        min_L = np.min(y) / (2 * np.pi)
        return max_L, min_L

    def calc_L_range_CH(self, n_shots, upper_bound_t):
        """Calculates the maximum and minimum errors of an iteration, with Chernoff-Hoeffding.

        The maximum error is

        .. math::
            L_{\\max} = \\frac{1}{2 \\pi} \\arcsin\\left(\\left(\\frac{2}{N_{shots}}
            \\log\\left(\\frac{2 T}{\\alpha}\\right)\\right)^{1 / 4}\\right),

        which is used in the no-overshooting condition.

        Args:
            n_shots (int): number of shots :math:`N_{shots}`.
            upper_bound_t (int): maximum number :math:`T` of rounds needed to achieve the target
                accuracy.

        Returns:
            tuple: maximum error, in units of :math:`2 \\pi`, and a lower value that is not used by
            :meth:`execute`.
        """
        max_L = (
            np.arcsin(
                (2 / (n_shots) * np.log(2 * upper_bound_t / self.alpha)) ** (1 / 4)
            )
            / 2
            / np.pi
        )
        min_L = np.arcsin(np.sin(max_L) ** 2)
        return max_L, min_L

    def find_next_k(self, uppercase_k_i, up_i, theta_l, theta_u, r=2):
        """Finds the next factor :math:`K`, which is the procedure *FindNextK* of [1].

        It is the largest :math:`K = 4k + 2 \\geq r K_{i}` such that the interval
        :math:`K [\\theta_{l}, \\theta_{u}]` lies completely in :math:`[0, \\pi]` or in
        :math:`[\\pi, 2 \\pi]`, modulo :math:`2 \\pi`. All of the angles are divided by
        :math:`2 \\pi`.

        Args:
            uppercase_k_i (int): current factor :math:`K_{i} = 4 k_{i} + 2`, where :math:`k_{i}`
                is the power of ``circuit_q``.
            up_i (bool): whether the current interval of :math:`K_{i} \\theta_{a}` lies in the
                upper half of the circle, :math:`[0, \\pi]`, or in the lower one.
            theta_l (float): lower limit of the current interval of :math:`\\theta_{a}`.
            theta_u (float): upper limit of the current interval of :math:`\\theta_{a}`.
            r (int, optional): minimum ratio :math:`K / K_{i}`. Defaults to :math:`2`.

        Returns:
            tuple: next factor :math:`K = 4k + 2`, and whether the extrapolated interval lies in
            the upper half of the circle. If there is no valid :math:`K`, it returns
            ``uppercase_k_i`` and ``up_i``.
        """
        uppercase_k_max = int(1 / (2 * (theta_u - theta_l)))
        uppercase_k = uppercase_k_max - (uppercase_k_max - 2) % 4
        while uppercase_k >= r * uppercase_k_i:
            theta_min = uppercase_k * theta_l - int(uppercase_k * theta_l)
            theta_max = uppercase_k * theta_u - int(uppercase_k * theta_u)
            if int(uppercase_k * theta_u) == int(uppercase_k * theta_l):
                if theta_max <= 1 / 2 and theta_min <= 1 / 2:
                    up = True
                    return (uppercase_k, up)
                elif theta_max >= 1 / 2 and theta_min >= 1 / 2:
                    up = False
                    return (uppercase_k, up)
            uppercase_k -= 4
        return (uppercase_k_i, up_i)

    def execute(self, backend=None):
        """Executes the IQAE algorithm.

        Args:
            backend (:class:`qibo.backends.abstract.Backend`, optional): backend used to execute
                the circuits. If ``None``, it uses the current backend. Defaults to ``None``.

        Returns:
            :class:`qibo.models.IterativeAmplitudeEstimationResult`: result with the estimation
            of :math:`a`, its interval and the information of each iteration.
        """
        from qibo.backends import _check_backend

        backend = _check_backend(backend)

        # Initializing all parameters
        k = [0]
        # uppercase_k=4k+2
        uppercase_k = [2]

        theta_u = 1 / 4
        theta_l = 0
        theta_intervals = [theta_l, theta_u]
        a_intervals = [
            np.sin(2 * np.pi * theta_l) ** 2,
            np.sin(2 * np.pi * theta_u) ** 2,
        ]
        a_min = [0]
        a_max = [1]
        theta_dif = np.abs(theta_u - theta_l)
        up = [True]
        samples_history = []
        n_shots_history = []

        eps = self.epsilon / (2 * np.pi)
        upper_bound_t = int(np.log2(np.pi / (8 * self.epsilon))) + 1
        n_total_shots = self.n_shots
        num_oracle_queries = 0

        if self.method == "chernoff":
            # Chernoff method
            max_L, _min_L = self.calc_L_range_CH(n_total_shots, upper_bound_t)

        else:
            # Clopper-Pearson (beta) method
            max_L, _min_L = self.calc_L_range_CP(n_total_shots, upper_bound_t)

        i = 0
        while theta_dif > 2 * eps:
            i = i + 1
            uppercase_k_i, up_i = self.find_next_k(
                uppercase_k[-1], up[-1], theta_l, theta_u
            )
            k_i = int((uppercase_k_i - 2) / 4)
            uppercase_k.append(uppercase_k_i)
            up.append(up_i)
            k.append(k_i)
            # Checking the no-overshooting condition
            if uppercase_k_i > int(max_L / eps):
                # We ensure to not make unnecessary measurement shots at last iterations of the algorithm
                n_shots_i = int((max_L / eps) * self.n_shots / uppercase_k_i / 10)
                # To avoid having a null number of shots
                if n_shots_i == 0:
                    n_shots_i = 1
            else:
                n_shots_i = self.n_shots

            # Calling and executing the quantum circuit
            qc = self.construct_qae_circuit(k_i)
            samples = backend.execute_circuit(qc, nshots=n_shots_i).frequencies(
                binary=True
            )["1"]

            samples_history.append(samples)
            n_shots_history.append(n_shots_i)

            num_oracle_queries += n_shots_i * k_i

            m = 1
            if i > 1:
                while uppercase_k[i - m] == uppercase_k[i] and i >= m + 1:
                    m += 1
                sum_total_samples = sum([samples_history[j] for j in range(i - m, i)])
                n_total_shots = sum([n_shots_history[j] for j in range(i - m, i)])

            else:
                sum_total_samples = samples
                n_total_shots = n_shots_i

            a = sum_total_samples / n_total_shots

            if self.method == "chernoff":
                delta_a = np.sqrt(
                    np.log(2 * upper_bound_t / self.alpha) / 2 / n_total_shots
                )
                a_min_i = max(0, a - delta_a)
                a_max_i = min(1, a + delta_a)
            else:
                a_min_i, a_max_i = self.clopper_pearson(
                    sum_total_samples, n_total_shots, alpha=self.alpha / upper_bound_t
                )
            a_min.append(a_min_i)
            a_max.append(a_max_i)

            if up_i:
                theta_min_i = np.arccos(1 - 2 * a_min_i) / 2 / np.pi
                theta_max_i = np.arccos(1 - 2 * a_max_i) / 2 / np.pi
            else:
                theta_min_i = 1 - np.arccos(1 - 2 * a_max_i) / 2 / np.pi
                theta_max_i = 1 - np.arccos(1 - 2 * a_min_i) / 2 / np.pi

            # the interval lies in a single turn, which is found from its center, because the
            # ends can be on the border of two turns, where rounding errors change the turn
            theta = int(uppercase_k_i * (theta_u + theta_l) / 2)

            theta_u = (theta + theta_max_i) / uppercase_k_i
            theta_l = (theta + theta_min_i) / uppercase_k_i
            theta_dif = np.abs(theta_u - theta_l)
            theta_intervals.append([theta_l, theta_u])

            a_l = np.sin(2 * np.pi * theta_l) ** 2
            a_u = np.sin(2 * np.pi * theta_u) ** 2
            a_intervals.append([a_l, a_u])

        result = IterativeAmplitudeEstimationResult()
        result.alpha = self.alpha
        result.epsilon_target = self.epsilon
        result.epsilon_estimated = (a_u - a_l) / 2
        result.estimate_intervals = a_intervals
        result.num_oracle_queries = num_oracle_queries
        result.estimation = (a_u + a_l) / 2
        result.theta_intervals = theta_intervals
        result.k_list = k
        result.ratios = samples_history
        result.shots = n_shots_history

        return result


class IterativeAmplitudeEstimationResult:
    """Result of :meth:`qibo.models.IQAE.execute`.

    Attributes:
        alpha (float): significance level :math:`\\alpha`.
        epsilon_target (float): target accuracy :math:`\\epsilon`.
        epsilon_estimated (float): estimated accuracy, *i.e.* half of the width
            :math:`(a_{u} - a_{l}) / 2` of the final interval.
        estimation (float): estimation :math:`(a_{l} + a_{u}) / 2` of :math:`a`.
        estimate_intervals (list): initial bounds :math:`0` and :math:`1` of :math:`a`, followed by
            the interval :math:`[a_{l}, a_{u}]` after each iteration.
        theta_intervals (list): initial bounds :math:`0` and :math:`1 / 4` of
            :math:`\\theta_{a} / (2 \\pi)`, followed by the interval
            :math:`[\\theta_{l}, \\theta_{u}] / (2 \\pi)` after each iteration.
        num_oracle_queries (int): total number of applications of ``circuit_q``.
        k_list (list): power of ``circuit_q``, starting with :math:`k_{0} = 0` and followed by the
            one of each iteration.
        ratios (list): number of times that the state :math:`\\ket{1}` was measured in each
            iteration.
        shots (list): number of shots of each iteration.
    """

    def __init__(self):
        self._alpha = None
        self._epsilon_target = None
        self._epsilon_estimated = None
        self._num_oracle_queries = None
        self._estimation = None
        self._estimate_intervals = None
        self._theta_intervals = None
        self._k_list = None
        self._ratios = None
        self._shots = None
