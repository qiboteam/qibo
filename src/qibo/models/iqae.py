import numpy as np
import scipy.stats

from qibo import Circuit, gates
from qibo.config import raise_error


class IQAE:
    """Model that performs the Iterative Quantum Amplitude Estimation algorithm.

    The implemented class in this code utilizes the Iterative Quantum Amplitude
    Estimation (IQAE) algorithm, which was proposed in `arxiv:1912.05559
    <https://arxiv.org/abs/1912.05559>`_. The algorithm provides an estimated
    output that, with a probability ``alpha``, differs from the target value by
    ``epsilon``. Both ``alpha`` and ``epsilon`` can be specified.

    Unlike Brassard's original QAE algorithm `arxiv:quant-ph/0005055
    <http://arxiv.org/abs/quant-ph/0005055>`_, this implementation does not rely
    on Quantum Phase Estimation but instead is based solely on Grover's
    algorithm. The IQAE algorithm employs a series of carefully selected Grover
    iterations to determine an estimate for the target amplitude.

    Args:
        circuit_a (:class:`qibo.models.circuit.Circuit`): quantum circuit that
            specifies the QAE problem.

        circuit_q (:class:`qibo.models.circuit.Circuit`): quantum circuit of the
            Grover/Amplification operator.

        alpha (float): confidence level, the target probability is 1 - ``alpha``,
            has values between 0 and 1.

        epsilon (float): target precision for estimation target `a`, has values
            between 0 and 0.5.

        method (str): statistical method used to estimate the confidence
            intervals in each iteration, can be either `chernoff` (default) for
            the Chernoff intervals or `beta` for the Clopper-Pearson intervals.

        n_shots (int): number of shots.

    Raises:
        ValueError: If ``epsilon`` is not in (0, 0.5].
        ValueError: If ``alpha`` is not in (0, 1).
        ValueError: If ``method`` is not supported.
        ValueError: If the number of qubits in ``circuit_a`` is greater than in ``circuit_q``.

    Example:
        .. testcode::

            from qibo import Circuit, gates
            from qibo.models.iqae import IQAE

            # Defining circuit A to integrate sin(x)^2 from [0,1]
            a_circuit = Circuit(2)
            a_circuit.add(gates.H(0))
            a_circuit.add(gates.RY(q = 1, theta = 1 / 2))
            a_circuit.add(gates.CU3(0, 1, 1, 0, 0))
            # Defining circuit Q = -A S_0 A^-1 S_X
            q_circuit = Circuit(2)
            # S_X
            q_circuit.add(gates.Z(q = 1))
            # A^-1
            q_circuit = q_circuit + a_circuit.invert()
            # S_0
            q_circuit.add(gates.X(0))
            q_circuit.add(gates.X(1))
            q_circuit.add(gates.CZ(0, 1))
            # A
            q_circuit = q_circuit + a_circuit

            # Executing IQAE and obtaining the result
            iae = IQAE(a_circuit, q_circuit)
            results = iae.execute()
            integral_value = results.estimation
            integral_error = results.epsilon_estimated
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
        """Generates quantum circuit for QAE.

        Args:
            k (int): number of times the amplification operator ``circuit_q`` is applied.
        Returns:
            The quantum circuit of the QAE algorithm.
        """
        initialization_circuit_a = self.circuit_a
        amplification_circuit_q = self.circuit_q

        qc = Circuit(amplification_circuit_q.nqubits)

        qc.add(
            initialization_circuit_a.on_qubits(
                *range(0, initialization_circuit_a.nqubits, 1)
            )
        )
        for i in range(k):
            qc = qc + amplification_circuit_q
        qc.add(gates.M(initialization_circuit_a.nqubits - 1))
        return qc

    def clopper_pearson(self, count, n, alpha):
        """Calculates the confidence interval for the quantity to estimate `a`.

        Args:
            count (int): number of successes.
            n (int): total number of trials.
            alpha (float): significance level. Must be in (0, 0.5).

        Return:
            The confidence interval [a_min, a_max].
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
        """Calculates the `h` function.

        Args:
            n_successes (int): number of successes.
            n_total_shots (int): total number of trials.
            upper_bound_t (int): maximum number of rounds to achieve the desired absolute error.

        Returns:
            The h function for the given inputs.
        """
        a_min, a_max = self.clopper_pearson(
            n_successes, n_total_shots, alpha=(self.alpha / upper_bound_t)
        )
        return np.abs(np.arccos(1 - 2 * a_max) - np.arccos(1 - 2 * a_min)) / 2

    def calc_L_range_CP(self, n_shots, upper_bound_t):
        """Calculate the confidence interval for the Clopper-Pearson method.

        Args:
            n_shots (int): number of shots.
            upper_bound_t (int): maximum number of rounds to achieve the desired absolute error.

        Returns:
            max_L, min_L (float, float): The maximum and minimum possible error which could be returned
            on a given iteration.
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
        """Calculate the confidence interval for the Chernoff method.

        Args:
            n_shots (int): number of shots.
            upper_bound_t (int): maximum number of rounds to achieve the desired absolute error.

        Returns:
            max_L, min_L (float, float): The maximum and minimum possible error which could be returned
            on a given iteration.
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
        r"""Find the largest integer ``uppercase_k`` such that the interval ``uppercase_k`` * [ ``theta_l`` , ``theta_u`` ]
        lies completely in [0, `\pi`] or [`\pi`, 2 `\pi`].

        Args:
            uppercase_k_i (int): the current ``uppercase_k`` such ``uppercase_k`` = 4 ``k`` + 2,
                where ``k`` is the power of the operator ``circuit_q``.

            up_i (bool): boolean flag of whether theta_interval lies in the
                upper half-circle [0, `\pi`] or in the lower one [`\pi`, 2 `\pi`].

            theta_l (float): the current lower limit of the confidence interval for the angle theta.

            theta_u (float): the current upper limit of the confidence interval for the angle theta.

            r (int): lower bound for ``uppercase_k``.

        Returns:
            The next power `K_i`, and boolean flag for the extrapolated interval.
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
        """Execute IQAE algorithm.

        Args:
            backend: the qibo backend.

        Returns:
            A :class:`qibo.models.iqae.IterativeAmplitudeEstimationResult` results object.
        """
        if backend is None:
            from qibo.backends import GlobalBackend

            backend = GlobalBackend()
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
            max_L, min_L = self.calc_L_range_CH(n_total_shots, upper_bound_t)

        else:
            # Clopper-Pearson (beta) method
            max_L, min_L = self.calc_L_range_CP(n_total_shots, upper_bound_t)

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
            samples = qc(nshots=n_shots_i).frequencies(binary=True)["1"]

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

            theta = min(int(uppercase_k_i * theta_u), int(uppercase_k_i * theta_l))

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
    """The ``IterativeAmplitudeEstimationResult`` result object."""

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
