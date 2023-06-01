import numpy as np
import scipy.stats

from qibo import gates
from qibo.config import raise_error
from qibo.models import Circuit


class IQAE:
    """Model that performs the Iterative Quantum Ampltidue Estimation algorithm.

    The implemented class in this code utilizes the Iterative Quantum Amplitude Estimation (IQAE)
    algorithm, which was proposed in [1]. The algorithm provides an estimated output that, with a
    probability `alpha', differs from the target value by `epsilon'. Both `alpha' and `epsilon' can
    be specified.

    Unlike Brassard's original QAE algorithm [2], this implementation does not rely on Quantum Phase
    Estimation but instead is based solely on Grover's algorithm. The IQAE algorithm employs a series
    of carefully selected Grover iterations to determine an estimate for the target amplitude.

    References:
    [1]: Grinko, D., Gacon, J., Zoufal, C., & Woerner, S. (2019). Iterative Quantum Amplitude Estimation.
    arXiv:1912.05559 <https://arxiv.org/abs/1912.05559>.
    [2]: Brassard, G., Hoyer, P., Mosca, M., & Tapp, A. (2000). Quantum Amplitude Amplification and
    Estimation. arXiv:quant-ph/0005055 <http://arxiv.org/abs/quant-ph/0005055>.

    Args:
        circuit_A (:class:`qibo.circuit`): quantum circuit that specifies the QAE problem
        circuit_Q (:class:`qibo.circuit`): quantum circuit of the Grover/Amplification operator

        alpha (float): confidence level, the target probability is 1 - `alpha', has values between 0 and 1
        epsilon (float): target precision for estimation target `a`, has values between 0 and 0.5
        method (string): statistical method used to estimate the confidence intervals in
                each iteration, can be either 'chernoff' (default) for the Chernoff intervals or 'beta' for the
                Clopper-Pearson intervals
        n_shots (int): number of shots
    Raises:
        ValueError: If epsilon is not in (0, 0.5]
        ValueError: If alpha is not in (0, 1)
        ValueError: If method is not supported
        ValueError: If circuit_A or circuit_Q are not provided or if the number of qubits in circuit_A is
        greater than in circuit_Q

    Example:
            import numpy as np
            from qibo import gates
            from qibo.models import Circuit
            from qibo.models.iqae import IQAE
            ...
            #Here the circuits A and Q are defined
            ...
            iae = IQAE(circuit_A=A, circuit_Q=Q)
            results=iae.execute()
            print(results.estimation)
    """

    def __init__(
        self,
        circuit_A=None,
        circuit_Q=None,
        alpha=0.05,
        epsilon=0.005,
        n_shots=1024,
        method="chernoff",
    ):
        self.circuit_A = circuit_A
        if not circuit_A:
            raise_error(
                ValueError,
                "Cannot create IQAE model if the A circuit is not specified.",
            )
        self.circuit_Q = circuit_Q
        if not circuit_Q:
            raise_error(
                ValueError,
                "Cannot create IQAE model if the Q circuit is not specified.",
            )
        if circuit_A.nqubits > circuit_Q.nqubits:
            raise_error(
                ValueError,
                "The number of qubits for Q must be greater or equal than the number"
                "of qubits of A ",
            )
        if not isinstance(n_shots, int):
            raise_error(
                ValueError,
                "The number of shots must be an integer number",
            )
        # validate ranges of input arguments
        if not 0 < epsilon <= 0.5:
            raise_error(ValueError, f"Epsilon must be in (0, 0.5], but is {epsilon}")

        if not 0 < alpha < 1:
            raise_error(
                ValueError,
                f"The confidence level alpha must be in (0, 1), but is {alpha}",
            )

        if method not in {"chernoff", "beta"}:
            raise_error(
                ValueError,
                f"The confidence interval method must be chernoff or beta, but is {method}",
            )

        self.alpha = alpha
        self.epsilon = epsilon
        self.n_shots = n_shots
        self.method = method

    # QUANTUM CIRCUIT:
    def construct_qae_circuit(self, k):
        """
        Generates quantum circuit for QAE
        Args:
            k: number of times the amplification operator Q is applied
        Return:
            qc: quantum circuit of the QAE algorithm
        """
        initialization_circuit_A = self.circuit_A
        amplification_circuit_Q = self.circuit_Q

        qc = Circuit(amplification_circuit_Q.nqubits)

        qc.add(
            initialization_circuit_A.on_qubits(
                *range(0, initialization_circuit_A.nqubits, 1)
            )
        )
        for i in range(k):
            qc = qc + amplification_circuit_Q
        qc.add(gates.M(initialization_circuit_A.nqubits - 1))
        return qc

    # CLASSICAL POSTPROCESSING FOR THE IQAE
    def clopper_pearson(self, count, n, alpha):
        """Calculates the confidence interval for the quantity to estimate `a'
        Args:
            count: number of successes
            n: total number of trials
            alpha: significance level. Must be in (0, 0.5)
        Returns
            The confidence interval a_min,a_max
        """
        beta_prob_function = scipy.stats.beta.ppf
        a_min = beta_prob_function(alpha / 2, count, n - count + 1)
        a_max = beta_prob_function(1 - alpha / 2, count + 1, n - count)
        if np.isnan(a_min):
            a_min = 0
        if np.isnan(a_max):
            a_max = 1
        return a_min, a_max

    def h_calc_CP(self, n_successes, n_total_shots, upper_bound_T):
        """
        Calculates the h function
        Args:
            n_successes: number of successes
            n_total_shots: total number of trials
            upper_bound_T: maximum number of rounds to achieve the desired absolute error
        Returns:
            The h function for the given inputs
        """
        a_min, a_max = self.clopper_pearson(
            n_successes, n_total_shots, alpha=(self.alpha / upper_bound_T)
        )
        return np.abs(np.arccos(1 - 2 * a_max) - np.arccos(1 - 2 * a_min)) / 2

    def calc_L_range_CP(self, n_shots, upper_bound_T):
        """
        Calculate the confidence interval for the Clopper-Pearson method
        Args:
            n_shots: number of shots
            upper_bound_T: maximum number of rounds to achieve the desired absolute error
        Returns:
            max_L, min_L: the maximum and minimum possible error which could be returned
            on a given iteration
        """
        x = np.linspace(0, np.pi, 10000)
        x_domain = [x <= 1.0 + 1 / 10 / n_shots]
        y = [
            self.h_calc_CP(int(t * n_shots), n_shots, upper_bound_T)
            for t in x[tuple(x_domain)]
        ]
        max_L = np.max(y) / (2 * np.pi)
        min_L = np.min(y) / (2 * np.pi)
        return max_L, min_L

    def calc_L_range_CH(self, n_shots, upper_bound_T):
        """
        Calculate the confidence interval for the Chernoff method
        Args:
            n_shots: number of shots
            upper_bound_T: maximum number of rounds to achieve the desired absolute error
        Returns:
            max_L, min_L: the maximum and minimum possible error which could be returned
            on a given iteration
        """
        max_L = (
            np.arcsin(
                (2 / (n_shots) * np.log(2 * upper_bound_T / self.alpha)) ** (1 / 4)
            )
            / 2
            / np.pi
        )
        min_L = np.arcsin(np.sin(max_L) ** 2)
        return max_L, min_L

    def find_next_k(self, uppercase_k_i, up_i, theta_l, theta_u, r=2):
        """Find the largest integer uppercase_k such that the interval uppercase_k*[theta_l,theta_u]
        lies completely in [0, pi] or [pi, 2pi].

        Args:
            uppercase_k_i: the current uppercase_k such uppercase_k=4k+2, where k is the power 
            of the Q operator
            up_i: boolean flag of whether theta_interval lies in the
                upper half-circle [0, pi] or in the lower one [pi, 2pi]
            theta_l: the current lower limit of the confidence interval for the angle theta
            theta_u: the current upper limit of the confidence interval for the angle theta
            r: lower bound for K

        Returns:
            The next power K_i, and boolean flag for the extrapolated interval
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

    # EXECUTING THE ALGORITHM
    def execute(self, backend=None):
        """Execute IQAE algorithm.

        Args:
            backend: the qibo backend.

        Returns:
            An IterativeAmplitudeEstimation results object.
        """
        if backend is None:
            from qibo.backends import GlobalBackend

            backend = GlobalBackend()
        # Initializing all parameters
        k = [0]
        #uppercase_k=4k+2
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
        upper_bound_T = int(np.log2(np.pi / (8 * self.epsilon))) + 1
        n_total_shots = self.n_shots
        num_oracle_queries = 0

        if self.method == "chernoff":
            # Chernoff method
            max_L, min_L = self.calc_L_range_CH(n_total_shots, upper_bound_T)

        else:
            # Clopper-Pearson (beta) method
            max_L, min_L = self.calc_L_range_CP(n_total_shots, upper_bound_T)

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
            if uppercase_k_i > int(max_L / eps):
                n_shots_i = int((max_L / eps) * self.n_shots / uppercase_k_i / 10)
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
                    np.log(2 * upper_bound_T / self.alpha) / 2 / n_total_shots
                )
                a_min_i = max(0, a - delta_a)
                a_max_i = min(1, a + delta_a)
            else:
                a_min_i, a_max_i = self.clopper_pearson(
                    sum_total_samples, n_total_shots, alpha=self.alpha / upper_bound_T
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


# THE CLASS WITH THE RESULTS
class IterativeAmplitudeEstimationResult:
    """The ``IterativeAmplitudeEstimation`` result object."""

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
