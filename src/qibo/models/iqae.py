import numpy as np
import scipy.stats
from qibo import gates
from qibo.models import Circuit
from qibo.config import log, raise_error

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
        A_circuit (:class:`qibo.circuit`): quantum circuit that specifies the QAE problem
        Q_circuit (:class:`qibo.circuit`): quantum circuit of the Grover/Amplification operator

        alpha (float): confidence level, the target probability is 1 - `alpha', has values between 0 and 1
        epsilon (float): target precision for estimation target `a`, has values between 0 and 0.5
        method (string): statistical method used to estimate the confidence intervals in
                each iteration, can be either 'chernoff' (default) for the Chernoff intervals or 'beta' for the
                Clopper-Pearson intervals
        N_shots (int): number of shots
    Raises:
        ValueError: If epsilon is not in (0, 0.5]
        ValueError: If alpha is not in (0, 1)
        ValueError: If method is not supported
        ValueError: If A or Q are not provided or if the number of qubits in A is greater than in Q

    Example:
            import numpy as np
            from qibo import gates
            from qibo.models import Circuit
            from qibo.models.IQAE import IQAE
            ...
            #Here the circuits A and Q are defined
            ...
            iae = IQAE(A_circuit=A, Q_circuit=Q)
            results=iae.execute()
            print(results.estimation)
    """
    

    def __init__(
        self,
        A_circuit=None,
        Q_circuit=None,
        alpha=0.05,
        epsilon=0.005,
        N_shots=1024,
        method='chernoff',
    ):

        if A_circuit:
            self.A_circuit = A_circuit
        else:
            if not A_circuit:
                raise_error(
                    ValueError,
                    "Cannot create IQAE model if the "
                    "A circuit is not specified.", )
        if A_circuit:
            self.Q_circuit = Q_circuit
        else:
            if not Q_circuit:
                raise_error(
                    ValueError,
                    "Cannot create IQAE model if the "
                    "Q circuit is not specified.",)
        if(A_circuit.nqubits>Q_circuit.nqubits):
            raise_error(
                    ValueError,
                    "The number of qubits for Q must be greater or equal than the number"
                    "of qubits of A ",)
        # validate ranges of input arguments
        if not 0 < epsilon <= 0.5:
            raise_error(ValueError,f"Epsilon must be in (0, 0.5], but is {epsilon}")

        if not 0 < alpha < 1:
            raise_error(ValueError,f"The confidence level alpha must be in (0, 1), but is {alpha}")
            
        if method not in {"chernoff", "beta"}:
            raise_error(ValueError,f"The confidence interval method must be chernoff or beta, but is {method}")

        self.nqubitsA = A_circuit.nqubits
        self.nqubitsQ = Q_circuit.nqubits
        self.alpha=alpha
        self.epsilon=epsilon
        self.N_shots=N_shots
        self.method=method


    #QUANTUM CIRCUITS:
    def construct_circuit(self,k):
        """
        Generates quantum circuit for QAE 
        Args:
            k: number of times the amplification operator Q is applied
        Return:
            qc: quantum circuit of the QAE algorithm
        """
        qc=Circuit(self.nqubitsQ)
        
        A=self.A_circuit
        Q=self.Q_circuit
        
        qc.add(A.on_qubits(*range(0, self.nqubitsA, 1)))
        for i in range(int(k)):
            qc=qc+Q
        qc.add(gates.M(self.nqubitsA-1))
        return qc
    def run_circuit(self,qc, shots):
        """
        Run the quantum circuit 
        Args:
            qc: quantum circuit corresponding to QAE algorithm
            shots: number of shots
        Return:
            result: counts of observing "1" for qc
        """
        result=qc(nshots=shots).frequencies(binary=True)['1']
        return result



    #CLASSICAL POSTPROCESSING FOR THE IQAE
    def clopper_pearson(self,count, n,alpha=0.05):
        """Calculates the confidence interval for the quantity to estimate a
        Args:
            count: number of successes
            nobs: total number of trials
            alpha: significance level,default 0.05. Must be in (0, 1)
        Returns 
            The confidence interval a_min,a_max
        """
        b = scipy.stats.beta.ppf
        a_min = b(alpha / 2, count, n - count + 1)
        a_max = b(1 - alpha / 2, count + 1, n - count)
        if np.isnan(a_min):
            a_min=0
        if np.isnan(a_max):
            a_max=1
        return a_min,a_max
    def calc_L_range_CP(self, N_shots,T):
        """
        Calculate the confidence interval for the Clopper-Pearson method 
        Args:
            N_shots: number of shots
            T: maximum number of rounds to achieve the desired absolute error
        Returns:
            L_max, L_min: the maximum and minimum possible error which could be returned 
            on a given iteration 
        """
        x = np.linspace(0, np.pi, 10000)
        x_domain = [(x >= 0) & (x <= 1.+1/10/N_shots)]
        alpha=self.alpha

        def h_calc_CP(N_total_plus,N_total_shots,alpha,T):
            """Calculates the h function"""
            #a_min, a_max = proportion_confint(N_total_plus, N_total_shots, method='beta', alpha=(alpha/T))
            a_min,a_max = self.clopper_pearson(N_total_plus, N_total_shots, alpha=(alpha/T))
            return np.abs(np.arccos(1-2*a_max)-np.arccos(1-2*a_min))/2
        
        y = [(lambda t: h_calc_CP(int(t*N_shots),N_shots,alpha,T))(t) for t in x[tuple(x_domain)]]
        L_max=np.max(y)/(2*np.pi)
        L_min=np.min(y)/(2*np.pi)
        return L_max,L_min
    def calc_L_range_CH(self, N_shots,T):
        """
        Calculate the confidence interval for the Chernoff method 
        Args:
            N_shots: number of shots
            T: maximum number of rounds to achieve the desired absolute error
        Returns:
            L_max, L_min: the maximum and minimum possible error which could be returned 
            on a given iteration 
        """
        L_max = np.arcsin((2/(N_shots)*np.log(2*T/self.alpha))**(1/4))/2/np.pi
        L_min = np.arcsin(np.sin(L_max)**2)
        return L_max, L_min
    def find_next_k(self,K_i, up_i, theta_l, theta_u, r=2):
        """Find the largest integer K such that the interval K*[theta_l,theta_u]
        lies completely in [0, pi] or [pi, 2pi].

        Args:
            k: the current power of the Q operator
            up_i: boolean flag of whether theta_interval lies in the
                upper half-circle [0, pi] or in the lower one [pi, 2pi]
            theta_l: the current lower limit of the confidence interval for the angle theta
            theta_u: the current upper limit of the confidence interval for the angle theta

        Returns:
            The next power k, and boolean flag for the extrapolated interval
        """
        K_max = int(1/(2*(theta_u-theta_l)))
        K = K_max - (K_max-2)%4
        while K >= r*K_i:
            theta_min = K*theta_l - int(K*theta_l)
            theta_max = K*theta_u - int(K*theta_u)
            if  theta_max <= 1/2 and theta_min <= 1/2 and int(K*theta_u) == int(K*theta_l):
                up = True
                return (K, up)
            elif theta_max >= 1/2 and theta_min >= 1/2 and int(K*theta_u) == int(K*theta_l):
                up = False
                return (K, up)
            K = K - 4
        return (K_i, up_i)
    #EXECUTING THE ALGORITHM
    def execute(self,  backend=None):
        """Execute IQAE algorithm.
        
        Args:
            backend: the qibo backend.

        Returns:
            An IterativeAmplitudeEstimation results object.
        """
        if backend is None:  
            from qibo.backends import GlobalBackend
            backend = GlobalBackend()
        #intilialize all parameters
        K = [2]  # = 4k+2 for k=0
        k=[0]
        theta_u = 1/4
        theta_l = 0
        theta_intervals=[theta_l,theta_u]
        a_intervals=[np.sin(2*np.pi*theta_l)**2,np.sin(2*np.pi*theta_u)**2]
        a_min = [0]
        a_max = [1]
        theta_dif = np.abs(theta_u-theta_l)
        up = [True]
        samples_history = []
        n_shots_history = []
        i=0

        eps = self.epsilon / np.pi
        T=int(np.log2(np.pi/(8*self.epsilon)))+1
        N_total_shots = self.N_shots
        num_oracle_queries = 0

        if self.method == 'chernoff':
            #Chernoff method
            L_max, L_min=self.calc_L_range_CH( N_total_shots,T)
            
        else:
            #Clopper-Pearson (beta) method
            if self.method=="beta":        
                L_max,L_min = self.calc_L_range_CP( N_total_shots,T)
            else:
                raise_error(
                    ValueError,
                    "The method should be either 'chernoff' or 'beta' ",
                )

        while theta_dif > 2*eps:

            i = i + 1
            K_i, up_i = self.find_next_k( K[-1], up[-1], theta_l, theta_u)
            k_i = int((K_i - 2)/4)
            K.append(K_i)
            up.append(up_i)
            k.append(k_i)
            if K_i > int(L_max/eps):
                N_shots_i = int((L_max/eps)*self.N_shots/K_i/10)
                if N_shots_i == 0:
                    N_shots_i = 1
            else:
                N_shots_i = self.N_shots

            #QUANTUM CIRCUIT
            qc = self.construct_circuit(k_i)  
            samples = self.run_circuit(qc,N_shots_i)

            samples_history.append(samples)
            n_shots_history.append(N_shots_i)

            # track number of A oracle calls
            num_oracle_queries += N_shots_i * k_i

            l=1
            if i>1:
                while K[i-l]==K[i] and i>=l+1:
                    l=l+1
                sum_total_samples = sum([samples_history[j] for j in range(i-l,i)])
                N_total_shots = sum([n_shots_history[j] for j in range(i-l,i)])

            else:
                sum_total_samples = samples
                N_total_shots = N_shots_i

            a = sum_total_samples/N_total_shots
            
            if self.method == 'chernoff':
                delta_a = np.sqrt(np.log(2*T/self.alpha)/2/N_total_shots)
                a_min_i = max(0, a - delta_a)
                a_max_i = min(1, a + delta_a)
            else:
                a_min_i, a_max_i = self.clopper_pearson(sum_total_samples, N_total_shots, alpha=self.alpha/T)  
            a_min.append(a_min_i)
            a_max.append(a_max_i)

            if up_i:
                theta_min_i = np.arccos(1-2*a_min_i)/2/np.pi
                theta_max_i = np.arccos(1-2*a_max_i)/2/np.pi
            else:
                theta_min_i = 1-np.arccos(1-2*a_max_i)/2/np.pi
                theta_max_i = 1-np.arccos(1-2*a_min_i)/2/np.pi

            theta = min(int(K_i*theta_u),int(K_i*theta_l))

            theta_u = (theta+theta_max_i)/K_i
            theta_l = (theta+theta_min_i)/K_i
            theta_dif= np.abs(theta_u-theta_l)
            theta_intervals.append([theta_l,theta_u])
            
            a_l = np.sin(2*np.pi*theta_l)**2
            a_u = np.sin(2*np.pi*theta_u)**2
            a_intervals.append([a_l, a_u])
            
        result = IterativeAmplitudeEstimationResult()
        result.alpha = self.alpha
        result.epsilon_target=self.epsilon
        result.epsilon_estimated=(a_u-a_l)/2
        result.estimate_intervals=a_intervals
        result.num_oracle_queries = num_oracle_queries
        result.estimation = (a_u+a_l)/2

        result.theta_intervals = theta_intervals
        result.k_list = k
        result.ratios = samples_history
        result.shots=n_shots_history

        return result

#THE CLASS WITH THE RESULTS
class IterativeAmplitudeEstimationResult():
    """The ``IterativeAmplitudeEstimation`` result object."""

    def __init__(self) -> None:
        super().__init__()
        self._alpha: float | None = None
        self._epsilon_target: float | None = None
        self._epsilon_estimated: float | None = None
        self._num_oracle_queries: float | None = None
        self._estimation: float | None = None
        self._estimate_intervals: list[list[float]] | None = None
        self._theta_intervals: list[list[float]] | None = None
        self._k_list: list[int] | None = None
        self._ratios: list[float] | None = None
        self._shots: list[int] | None = None