import aux_functions as aux
import matplotlib.pyplot as plt
import numpy as np

from qibo import Circuit, gates


def rw_circuit(qubits, parameters, X=True):
    """Circuit that implements the amplitude distributor part of the option pricing algorithm.
    Args:
        qubits (int): number of qubits used for the unary basis.
        paramters (list): values to be introduces into the fSim gates for amplitude distribution.
        X (bool): whether or not the first X gate is executed.

    Returns:
        generator that yield the gates needed for the amplitude distributor circuit
    """
    if qubits % 2 == 0:
        mid1 = int(qubits / 2)
        mid0 = int(mid1 - 1)
        if X:
            yield gates.X(mid1)
        yield gates.fSim(mid1, mid0, parameters[mid0] / 2, 0)
        for i in range(mid0):
            yield gates.fSim(mid0 - i, mid0 - i - 1, parameters[mid0 - i - 1] / 2, 0)
            yield gates.fSim(mid1 + i, mid1 + i + 1, parameters[mid1 + i] / 2, 0)
    else:
        mid = int((qubits - 1) / 2)
        if X:
            yield gates.X(mid)
        for i in range(mid):
            yield gates.fSim(mid - i, mid - i - 1, parameters[mid - i - 1] / 2, 0)
            yield gates.fSim(mid + i, mid + i + 1, parameters[mid + i] / 2, 0)


def rw_circuit_inv(qubits, parameters, X=True):
    """Circuit that implements the amplitude distributor part of the option pricing algorithm in reverse.
    Used in the amplitude estimation part of the algorithm.
    Args:
        qubits (int): number of qubits used for the unary basis.
        paramters (list): values to be introduces into the fSim gates for amplitude distribution.
        X (bool): whether or not the first X gate is executed.

    Returns:
        generator that yield the gates needed for the amplitude distributor circuit in reverse order.
    """
    if qubits % 2 == 0:
        mid1 = int(qubits / 2)
        mid0 = int(mid1 - 1)
        for i in range(mid0 - 1, -1, -1):
            yield gates.fSim(mid0 - i, mid0 - i - 1, -parameters[mid0 - i - 1] / 2, 0)
            yield gates.fSim(mid1 + i, mid1 + i + 1, -parameters[mid1 + i] / 2, 0)
        yield gates.fSim(mid1, mid0, -parameters[mid0] / 2, 0)
        if X:
            yield gates.X(mid1)
    else:
        mid = int((qubits - 1) / 2)
        for i in range(mid - 1, -1, -1):
            yield gates.fSim(mid + i, mid + i + 1, -parameters[mid + i] / 2, 0)
            yield gates.fSim(mid - i, mid - i - 1, -parameters[mid - i - 1] / 2, 0)
        if X:
            yield gates.X(mid)


def create_qc(qubits):
    """Creation of the quantum circuit and registers where the circuit will be implemented.
    Args:
        qubits (int): number of qubits used for the unary basis.

    Returns:
        q (list): quantum register encoding the asset's price in the unary bases.
        ancilla (int): qubit that encodes the payoff of the options.
        circuit (Circuit): quantum circuit with enough allocated space for the algorithm to run.
    """
    q = [i for i in range(qubits)]
    ancilla = qubits
    circuit = Circuit(qubits + 1)
    return q, ancilla, circuit


def rw_parameters(qubits, pdf):
    """Parameters that encode a target probability distribution into the unary basis
    Args:
        qubits (int): number of qubits used for the unary basis.
        pdf (list): known probability distribution function that wants to be reproduced.

    Returns:
        paramters (list): values to be introduces into the fSim gates for amplitude distribution.
    """
    if qubits % 2 == 0:
        mid = qubits // 2
    else:
        mid = (qubits - 1) // 2  # Important to keep track of the centre
    last = 1
    parameters = []
    for i in range(mid - 1):
        angle = 2 * np.arctan(np.sqrt(pdf[i] / (pdf[i + 1] * last)))
        parameters.append(angle)
        last = (
            np.cos(angle / 2)
        ) ** 2  # The last solution is needed to solve the next one
    angle = 2 * np.arcsin(np.sqrt(pdf[mid - 1] / last))
    parameters.append(angle)
    last = (np.cos(angle / 2)) ** 2
    for i in range(mid, qubits - 1):
        angle = 2 * np.arccos(np.sqrt(pdf[i] / last))
        parameters.append(angle)
        last *= (np.sin(angle / 2)) ** 2
    return parameters


def measure_probability(q):
    """Measuring gates on the unary basis qubits to check the validity of the amplitude distributor.
    Args:
        q (list): quantum register encoding the asset's price in the unary bases.

    Returns:
        generator that yels the measuring gates to check the probability distribution.
    """
    yield gates.M(
        *q, register_name="prob"
    )  # No measure on the ancilla qubit is necessary


def extract_probability(qubits, counts, samples):
    """Measuring gates on the unary basis qubits to check the validity of the amplitude distributor.
    Args:
        qubits (int): number of qubits used for the unary basis.
        counts (dict): times each output has been measured.
        samples (int): number of samples for normalization.

    Returns:
        prob (list): normalized probabilities for the measured outcomes.
    """
    form = "{0:0%sb}" % str(qubits)  # qubits?
    prob = []
    for i in reversed(range(qubits)):
        prob.append(counts.get(form.format(2**i), 0) / samples)
    return prob


def get_pdf(qubits, S0, sig, r, T):
    """Get a pdf to input into the quantum register from a target probability distribution.
    Args:
        qubits (int): number of qubits used for the unary basis.
        S0 (real): initial asset price.
        sig (real): market volatility.
        r (real): market rate.
        T (real): maturity time.

    Returns:
        values (np.array): price values associated to the unary basis.
        pdf (np.array): probability distribution for the asset's price evolution.
    """
    mu = (r - 0.5 * sig**2) * T + np.log(S0)
    mean = np.exp(mu + 0.5 * T * sig**2)
    variance = (np.exp(T * sig**2) - 1) * np.exp(2 * mu + T * sig**2)
    values = np.linspace(
        max(mean - 3 * np.sqrt(variance), 0), mean + 3 * np.sqrt(variance), qubits
    )
    pdf = aux.log_normal(values, mu, sig * np.sqrt(T))
    return values, pdf


def load_quantum_sim(qu, S0, sig, r, T):
    """Get a pdf to input into the quantum register from a target probability distribution.
    Args:
        qubits (int): number of qubits used for the unary basis.
        S0 (real): initial asset price.
        sig (real): market volatility.
        r (real): market rate.
        T (real): maturity time.

    Returns:
        circuit (Circuit): quantum circuit with the target probability encoded in the unary basis
        values (np.array): price values associated to the unary basis.
        pdf (np.array): probability distribution for the asset's price evolution.
    """
    (values, pdf) = get_pdf(qu, S0, sig, r, T)
    q, ancilla, circuit = create_qc(qu)
    lognormal_parameters = rw_parameters(
        qu, pdf
    )  # Solve for the parameters needed to create the target lognormal distribution
    circuit.add(
        rw_circuit(qu, lognormal_parameters)
    )  # Build the probaility loading circuit with the adjusted parameters
    circuit.add(
        measure_probability(q)
    )  # Circuit to test the precision of the probability loading algorithm
    return circuit, (values, pdf)


def run_quantum_sim(qubits, circuit, shots):
    """Execute the quantum circuit and extract the probability of measuring each state of the unary basis
    Args:
        qubits (int): number of qubits used for the unary basis.
        circuit (Circuit): quantum circuit with the target probability encoded in the unary basis.
        shots (int): number of samples to extract from the circuit.

    Returns:
        prob_sim (list): normalized probability of each possible output in the unary basis.
    """
    result = circuit(nshots=shots)
    frequencies = result.frequencies(binary=True, registers=False)

    prob_sim = extract_probability(qubits, frequencies, shots)

    return prob_sim


def payoff_circuit(qubits, ancilla, K, S):
    """Quantum circuit that encodes the expected payoff into the probability of measuring an acilla qubit.
    Args:
        qubits (int): number of qubits used for the unary basis.
        ancilla (int): qubit that encodes the payoff of the options.
        K (real): strike price.
        S (np.array): equivalent asset price for each element of the unary basis.

    Returns:
        generator that yields the gates required to encode the payoff into an ancillary qubit.
    """
    for i in range(qubits):  # Determine the first qubit's price that
        qK = i  # surpasses the strike price
        if K < S[i]:
            break
    for i in range(qK, qubits):  # Control-RY rotations controled by states
        angle = 2 * np.arcsin(
            np.sqrt((S[i] - K) / (S[qubits - 1] - K))
        )  # with higher value than the strike
        yield gates.RY(ancilla, angle).controlled_by(i)  # targeting the ancilla qubit


def payoff_circuit_inv(qubits, ancilla, K, S):
    """Quantum circuit that encodes the expected payoff into the probability of measuring an acilla qubit in reverse.
    Circuit used in the amplitude estimation part of the algorithm.
    Args:
        qubits (int): number of qubits used for the unary basis.
        ancilla (int): qubit that encodes the payoff of the options.
        K (real): strike price.
        S (np.array): equivalent asset price for each element of the unary basis.

    Returns:
        generator that yields the gates required for the inverse of the circuit used to encode
        the payoff into an ancillary qubit.
    """
    for i in range(qubits):  # Determine the first qubit's price that
        qK = i  # surpasses the strike price
        if K < S[i]:
            break
    for i in range(qK, qubits):  # Control-RY rotations controled by states
        angle = 2 * np.arcsin(
            np.sqrt((S[i] - K) / (S[qubits - 1] - K))
        )  # with higher value than the strike
        yield gates.RY(ancilla, -angle).controlled_by(i)  # targeting the ancilla qubit


def measure_payoff(q, ancilla):
    """Measurement gates needed to measure the expected payoff and perform post-selection
    Args:
        q (list): quantum register encoding the asset's price in the unary bases.
        ancilla (int): qubit that encodes the payoff of the options.

    Returns:
        generator that yields the measurement gates to recover the expected payoff.
    """
    yield gates.M(*(q + [ancilla]), register_name="payoff")


def load_payoff_quantum_sim(qubits, S0, sig, r, T, K):
    """Measurement gates needed to measure the expected payoff and perform post-selection
    Args:
        qubits (int): number of qubits used for the unary basis.
        S0 (real): initial asset price.
        sig (real): market volatility.
        r (real): market rate.
        T (real): maturity time.
        K (real): strike price.

    Returns:
        circuit (Circuit): full quantum circuit with the amplitude distributor and payoff estimator.
        S (np.array): equivalent asset price for each element of the unary basis.
    """
    mu = (r - 0.5 * sig**2) * T + np.log(S0)
    mean = np.exp(mu + 0.5 * T * sig**2)
    variance = (np.exp(T * sig**2) - 1) * np.exp(2 * mu + T * sig**2)
    S = np.linspace(
        max(mean - 3 * np.sqrt(variance), 0), mean + 3 * np.sqrt(variance), qubits
    )
    ln = aux.log_normal(S, mu, sig * np.sqrt(T))
    q, ancilla, circuit = create_qc(qubits)
    lognormal_parameters = rw_parameters(qubits, ln)
    circuit.add(rw_circuit(qubits, lognormal_parameters))
    circuit.add(payoff_circuit(qubits, ancilla, K, S))
    circuit.add(measure_payoff(q, ancilla))
    return circuit, S


def run_payoff_quantum_sim(qubits, circuit, shots, S, K):
    """Exacute the circuit that estimates the payoff of the option in the unary representation. Includes
    post-selection scheme.
    Args:
        qubits (int): number of qubits used for the unary basis.
        circuit (Circuit): full quantum circuit with the amplitude distributor and payoff estimator.
        shots (int): number of shots to be performed
        S (np.array): equivalent asset price for each element of the unary basis.
        K (real): strike price.

    Returns:
        qu_payoff_sim (real): estimated payoff from the probability of the ancillary qubit.
    """
    job_payoff_sim = circuit(nshots=shots)
    counts_payoff_sim = job_payoff_sim.frequencies(binary=True, registers=False)
    ones = 0
    zeroes = 0
    for key in counts_payoff_sim.keys():  # Post-selection
        unary = 0
        for i in range(0, qubits):
            unary += int(key[i])
        if unary == 1:
            if int(key[qubits]) == 0:
                zeroes += counts_payoff_sim.get(key)
            else:
                ones += counts_payoff_sim.get(key)
    qu_payoff_sim = ones * (S[qubits - 1] - K) / (ones + zeroes)
    return qu_payoff_sim


def diff_qu_cl(qu_payoff_sim, cl_payoff):
    """Calculation of the error from the simulated results and the classical expeted value.
    Args:
        qu_payoff_sim (real): estimated payoff from the probability of the ancillary qubit.
        cl_payoff (real): exact value computed classically.

    Returns:
        error (real): relative error between the simulated and exact result, in percentage.
    """
    error = 100 * np.abs(qu_payoff_sim - cl_payoff) / cl_payoff
    return error


def diffusion_operator(qubits):
    """Quantum circuit that performs the diffusion operator, part of the amplitude estimation algorithm.
    Args:
        qubits (int): number of qubits used for the unary basis.

    Returns:
        generator that yield the necessary gates to perform the diffusion operator.
    """
    if qubits % 2 == 0:
        mid = int(qubits / 2)
    else:
        mid = int((qubits - 1) / 2)  # The random walk starts from the middle qubit
    yield gates.X(qubits)
    yield gates.H(qubits)
    yield gates.CNOT(mid, qubits)
    yield gates.H(qubits)
    yield gates.X(qubits)


def oracle_operator(qubits):
    """Quantum circuit that performs the oracle operator, part of the amplitude estimation algorithm.
    Args:
        qubits (int): number of qubits used for the unary basis.

    Returns:
        generator that yield the necessary gates to perform the oracke operator.
    """
    yield gates.Z(qubits)


def Q(qubits, ancilla, K, S, lognormal_parameters):
    """Quantum circuit that performs the main operator for the amplitude estimation algorithm.
    Args:
        qubits (int): number of qubits used for the unary basis.
        ancilla (int): qubit that encodes the payoff of the options.
        K (real): strike price.
        S (np.array): equivalent asset price for each element of the unary basis.
        lognormal_parameters (list): values to be introduces into the fSim gates for amplitude distribution.

    Returns:
        generator that yield the necessary gates to perform the main operator for AE.
    """
    yield oracle_operator(qubits)
    yield payoff_circuit_inv(qubits, ancilla, K, S)
    yield rw_circuit_inv(qubits, lognormal_parameters, X=False)
    yield diffusion_operator(qubits)
    yield rw_circuit(qubits, lognormal_parameters, X=False)
    yield payoff_circuit(qubits, ancilla, K, S)


def load_Q_operator(qubits, iterations, S0, sig, r, T, K):
    """Quantum circuit that performs the main operator for the amplitude estimation algorithm.
    Args:
        qubits (int): number of qubits used for the unary basis.
        iterations (int): number of consecutive implementations of operator Q.
        S0 (real): initial asset price.
        sig (real): market volatility.
        r (real): market rate.
        T (real): maturity time.
        K (real): strike price.

    Returns:
        circuit (Circuit): quantum circuit that performs the m=iterations step of the iterative
            amplitude estimation algorithm.
    """
    iterations = int(iterations)
    mu = (r - 0.5 * sig**2) * T + np.log(S0)
    mean = np.exp(mu + 0.5 * T * sig**2)
    variance = (np.exp(T * sig**2) - 1) * np.exp(2 * mu + T * sig**2)
    S = np.linspace(
        max(mean - 3 * np.sqrt(variance), 0), mean + 3 * np.sqrt(variance), qubits
    )
    ln = aux.log_normal(S, mu, sig * np.sqrt(T))
    lognormal_parameters = rw_parameters(qubits, ln)
    q, ancilla, circuit = create_qc(qubits)
    circuit.add(rw_circuit(qubits, lognormal_parameters))
    circuit.add(payoff_circuit(qubits, ancilla, K, S))
    for i in range(iterations):
        circuit.add(Q(qubits, ancilla, K, S, lognormal_parameters))
    circuit.add(measure_payoff(q, ancilla))
    return circuit


def run_Q_operator(qubits, circuit, shots):
    """Execution of the quantum circuit for a step in the used amplitude estimation algorithm.
    Args:
        qubits (int): number of qubits used for the unary basis.
        circuit (Circuit): quantum circuit that performs the m=iterations step of the iterative
            amplitude estimation algorithm.
        shots (int): number of shots to be taken in intermediate steps of the AE algorithm.

    Returns:
        ones (int): number of measured ones after post-selection.
        zeroes (int): number of measured zeroes after post-selection.
    """
    job_payoff_sim = circuit(nshots=shots)
    counts_payoff_sim = job_payoff_sim.frequencies(binary=True, registers=False)
    ones = 0
    zeroes = 0
    for key in counts_payoff_sim.keys():
        unary = 0
        for i in range(0, qubits):
            unary += int(key[i])
        if unary == 1:
            if int(key[qubits]) == 0:
                zeroes += counts_payoff_sim.get(key)
            else:
                ones += counts_payoff_sim.get(key)
    return ones, zeroes


def paint_prob_distribution(bins, prob_sim, S0, sig, r, T):
    """Funtion that returns a histogram with the probabilities of the outcome measures and compares it
    with the target probability distribution.
    Args:
        bins (int): number of bins of precision.
        prob_sim (list): probabilities from measuring the quantum circuit.
        S0 (real): initial asset price.
        sig (real): market volatility.
        r (real): market rate.
        T (real): maturity time.

    Returns:
        image of the probability histogram in a .png file.
    """
    from scipy.integrate import trapz

    mu = (r - 0.5 * sig**2) * T + np.log(S0)
    mean = np.exp(mu + 0.5 * T * sig**2)
    variance = (np.exp(T * sig**2) - 1) * np.exp(2 * mu + T * sig**2)
    S = np.linspace(
        max(mean - 3 * np.sqrt(variance), 0), mean + 3 * np.sqrt(variance), bins
    )
    width = (S[1] - S[0]) / 1.2
    fig, ax = plt.subplots()
    ax.bar(S, prob_sim, width, label="Quantum", alpha=0.8)
    x = np.linspace(
        max(mean - 3 * np.sqrt(variance), 0), mean + 3 * np.sqrt(variance), bins * 100
    )
    y = aux.log_normal(x, mu, sig * np.sqrt(T))
    y = y * trapz(prob_sim, S) / trapz(y, x)
    ax.plot(x, y, label="PDF", color="black")
    plt.ylabel("Probability")
    plt.xlabel("Option price")
    plt.title(f"Option price distribution for {bins} qubits ")
    ax.legend()
    fig.tight_layout()
    fig.savefig("Probability_distribution.png")


def paint_AE(a, a_conf, bins, M, data, shots=10000, alpha=0.05):
    """Visualization of the results of applying amplitude estimation to the option pricing algorithm.
    Args:
        a (np.array): estimated values for the probability of measuring the ancilla.
        a_conf (np.array): errors on the estimation of the probability of measuring the ancilla.
        bins (int): number of bins of precision.
        M (int): total number of aplications of the Q operator.
        data (tuple): data necessary to characterize the probability distribution.
        shots (int): number of shots to be taken in intermediate steps of the AE algorithm.
        alpha (real): confidence interval.

    Returns:
        images of the results and uncertainties of performing amplitude estimation up to M times in .png format.
    """
    S0, sig, r, T, K = data
    values, pdf = get_pdf(bins, S0, sig, r, T)
    a_un = np.sum(pdf[values >= K] * (values[values >= K] - K))
    cl_payoff = aux.classical_payoff(S0, sig, r, T, K, samples=1000000)
    fig, ax = plt.subplots()
    un_data = a * (values[bins - 1] - K)
    un_conf = a_conf * (values[bins - 1] - K)
    ax.scatter(
        np.arange(M + 1), un_data, c="C0", marker="x", zorder=10, label="Measurements"
    )
    ax.fill_between(
        np.arange(M + 1), un_data - un_conf, un_data + un_conf, color="C0", alpha=0.3
    )
    ax.plot([0, M], [cl_payoff, cl_payoff], c="black", ls="--", label="Cl. payoff")
    ax.plot([0, M], [a_un, a_un], c="blue", ls="--", label="Optimal approximation")
    ax.set(ylim=[0.15, 0.17])
    ax.legend()
    fig.tight_layout()
    fig.savefig("Amplitude_Estimation_Results.png")
    from scipy.special import erfinv

    z = erfinv(1 - alpha / 2)
    fig, bx = plt.subplots()
    bx.scatter(
        np.arange(M + 1), un_conf, c="C0", marker="x", zorder=10, label="Measurements"
    )
    a_max = np.max(values) - K
    bound_down = (
        np.sqrt(un_data)
        * np.sqrt(a_max - un_data)
        * z
        / np.sqrt(shots)
        / np.cumsum(1 + 2 * (np.arange(M + 1)))
    )
    bound_up = (
        np.sqrt(un_data)
        * np.sqrt(a_max - un_data)
        * z
        / np.sqrt(shots)
        / np.sqrt(np.cumsum(1 + 2 * (np.arange(M + 1))))
    )
    bx.plot(np.arange(M + 1), bound_up, ls=":", c="C0", label="Classical sampling")
    bx.plot(
        np.arange(M + 1), bound_down, ls="-.", c="C0", label="Optimal Quantum Sampling"
    )
    bx.legend()
    bx.set(yscale="log")
    fig.tight_layout()
    fig.savefig("Amplitude_Estimation_Uncertainties.png")


def amplitude_estimation(bins, M, data, shots=10000):
    """Execution of the quantum circuit for a step in the used amplitude estimation algorithm.
    Args:
        bins (int): number of bins of precision.
        M (int): total number of aplications of the Q operator.
        data (tuple): data necessary to characterize the probability distribution.
        shots (int): number of shots to be taken in intermediate steps of the AE algorithm.

    Returns:
        a_s (np.array): estimated values for the probability of measuring the ancilla.
        error_s (np.array): errors on the estimation of the probability of measuring the ancilla.
    """
    S0, sig, r, T, K = data
    circuit, S = load_payoff_quantum_sim(bins, S0, sig, r, T, K)
    qu_payoff_sim = run_payoff_quantum_sim(bins, circuit, shots, S, K)
    m_s = np.arange(0, M + 1, 1)
    circuits = []
    for j, m in enumerate(m_s):
        qc = load_Q_operator(bins, m, S0, sig, r, T, K)
        circuits.append(qc)
    ones_s = []
    zeroes_s = []
    for j, m in enumerate(m_s):
        ones, zeroes = run_Q_operator(bins, circuits[j], shots)
        ones_s.append(ones)
        zeroes_s.append(zeroes)
    theta_max_s, error_theta_s = aux.get_theta(m_s, ones_s, zeroes_s)
    a_s, error_s = np.sin(theta_max_s) ** 2, np.abs(
        np.sin(2 * theta_max_s) * error_theta_s
    )
    return a_s, error_s
