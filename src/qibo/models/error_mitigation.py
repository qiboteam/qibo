"""Error Mitigation Methods."""

from itertools import product

import numpy as np
from scipy.optimize import curve_fit

from qibo import gates
from qibo.backends import GlobalBackend, _check_backend
from qibo.config import raise_error


def get_gammas(noise_levels, analytical: bool = True):
    """Standalone function to compute the ZNE coefficients given the noise levels.

    Args:
        noise_levels (numpy.ndarray): array containing the different noise levels.
            Note that in the CNOT insertion paradigm this corresponds to
            the number of CNOT pairs to be inserted. The canonical ZNE
            noise levels are obtained as ``2 * c + 1``.
        analytical (bool, optional): if ``True``, computes the coeffients by solving the
            linear system. If ``False``, use the analytical solution valid
            for the CNOT insertion method. Default is ``True``.

    Returns:
        numpy.ndarray: the computed coefficients.
    """
    if analytical:
        noise_levels = 2 * noise_levels + 1
        a_matrix = np.array([noise_levels**i for i in range(len(noise_levels))])
        b_vector = np.zeros(len(noise_levels))
        b_vector[0] = 1
        zne_coefficients = np.linalg.solve(a_matrix, b_vector)
    else:
        max_noise_level = noise_levels[-1]
        zne_coefficients = np.array(
            [
                1
                / (2 ** (2 * max_noise_level) * np.math.factorial(i))
                * (-1) ** i
                / (1 + 2 * i)
                * np.math.factorial(1 + 2 * max_noise_level)
                / (
                    np.math.factorial(max_noise_level)
                    * np.math.factorial(max_noise_level - i)
                )
                for i in noise_levels
            ]
        )

    return zne_coefficients


def get_noisy_circuit(circuit, num_insertions: int, insertion_gate: str = "CNOT"):
    """Standalone function to generate the noisy circuit with the inverse gate pairs insertions.

    Args:
        circuit (:class:`qibo.models.circuit.Circuit`): circuit to modify.
        num_insertions (int): number of insertion gate pairs to add.
        insertion_gate (str, optional): gate to be used in the insertion.
            If ``"RX"``, the gate used is :math:``RX(\\pi / 2)``.
            Default is ``"CNOT"``.

    Returns:
        :class:`qibo.models.Circuit`: circuit with the inserted gate pairs.
    """
    if insertion_gate not in ("CNOT", "RX"):  # pragma: no cover
        raise_error(
            ValueError,
            "Invalid insertion gate specification. Please select between 'CNOT' and 'RX'.",
        )
    if insertion_gate == "CNOT" and circuit.nqubits < 2:  # pragma: no cover
        raise_error(
            ValueError,
            "Provide a circuit with at least 2 qubits when using the 'CNOT' insertion gate. "
            + "Alternatively, try with the 'RX' insertion gate instead.",
        )

    i_gate = gates.CNOT if insertion_gate == "CNOT" else gates.RX

    theta = np.pi / 2
    noisy_circuit = circuit.__class__(**circuit.init_kwargs)

    for gate in circuit.queue:
        noisy_circuit.add(gate)

        if isinstance(gate, i_gate):
            if insertion_gate == "CNOT":
                control = gate.control_qubits[0]
                target = gate.target_qubits[0]
                for _ in range(num_insertions):
                    noisy_circuit.add(gates.CNOT(control, target))
                    noisy_circuit.add(gates.CNOT(control, target))
            elif gate.init_kwargs["theta"] == theta:
                qubit = gate.qubits[0]
                for _ in range(num_insertions):
                    noisy_circuit.add(gates.RX(qubit, theta=theta))
                    noisy_circuit.add(gates.RX(qubit, theta=-theta))

    return noisy_circuit


def ZNE(
    circuit,
    observable,
    noise_levels,
    noise_model=None,
    nshots=10000,
    solve_for_gammas=False,
    insertion_gate="CNOT",
    readout=None,
    qubit_map=None,
    backend=None,
):
    """Runs the Zero Noise Extrapolation method for error mitigation.

    The different noise levels are realized by the insertion of pairs of
    either ``CNOT`` or ``RX(pi/2)`` gates that resolve to the identiy in
    the noise-free case.

    Args:
        circuit (:class:`qibo.models.Circuit`): input circuit.
        observable (:class:`qibo.hamiltonians.Hamiltonian/:class:`qibo.hamiltonians.SymbolicHamiltonian`): Observable to measure.
        noise_levels (numpy.ndarray): Sequence of noise levels.
        noise_model (:class:`qibo.noise.NoiseModel`, optional): Noise model applied
            to simulate noisy computation.
        nshots (int, optional): Number of shots. Defaults to :math:`10000`.
        solve_for_gammas (bool, optional): If ``True``, explicitly solve the
            equations to obtain the ``gamma`` coefficients. Default is ``False``.
        insertion_gate (str, optional): gate to be used in the insertion.
            If ``"RX"``, the gate used is :math:``RX(\\pi / 2)``.
            Defaults to ``"CNOT"``.
        readout (dict, optional): a dictionary that may contain the following keys:

            *    ncircuits: int, specifies the number of random circuits to use for the randomized method of readout error mitigation.
            *    response_matrix: numpy.ndarray, used for applying a pre-computed response matrix for readout error mitigation.
            *    ibu_iters: int, specifies the number of iterations for the iterative Bayesian unfolding method of readout error mitigation. If provided, the corresponding readout error mitigation method is used. Defaults to {}.

        qubit_map (list, optional): the qubit map. If None, a list of range of circuit's qubits is used. Defaults to ``None``.
        backend (:class:`qibo.backends.abstract.Backend`, optional): backend to be used
            in the execution. If ``None``, it uses :class:`qibo.backends.GlobalBackend`.
            Defaults to ``None``.

    Returns:
        numpy.ndarray: Estimate of the expected value of ``observable`` in the noise free condition.

    Reference:
        1. K. Temme, S. Bravyi et al, *Error mitigation for short-depth quantum circuits*.
           `arXiv:1612.02058 [quant-ph] <https://arxiv.org/abs/1612.02058>`_.
    """
    backend = _check_backend(backend)

    if readout is None:
        readout = {}

    expected_values = []
    for num_insertions in noise_levels:
        noisy_circuit = get_noisy_circuit(
            circuit, num_insertions, insertion_gate=insertion_gate
        )
        val = get_expectation_val_with_readout_mitigation(
            noisy_circuit,
            observable,
            noise_model,
            nshots,
            readout,
            qubit_map,
            backend=backend,
        )
        expected_values.append(val)

    gamma = get_gammas(noise_levels, analytical=solve_for_gammas)

    return np.sum(gamma * expected_values)


def sample_training_circuit_cdr(
    circuit,
    replacement_gates: list = None,
    sigma: float = 0.5,
    backend=None,
):
    """Samples a training circuit for CDR by susbtituting some of the non-Clifford gates.

    Args:
        circuit (:class:`qibo.models.Circuit`): circuit to sample from,
            decomposed in ``RX(pi/2)``, ``X``, ``CNOT`` and ``RZ`` gates.
        replacement_gates (list, optional): candidates for the substitution of the
            non-Clifford gates. The ``list`` should be composed by ``tuples`` of the
            form (``gates.XYZ``, ``kwargs``). For example, phase gates are used by default:
            ``list((RZ, {'theta':0}), (RZ, {'theta':pi/2}), (RZ, {'theta':pi}), (RZ, {'theta':3*pi/2}))``.
        sigma (float, optional): standard devation of the Gaussian distribution used for sampling.
        backend (:class:`qibo.backends.abstract.Backend`, optional): backend to be used
            in the execution. If ``None``, it uses :class:`qibo.backends.GlobalBackend`.
            Defaults to ``None``.

    Returns:
        :class:`qibo.models.Circuit`: The sampled circuit.
    """
    backend = _check_backend(backend)

    if replacement_gates is None:
        replacement_gates = [(gates.RZ, {"theta": n * np.pi / 2}) for n in range(4)]

    gates_to_replace = []
    for i, gate in enumerate(circuit.queue):
        if isinstance(gate, gates.RZ):
            if gate.init_kwargs["theta"] % (np.pi / 2) != 0.0:
                gates_to_replace.append((i, gate))

    if not gates_to_replace:
        raise_error(ValueError, "No non-Clifford RZ gate found, no circuit sampled.")

    replacement, distance = [], []
    for _, gate in gates_to_replace:
        rep_gates = np.array(
            [rg(*gate.init_args, **kwargs) for rg, kwargs in replacement_gates]
        )

        replacement.append(rep_gates)
        distance.append(
            np.linalg.norm(
                gate.matrix(backend)
                - [rep_gate.matrix(backend) for rep_gate in rep_gates],
                ord="fro",
                axis=(1, 2),
            )
        )

    distance = np.vstack(distance)
    prob = np.exp(-(distance**2) / sigma**2)

    index = np.random.choice(
        range(len(gates_to_replace)),
        size=min(int(len(gates_to_replace) / 2), 50),
        replace=False,
        p=prob.sum(-1) / prob.sum(),
    )

    gates_to_replace = np.array([gates_to_replace[i] for i in index])
    prob = [prob[i] for i in index]

    replacement = np.array([replacement[i] for i in index])
    replacement = [
        replacement[i][np.random.choice(range(len(p)), size=1, p=p / p.sum())[0]]
        for i, p in enumerate(prob)
    ]
    replacement = {i[0]: g for i, g in zip(gates_to_replace, replacement)}

    sampled_circuit = circuit.__class__(**circuit.init_kwargs)
    for i, gate in enumerate(circuit.queue):
        sampled_circuit.add(replacement.get(i, gate))

    return sampled_circuit


def CDR(
    circuit,
    observable,
    noise_model,
    nshots: int = 10000,
    model=lambda x, a, b: a * x + b,
    n_training_samples: int = 100,
    full_output: bool = False,
    readout=None,
    qubit_map=None,
    backend=None,
):
    """Runs the Clifford Data Regression error mitigation method.

    Args:
        circuit (:class:`qibo.models.Circuit`): input circuit decomposed in the
            primitive gates ``X``, ``CNOT``, ``RX(pi/2)``, ``RZ(theta)``.
        observable (:class:`qibo.hamiltonians.Hamiltonian/:class:`qibo.hamiltonians.SymbolicHamiltonian`): observable to be measured.
        noise_model (:class:`qibo.noise.NoiseModel`): noise model used for simulating
            noisy computation.
        nshots (int, optional): number of shots. Defaults :math:`10000`.
        model (callable, optional): model used for fitting. This should be a callable
            function object ``f(x, *params)``, taking as input the predictor variable
            and the parameters. Default is a simple linear model ``f(x,a,b) := a*x + b``.
        n_training_samples (int, optional): number of training circuits to sample. Defaults to 100.
        full_output (bool, optional): if ``True``, this function returns additional
            information: ``val``, ``optimal_params``, ``train_val``. Defaults to ``False``.
        readout (dict, optional): a dictionary that may contain the following keys:

            *    ncircuits: int, specifies the number of random circuits to use for the randomized method of readout error mitigation.
            *    response_matrix: numpy.ndarray, used for applying a pre-computed response matrix for readout error mitigation.
            *    ibu_iters: int, specifies the number of iterations for the iterative Bayesian unfolding method of readout error mitigation. If provided, the corresponding readout error mitigation method is used. Defaults to {}.

        qubit_map (list, optional): the qubit map. If None, a list of range of circuit's qubits is used. Defaults to ``None``.
        backend (:class:`qibo.backends.abstract.Backend`, optional): backend to be used
            in the execution. If ``None``, it uses :class:`qibo.backends.GlobalBackend`.
            Defaults to ``None``.

    Returns:
        mit_val (float): Mitigated expectation value of `observable`.
        val (float): Noisy expectation value of `observable`.
        optimal_params (list): Optimal values for `params`.
        train_val (dict): Contains the noise-free and noisy expectation values obtained with the training circuits.

    Reference:
        1. P. Czarnik, A. Arrasmith et al, *Error mitigation with Clifford quantum-circuit data*.
           `arXiv:2005.10189 [quant-ph] <https://arxiv.org/abs/2005.10189>`_.
    """
    backend = _check_backend(backend)

    if readout is None:
        readout = {}

    training_circuits = [
        sample_training_circuit_cdr(circuit) for _ in range(n_training_samples)
    ]

    train_val = {"noise-free": [], "noisy": []}
    for circ in training_circuits:
        result = backend.execute_circuit(circ, nshots=nshots)
        val = result.expectation_from_samples(observable)
        train_val["noise-free"].append(val)
        val = get_expectation_val_with_readout_mitigation(
            circ, observable, noise_model, nshots, readout, qubit_map, backend=backend
        )
        train_val["noisy"].append(val)

    optimal_params = curve_fit(model, train_val["noisy"], train_val["noise-free"])[0]

    val = get_expectation_val_with_readout_mitigation(
        circuit, observable, noise_model, nshots, readout, qubit_map, backend=backend
    )
    mit_val = model(val, *optimal_params)

    if full_output is True:
        return mit_val, val, optimal_params, train_val

    return mit_val


def vnCDR(
    circuit,
    observable,
    noise_levels,
    noise_model,
    nshots: int = 10000,
    model=lambda x, *params: (x * np.array(params).reshape(-1, 1)).sum(0),
    n_training_samples: int = 100,
    insertion_gate: str = "CNOT",
    full_output: bool = False,
    readout=None,
    qubit_map=None,
    backend=None,
):
    """Runs the variable-noise Clifford Data Regression error mitigation method.

    Args:
        circuit (:class:`qibo.models.Circuit`): input circuit decomposed in the
            primitive gates ``X``, ``CNOT``, ``RX(pi/2)``, ``RZ(theta)``.
        observable (:class:`qibo.hamiltonians.Hamiltonian/:class:`qibo.hamiltonians.SymbolicHamiltonian`): observable to be measured.
        noise_levels (numpy.ndarray): sequence of noise levels.
        noise_model (:class:`qibo.noise.NoiseModel`): noise model used for
            simulating noisy computation.
        nshots (int, optional): number of shots. Defaults to :math:`10000`.
        model (callable, optional): model used for fitting. This should be a callable
            function object ``f(x, *params)``, taking as input the predictor variable
            and the parameters. Default is a simple linear model ``f(x,a,b) := a*x + b``.
        n_training_samples (int, optional): number of training circuits to sample.
        insertion_gate (str, optional): gate to be used in the insertion.
            If ``"RX"``, the gate used is :math:``RX(\\pi / 2)``.
            Default is ``"CNOT"``.
        full_output (bool, optional): if ``True``, this function returns additional
            information: ``val``, ``optimal_params``, ``train_val``. Defaults to ``False``.
        readout (dict, optional): a dictionary that may contain the following keys:

            *    ncircuits: int, specifies the number of random circuits to use for the randomized method of readout error mitigation.
            *    response_matrix: numpy.ndarray, used for applying a pre-computed response matrix for readout error mitigation.
            *    ibu_iters: int, specifies the number of iterations for the iterative Bayesian unfolding method of readout error mitigation. If provided, the corresponding readout error mitigation method is used. Defaults to {}.

        qubit_map (list, optional): the qubit map. If None, a list of range of circuit's qubits is used. Defaults to ``None``.
        backend (:class:`qibo.backends.abstract.Backend`, optional): backend to be used
            in the execution. If ``None``, it uses :class:`qibo.backends.GlobalBackend`.
            Defaults to ``None``.

    Returns:
        mit_val (float): Mitigated expectation value of `observable`.
        val (list): Expectation value of `observable` with increased noise levels.
        optimal_params (list): Optimal values for `params`.
        train_val (dict): Contains the noise-free and noisy expectation values obtained
        with the training circuits.

    Reference:
        1. A. Lowe, MH. Gordon et al, *Unified approach to data-driven quantum error mitigation*.
           `arXiv:2011.01157 [quant-ph] <https://arxiv.org/abs/2011.01157>`_.
    """
    backend = _check_backend(backend)

    if readout is None:
        readout = {}

    training_circuits = [
        sample_training_circuit_cdr(circuit) for _ in range(n_training_samples)
    ]
    train_val = {"noise-free": [], "noisy": []}

    for circ in training_circuits:
        result = backend.execute_circuit(circ, nshots=nshots)
        val = result.expectation_from_samples(observable)
        train_val["noise-free"].append(val)
        for level in noise_levels:
            noisy_c = get_noisy_circuit(circ, level, insertion_gate=insertion_gate)
            val = get_expectation_val_with_readout_mitigation(
                noisy_c,
                observable,
                noise_model,
                nshots,
                readout,
                qubit_map,
                backend=backend,
            )
            train_val["noisy"].append(val)

    noisy_array = np.array(train_val["noisy"]).reshape(-1, len(noise_levels))

    params = np.random.rand(len(noise_levels))
    optimal_params = curve_fit(model, noisy_array.T, train_val["noise-free"], p0=params)

    val = []
    for level in noise_levels:
        noisy_c = get_noisy_circuit(circuit, level, insertion_gate=insertion_gate)
        expval = get_expectation_val_with_readout_mitigation(
            noisy_c,
            observable,
            noise_model,
            nshots,
            readout,
            qubit_map,
            backend=backend,
        )
        val.append(expval)

    mit_val = model(np.array(val).reshape(-1, 1), *optimal_params[0])[0]

    if full_output is True:
        return mit_val, val, optimal_params, train_val

    return mit_val


def iterative_bayesian_unfolding(probabilities, response_matrix, iterations=10):
    """
    Iterative Bayesian Unfolding (IBU) method for readout mitigation.

    Args:
        probabilities (numpy.ndarray): the input probabilities to be unfolded.
        response_matrix (numpy.ndarray): the response matrix.
        iterations (int, optional): the number of iterations to perform. Defaults to 10.

    Returns:
        numpy.ndarray: the unfolded probabilities.

    Reference:
        1. B. Nachman, M. Urbanek et al, *Unfolding Quantum Computer Readout Noise*.
           `arXiv:1910.01969 [quant-ph] <https://arxiv.org/abs/1910.01969>`_.
        2. S. Srinivasan, B. Pokharel et al, *Scalable Measurement Error Mitigation via Iterative Bayesian Unfolding*.
           `arXiv:2210.12284 [quant-ph] <https://arxiv.org/abs/2210.12284>`_.
    """
    unfolded_probabilities = np.ones((len(probabilities), 1)) / len(probabilities)

    for _ in range(iterations):
        unfolded_probabilities = unfolded_probabilities * (
            np.transpose(response_matrix)
            @ (probabilities / (response_matrix @ unfolded_probabilities))
        )

    return unfolded_probabilities


def get_response_matrix(
    nqubits, qubit_map=None, noise_model=None, nshots: int = 10000, backend=None
):
    """Computes the response matrix for readout mitigation.

    Args:
        nqubits (int): Total number of qubits.
        qubit_map (list, optional): the qubit map. If None, a list of range of circuit's qubits is used. Defaults to ``None``.
        noise_model (:class:`qibo.noise.NoiseModel`, optional): noise model used for simulating
            noisy computation. This matrix can be used to mitigate the effect of
            `qibo.noise.ReadoutError`.
        nshots (int, optional): number of shots. Defaults to :math:`10000`.
        backend (:class:`qibo.backends.abstract.Backend`, optional): backend to be used
            in the execution. If ``None``, it uses :class:`qibo.backends.GlobalBackend`.
            Defaults to ``None``.

    Returns:
        numpy.ndarray : the computed (`nqubits`, `nqubits`) response matrix for
            readout mitigation.
    """
    from qibo import Circuit  # pylint: disable=import-outside-toplevel

    backend = _check_backend(backend)

    response_matrix = np.zeros((2**nqubits, 2**nqubits))

    for i in range(2**nqubits):
        binary_state = format(i, f"0{nqubits}b")

        circuit = Circuit(nqubits, density_matrix=True)
        for qubit, bit in enumerate(binary_state):
            if bit == "1":
                circuit.add(gates.X(qubit))
        circuit.add(gates.M(*range(nqubits)))

        circuit_result = _execute_circuit(
            circuit, qubit_map, noise_model, nshots, backend=backend
        )

        frequencies = circuit_result.frequencies()

        column = np.zeros(2**nqubits)
        for key, value in frequencies.items():
            column[int(key, 2)] = value / nshots
        response_matrix[:, i] = column

    return response_matrix


def apply_resp_mat_readout_mitigation(state, response_matrix, iterations=None):
    """
    Applies readout error mitigation to the given state using the provided response matrix.

    Args:
        state (:class:`qibo.measurements.CircuitResult`): the input state to be updated. This state should contain the
            frequencies that need to be mitigated.
        response_matrix (numpy.ndarray): the response matrix for readout mitigation.
        iterations (int, optional): the number of iterations to use for the Iterative Bayesian Unfolding method.
            If ``None`` the 'inverse' method is used. Defaults to ``None``.

    Returns:
        :class:`qibo.measurements.CircuitResult`: the input state with the updated (mitigated) frequencies.
    """
    frequencies = np.zeros(2 ** len(state.measurements[0].qubits))
    for key, value in state.frequencies().items():
        frequencies[int(key, 2)] = value

    frequencies = frequencies.reshape(-1, 1)

    if iterations is None:
        calibration_matrix = np.linalg.inv(response_matrix)
        mitigated_frequencies = calibration_matrix @ frequencies
    else:
        mitigated_probabilities = iterative_bayesian_unfolding(
            frequencies / np.sum(frequencies), response_matrix, iterations
        )
        mitigated_frequencies = np.round(
            mitigated_probabilities * np.sum(frequencies), 0
        )
        mitigated_frequencies = (
            mitigated_frequencies / np.sum(mitigated_frequencies)
        ) * np.sum(frequencies)

    for i, value in enumerate(mitigated_frequencies):
        state._frequencies[i] = float(value)

    return state


def apply_randomized_readout_mitigation(
    circuit,
    noise_model=None,
    nshots: int = 10000,
    ncircuits: int = 10,
    qubit_map=None,
    backend=None,
):
    """Readout mitigation method that transforms the bias in an expectation value into a measurable multiplicative factor. This factor can be eliminated at the expense of increased sampling complexity for the observable.

    Args:
        circuit (:class:`qibo.models.Circuit`): input circuit.
        noise_model(:class:`qibo.noise.NoiseModel`, optional): noise model used for
            simulating noisy computation. Defaults to ``None``.
        nshots (int, optional): number of shots. Defaults to :math:`10000`.
        ncircuits (int, optional): number of randomized circuits. Each of them uses
            ``int(nshots / ncircuits)`` shots. Defaults to 10.
        qubit_map (list, optional): the qubit map. If None, a list of range of circuit's qubits is used. Defaults to ``None``.
        backend (:class:`qibo.backends.abstract.Backend`, optional): backend to be used
            in the execution. If ``None``, it uses :class:`qibo.backends.GlobalBackend`.
            Defaults to ``None``.

    Return:
        :class:`qibo.measurements.CircuitResult`: the state of the input circuit with
            mitigated frequencies.


    Reference:
        1. Ewout van den Berg, Zlatko K. Minev et al, *Model-free readout-error mitigation for quantum expectation values*.
           `arXiv:2012.09738 [quant-ph] <https://arxiv.org/abs/2012.09738>`_.
    """
    from qibo import Circuit  # pylint: disable=import-outside-toplevel
    from qibo.quantum_info import (  # pylint: disable=import-outside-toplevel
        random_pauli,
    )

    backend = _check_backend(backend)

    meas_qubits = circuit.measurements[0].qubits
    nshots_r = int(nshots / ncircuits)
    freq = np.zeros((ncircuits, 2), object)
    for k in range(ncircuits):
        circuit_c = circuit.copy(True)
        circuit_c.queue.pop()
        cal_circuit = Circuit(circuit.nqubits, density_matrix=True)

        x_gate = random_pauli(circuit.nqubits, 1, subset=["I", "X"]).queue

        error_map = {}
        for j, gate in enumerate(x_gate):
            if isinstance(gate, gates.X) and gate.qubits[0] in meas_qubits:
                error_map[gate.qubits[0]] = 1

        circuits = [circuit_c, cal_circuit]
        results = []
        freqs = []
        for circ in circuits:
            circ.add(x_gate)
            circ.add(gates.M(*meas_qubits))

            result = _execute_circuit(
                circ, qubit_map, noise_model, nshots_r, backend=backend
            )
            result._samples = result.apply_bitflips(error_map)
            results.append(result)
            freqs.append(result.frequencies(binary=False))
        freq[k, :] = freqs

    for j in range(2):
        results[j].nshots = nshots
        freq_sum = freq[0, j]
        for frs in freq[1::, j]:
            freq_sum += frs
        results[j]._frequencies = freq_sum

    return results


def get_expectation_val_with_readout_mitigation(
    circuit,
    observable,
    noise_model=None,
    nshots: int = 10000,
    readout=None,
    qubit_map=None,
    backend=None,
):
    """
    Applies readout error mitigation to the given circuit and observable.

    Args:
        circuit (qibo.models.Circuit): input circuit.
        observable (:class:`qibo.hamiltonians.Hamiltonian/:class:`qibo.hamiltonians.SymbolicHamiltonian`): The observable to be measured.
        noise_model (qibo.models.noise.Noise, optional): the noise model to be applied. Defaults to ``None``.
        nshots (int, optional): the number of shots for the circuit execution. Defaults to :math:`10000`.
        readout (dict, optional): a dictionary that may contain the following keys:

            *    ncircuits: int, specifies the number of random circuits to use for the randomized method of readout error mitigation.
            *    response_matrix: numpy.ndarray, used for applying a pre-computed response matrix for readout error mitigation.
            *    ibu_iters: int, specifies the number of iterations for the iterative Bayesian unfolding method of readout error mitigation. If provided, the corresponding readout error mitigation method is used. Defaults to {}.

        qubit_map (list, optional): the qubit map. If None, a list of range of circuit's qubits is used. Defaults to ``None``.
        backend (qibo.backends.abstract.Backend, optional): the backend to be used in the execution.
            If None, it uses the global backend. Defaults to ``None``.

    Returns:
        float: the mitigated expectation value of the observable.
    """
    backend = _check_backend(backend)

    if readout is None:  # pragma: no cover
        readout = {}

    if "ncircuits" in readout:
        circuit_result, circuit_result_cal = apply_randomized_readout_mitigation(
            circuit, noise_model, nshots, readout["ncircuits"], backend
        )
    else:
        circuit_result = _execute_circuit(
            circuit, qubit_map, noise_model, nshots, backend=backend
        )
        if "response_matrix" in readout:
            circuit_result = apply_resp_mat_readout_mitigation(
                circuit_result,
                readout["response_matrix"],
                readout.get("ibu_iters", None),
            )

    exp_val = circuit_result.expectation_from_samples(observable)

    if "ncircuits" in readout:
        exp_val /= circuit_result_cal.expectation_from_samples(observable)

    return exp_val


def sample_clifford_training_circuit(
    circuit,
    backend=None,
):
    """Samples a training circuit for CDR by susbtituting all the non-Clifford gates.

    Args:
        circuit (:class:`qibo.models.Circuit`): circuit to sample from.
        backend (:class:`qibo.backends.abstract.Backend`, optional): backend to be used
            in the execution. If ``None``, it uses :class:`qibo.backends.GlobalBackend`.
            Defaults to ``None``.

    Returns:
        :class:`qibo.models.Circuit`: the sampled circuit.
    """
    from qibo.quantum_info import (  # pylint: disable=import-outside-toplevel
        random_clifford,
    )

    backend = _check_backend(backend)

    non_clifford_gates_indices = [
        i
        for i, gate in enumerate(circuit.queue)
        if not gate.clifford and not isinstance(gate, gates.M)
    ]

    if not non_clifford_gates_indices:
        raise_error(ValueError, "No non-Clifford gate found, no circuit sampled.")

    sampled_circuit = circuit.__class__(**circuit.init_kwargs)

    for i, gate in enumerate(circuit.queue):
        if isinstance(gate, gates.M):
            for q in gate.qubits:
                gate_rand = gates.Unitary(
                    random_clifford(1, backend=backend, return_circuit=False),
                    q,
                )
                gate_rand.clifford = True
                sampled_circuit.add(gate_rand)
            sampled_circuit.add(gate)
        else:
            if i in non_clifford_gates_indices:
                gate = gates.Unitary(
                    random_clifford(
                        len(gate.qubits), backend=backend, return_circuit=False
                    ),
                    *gate.qubits,
                )
                gate.clifford = True
            sampled_circuit.add(gate)

    return sampled_circuit


def error_sensitive_circuit(circuit, observable, backend=None):
    """
    Generates a Clifford circuit that preserves the same circuit frame as the input circuit, and stabilizes the specified Pauli observable.

    Args:
        circuit (:class:`qibo.models.Circuit`): input circuit.
        observable (:class:`qibo.hamiltonians.Hamiltonian/:class:`qibo.hamiltonians.SymbolicHamiltonian`): Pauli observable to be measured.
        backend (:class:`qibo.backends.abstract.Backend`, optional): backend to be used
            in the execution. if ``None``, it uses :class:`qibo.backends.GlobalBackend`.
            Defaults to ``None``.

    Returns:
        :class:`qibo.models.Circuit`: the error sensitive circuit.
        :class:`qibo.models.Circuit`: the sampled Clifford circuit.
        list: the list of adjustment gates.

    Reference:
        1. Dayue Qin, Yanzhu Chen et al, *Error statistics and scalability of quantum error mitigation formulas*.
           `arXiv:2112.06255 [quant-ph] <https://arxiv.org/abs/2112.06255>`_.
    """
    from qibo.quantum_info import (  # pylint: disable=import-outside-toplevel
        comp_basis_to_pauli,
        random_clifford,
        vectorization,
    )

    backend = _check_backend(backend)

    sampled_circuit = sample_clifford_training_circuit(circuit, backend=backend)
    unitary_matrix = sampled_circuit.unitary(backend=backend)
    num_qubits = sampled_circuit.nqubits

    comp_to_pauli = comp_basis_to_pauli(num_qubits, backend=backend)
    observable.nqubits = num_qubits
    observable_liouville = vectorization(
        np.transpose(np.conjugate(unitary_matrix)) @ observable.matrix @ unitary_matrix,
        order="row",
        backend=backend,
    )
    observable_pauli_liouville = comp_to_pauli @ observable_liouville

    index = int(np.where(abs(observable_pauli_liouville) >= 1e-5)[0][0])

    observable_pauli = list(product(["I", "X", "Y", "Z"], repeat=num_qubits))[index]

    pauli_gates = {
        "I": backend.cast(gates.I(0).matrix(backend=backend)),
        "X": backend.cast(gates.X(0).matrix(backend=backend)),
        "Y": backend.cast(gates.Y(0).matrix(backend=backend)),
        "Z": backend.cast(gates.Z(0).matrix(backend=backend)),
    }

    adjustment_gates = []
    for i in range(num_qubits):
        observable_i = pauli_gates[observable_pauli[i]]
        random_init = pauli_gates["I"]
        while np.any(abs(observable_i - pauli_gates["Z"]) > 1e-5) and np.any(
            abs(observable_i - pauli_gates["I"]) > 1e-5
        ):
            random_init = random_clifford(1, backend=backend, return_circuit=False)
            observable_i = (
                np.conjugate(np.transpose(random_init))
                @ pauli_gates[observable_pauli[i]]
                @ random_init
            )

        adjustment_gate = gates.Unitary(random_init, i)
        adjustment_gate.clifford = True
        adjustment_gates.append(adjustment_gate)

    sensitive_circuit = sampled_circuit.__class__(**sampled_circuit.init_kwargs)

    for gate in adjustment_gates:
        sensitive_circuit.add(gate)
    for gate in sampled_circuit.queue:
        sensitive_circuit.add(gate)

    return sensitive_circuit, sampled_circuit, adjustment_gates


def ICS(
    circuit,
    observable,
    readout=None,
    qubit_map=None,
    noise_model=None,
    nshots=int(1e4),
    n_training_samples=10,
    full_output=False,
    backend=None,
):
    """
    Computes the Important Clifford Sampling method.

    Args:
        circuit (:class:`qibo.models.Circuit`): input circuit.
        observable (:class:`qibo.hamiltonians.Hamiltonian/:class:`qibo.hamiltonians.SymbolicHamiltonian`): the observable to be measured.
        readout (dict, optional): a dictionary that may contain the following keys:

            *    ncircuits: int, specifies the number of random circuits to use for the randomized method of readout error mitigation.
            *    response_matrix: numpy.ndarray, used for applying a pre-computed response matrix for readout error mitigation.
            *    ibu_iters: int, specifies the number of iterations for the iterative Bayesian unfolding method of readout error mitigation. If provided, the corresponding readout error mitigation method is used. Defaults to {}.

        qubit_map (list, optional): the qubit map. If ``None``, a list of range of circuit's qubits is used. Defaults to ``None``.
        noise_model (qibo.models.noise.Noise, optional): the noise model to be applied. Defaults to ``None``.
        nshots (int, optional): the number of shots for the circuit execution. Defaults to :math:`10000`.
        n_training_samples (int, optional): the number of training samples. Defaults to 10.
        full_output (bool, optional): if ``True``, this function returns additional
            information: ``val``, ``optimal_params``, ``train_val``. Defaults to ``False``.
        backend (qibo.backends.abstract.Backend, optional): the backend to be used in the execution.
            If None, it uses the global backend. Defaults to ``None``.

    Returns:
        mitigated_expectation (float): the mitigated expectated value.
        mitigated_expectation_std (float): the standard deviation of the mitigated expectated value.
        dep_param (float): the depolarizing parameter.
        dep_param_std (float): the standard deviation of the depolarizing parameter.
        lambda_list (list): the list of the depolarizing parameters.
        data (dict): the data dictionary containing the noise-free and noisy expectation values obtained with the training circuits.

    Reference:
        1. Dayue Qin, Yanzhu Chen et al, *Error statistics and scalability of quantum error mitigation formulas*.
           `arXiv:2112.06255 [quant-ph] <https://arxiv.org/abs/2112.06255>`_.
    """
    backend = _check_backend(backend)

    if readout is None:
        readout = {}

    if qubit_map is None:
        qubit_map = list(range(circuit.nqubits))

    training_circuits = [
        error_sensitive_circuit(circuit, observable, backend=backend)[0]
        for _ in range(n_training_samples)
    ]

    data = {"noise-free": [], "noisy": []}
    lambda_list = []

    for training_circuit in training_circuits:
        circuit_result = backend.execute_circuit(training_circuit, nshots=nshots)
        expectation = observable.expectation_from_samples(circuit_result.frequencies())

        noisy_expectation = get_expectation_val_with_readout_mitigation(
            training_circuit,
            observable,
            noise_model,
            nshots,
            readout,
            qubit_map,
            backend=backend,
        )

        data["noise-free"].append(expectation)
        data["noisy"].append(noisy_expectation)
        lambda_list.append(1 - noisy_expectation / expectation)

    dep_param = np.mean(lambda_list)
    dep_param_std = np.std(lambda_list)

    noisy_expectation = get_expectation_val_with_readout_mitigation(
        circuit, observable, noise_model, nshots, readout, qubit_map, backend=backend
    )
    one_dep_squared = (1 - dep_param) ** 2
    dep_std_squared = dep_param_std**2

    mitigated_expectation = (
        (1 - dep_param) * noisy_expectation / (one_dep_squared + dep_std_squared)
    )
    mitigated_expectation_std = (
        dep_param_std
        * abs(noisy_expectation)
        * abs((1 - dep_param) ** 2 - dep_std_squared)
        / (one_dep_squared + dep_std_squared) ** 2
    )

    if full_output:
        return (
            mitigated_expectation,
            mitigated_expectation_std,
            dep_param,
            dep_param_std,
            lambda_list,
            data,
        )

    return mitigated_expectation


def _execute_circuit(circuit, qubit_map, noise_model=None, nshots=10000, backend=None):
    """
    Helper function to execute the given circuit with the specified parameters.

    Args:
        circuit (qibo.models.Circuit): input circuit.
        qubit_map (list): the qubit map. If ``None``, a list of range of circuit's qubits is used. Defaults to ``None``.
        noise_model (qibo.models.noise.Noise, optional): The noise model to be applied. Defaults to ``None``.
        nshots (int): the number of shots for the circuit execution. Defaults to :math:`10000`..
        backend (qibo.backends.abstract.Backend, optional): the backend to be used in the execution.
            If None, it uses the global backend. Defaults to ``None``.

    Returns:
        qibo.states.CircuitResult: The result of the circuit execution.
    """
    from qibo.transpiler.placer import Custom

    if backend is None:  # pragma: no cover
        backend = GlobalBackend()
    elif backend.name == "qibolab":  # pragma: no cover
        backend.transpiler.passes[1] = Custom(
            map=qubit_map, connectivity=backend.platform.topology
        )
    elif noise_model is not None:
        circuit = noise_model.apply(circuit)

    circuit_result = backend.execute_circuit(circuit, nshots=nshots)

    return circuit_result
