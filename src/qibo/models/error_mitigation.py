"""Error Mitigation Methods."""

from math import factorial

import numpy as np
from scipy.optimize import curve_fit

from qibo import gates
from qibo.backends import GlobalBackend
from qibo.config import raise_error


def get_gammas(c, solve: bool = True):
    """Standalone function to compute the ZNE coefficients given the noise levels.

    Args:
        c (numpy.ndarray): array containing the different noise levels.
            Note that in the CNOT insertion paradigm this corresponds to
            the number of CNOT pairs to be inserted. The canonical ZNE
            noise levels are obtained as ``2 * c + 1``.
        solve (bool, optional): If ``True``, computes the coeffients by solving the
            linear system. If ``False``, use the analytical solution valid
            for the CNOT insertion method. Default is ``True``.

    Returns:
        numpy.ndarray: The computed coefficients.
    """
    if solve:
        c = 2 * c + 1
        a = np.array([c**i for i in range(len(c))])
        b = np.zeros(len(c))
        b[0] = 1
        gammas = np.linalg.solve(a, b)
    else:
        cmax = c[-1]
        gammas = np.array(
            [
                1
                / (2 ** (2 * cmax) * factorial(i))
                * (-1) ** i
                / (1 + 2 * i)
                * factorial(1 + 2 * cmax)
                / (factorial(cmax) * factorial(cmax - i))
                for i in c
            ]
        )

    return gammas


def get_noisy_circuit(circuit, num_insertions: int, insertion_gate: str = "CNOT"):
    """Standalone function to generate the noisy circuit with the inverse gate pairs insertions.

    Args:
        circuit (:class:`qibo.models.circuit.Circuit`): circuit to modify.
        num_insertions (int): number of insertion gate pairs to add.
        insertion_gate (str, optional): gate to be used in the insertion.
            If ``"RX"``, the gate used is :math:``RX(\\pi / 2)``.
            Default is ``"CNOT"``.

    Returns:
        :class:`qibo.models.Circuit`: The circuit with the inserted CNOT pairs.
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
                for i in range(num_insertions):
                    noisy_circuit.add(gates.CNOT(control, target))
                    noisy_circuit.add(gates.CNOT(control, target))
            elif gate.init_kwargs["theta"] == theta:
                qubit = gate.qubits[0]
                for i in range(num_insertions):
                    noisy_circuit.add(gates.RX(qubit, theta=theta))
                    noisy_circuit.add(gates.RX(qubit, theta=-theta))

    return noisy_circuit


def ZNE(
    circuit,
    observable,
    noise_levels,
    noise_model=None,
    nshots=int(1e4),
    solve_for_gammas=False,
    insertion_gate="CNOT",
    readout: dict = {},
    backend=None,
):
    """Runs the Zero Noise Extrapolation method for error mitigation.

    The different noise levels are realized by the insertion of pairs of
    either ``CNOT`` or ``RX(pi/2)`` gates that resolve to the identiy in
    the noise-free case.

    Args:
        circuit (:class:`qibo.models.Circuit`): input circuit.
        observable (numpy.ndarray): Observable to measure.
        noise_levels (numpy.ndarray): Sequence of noise levels.
        noise_model (:class:`qibo.noise.NoiseModel`, optional): Noise model applied
            to simulate noisy computation.
        nshots (int, optional): Number of shots.
        solve_for_gammas (bool, optional): If ``True``, explicitly solve the
            equations to obtain the ``gamma`` coefficients.
        insertion_gate (str, optional): gate to be used in the insertion.
            If ``"RX"``, the gate used is :math:``RX(\\pi / 2)``.
            Default is ``"CNOT"``.
        readout (dict, optional): It has the structure
            {'calibration_matrix': `numpy.ndarray`, 'ncircuits': `int`}.
            If passed, the calibration matrix or the randomized method is
            used to mitigate readout errors.
        backend (:class:`qibo.backends.abstract.Backend`, optional): Calculation engine.

    Returns:
        numpy.ndarray: Estimate of the expected value of ``observable`` in the noise free condition.
    """

    if backend is None:  # pragma: no cover
        backend = GlobalBackend()

    expected_val = []
    for num_insertions in noise_levels:
        noisy_circuit = get_noisy_circuit(
            circuit, num_insertions, insertion_gate=insertion_gate
        )
        if "ncircuits" in readout.keys():
            circuit_result, circuit_result_cal = apply_randomized_readout_mitigation(
                noisy_circuit, noise_model, nshots, readout["ncircuits"], backend
            )
        else:
            if noise_model is not None and backend.name != "qibolab":
                noisy_circuit = noise_model.apply(noisy_circuit)
            circuit_result = backend.execute_circuit(noisy_circuit, nshots=nshots)
        if "calibration_matrix" in readout.keys() is not None:
            circuit_result = apply_readout_mitigation(
                circuit_result, readout["calibration_matrix"]
            )
        val = circuit_result.expectation_from_samples(observable)
        if "ncircuits" in readout.keys():
            val /= circuit_result_cal.expectation_from_samples(observable)
        expected_val.append(val)

    gamma = get_gammas(noise_levels, solve=solve_for_gammas)

    return np.sum(gamma * expected_val)


def sample_training_circuit(
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
        backend (:class:`qibo.backends.abstract.Backend`, optional): Calculation engine.

    Returns:
        :class:`qibo.models.Circuit`: The sampled circuit.
    """
    if backend is None:  # pragma: no cover
        backend = GlobalBackend()

    if replacement_gates is None:
        replacement_gates = [(gates.RZ, {"theta": n * np.pi / 2}) for n in range(4)]

    # Find all the non-Clifford RZ gates
    gates_to_replace = []
    for i, gate in enumerate(circuit.queue):
        if isinstance(gate, gates.RZ):
            if gate.init_kwargs["theta"] % (np.pi / 2) != 0.0:
                gates_to_replace.append((i, gate))

    if len(gates_to_replace) == 0:
        raise_error(ValueError, "No non-Clifford RZ gate found, no circuit sampled.")

    # For each RZ gate build the possible candidates and
    # compute the frobenius distance to the candidates
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
    # Compute the scores
    prob = np.exp(-(distance**2) / sigma**2)
    # Sample which of the RZ found to substitute
    index = np.random.choice(
        range(len(gates_to_replace)),
        size=min(int(len(gates_to_replace) / 2), 50),
        replace=False,
        p=prob.sum(-1) / prob.sum(),
    )
    gates_to_replace = np.array([gates_to_replace[i] for i in index])
    prob = [prob[i] for i in index]
    # Sample which replacement gate to substitute with
    replacement = np.array([replacement[i] for i in index])
    replacement = [
        replacement[i][np.random.choice(range(len(p)), size=1, p=p / p.sum())[0]]
        for i, p in enumerate(prob)
    ]
    replacement = {i[0]: g for i, g in zip(gates_to_replace, replacement)}
    # Build the training circuit by substituting the sampled gates
    sampled_circuit = circuit.__class__(**circuit.init_kwargs)
    for i, gate in enumerate(circuit.queue):
        if i in replacement.keys():
            sampled_circuit.add(replacement[i])
        else:
            sampled_circuit.add(gate)

    return sampled_circuit


def CDR(
    circuit,
    observable,
    noise_model,
    nshots: int = int(1e4),
    model=lambda x, a, b: a * x + b,
    n_training_samples: int = 100,
    full_output: bool = False,
    readout: dict = {},
    backend=None,
):
    """Runs the Clifford Data Regression error mitigation method.

    Args:
        circuit (:class:`qibo.models.Circuit`): input circuit decomposed in the
            primitive gates ``X``, ``CNOT``, ``RX(pi/2)``, ``RZ(theta)``.
        observable (numpy.ndarray): observable to be measured.
        noise_model (:class:`qibo.noise.NoiseModel`): noise model used for simulating
            noisy computation.
        nshots (int, optional): number of shots.
        model (callable, optional): model used for fitting. This should be a callable
            function object ``f(x, *params)``, taking as input the predictor variable
            and the parameters. Default is a simple linear model ``f(x,a,b) := a*x + b``.
        n_training_samples (int, optional): number of training circuits to sample.
        full_output (bool, optional): if ``True``, this function returns additional
            information: ``val``, ``optimal_params``, ``train_val``.
        readout (dict, optional): It has the structure
            {'calibration_matrix': `numpy.ndarray`, 'ncircuits': `int`}.
            If passed, the calibration matrix or the randomized method is
            used to mitigate readout errors.
        backend (:class:`qibo.backends.abstract.Backend`, optional): calculation engine.

    Returns:
        mit_val (float): Mitigated expectation value of `observable`.
        val (float): Noisy expectation value of `observable`.
        optimal_params (list): Optimal values for `params`.
        train_val (dict): Contains the noise-free and noisy expectation values obtained with the training circuits.
    """

    # Set backend
    if backend is None:  # pragma: no cover
        backend = GlobalBackend()
    # Sample the training set
    training_circuits = [
        sample_training_circuit(circuit) for n in range(n_training_samples)
    ]
    # Run the sampled circuits
    train_val = {"noise-free": [], "noisy": []}
    for c in training_circuits:
        val = c(nshots=nshots).expectation_from_samples(observable)
        train_val["noise-free"].append(val)
        if "ncircuits" in readout.keys():
            circuit_result, circuit_result_cal = apply_randomized_readout_mitigation(
                c, noise_model, nshots, readout["ncircuits"], backend
            )
        else:
            if noise_model is not None and backend.name != "qibolab":
                c = noise_model.apply(c)
            circuit_result = backend.execute_circuit(c, nshots=nshots)
        if "calibration_matrix" in readout.keys() is not None:
            circuit_result = apply_readout_mitigation(
                circuit_result, readout["calibration_matrix"]
            )
        val = circuit_result.expectation_from_samples(observable)
        if "ncircuits" in readout.keys():
            val /= circuit_result_cal.expectation_from_samples(observable)
        train_val["noisy"].append(val)
    # Fit the model
    optimal_params = curve_fit(model, train_val["noisy"], train_val["noise-free"])[0]
    # Run the input circuit
    if "ncircuits" in readout.keys():
        circuit_result, circuit_result_cal = apply_randomized_readout_mitigation(
            circuit, noise_model, nshots, readout["ncircuits"], backend
        )
    else:
        if noise_model is not None and backend.name != "qibolab":
            circuit = noise_model.apply(circuit)
        circuit_result = backend.execute_circuit(circuit, nshots=nshots)
    if "calibration_matrix" in readout.keys() is not None:
        circuit_result = apply_readout_mitigation(
            circuit_result, readout["calibration_matrix"]
        )
    val = circuit_result.expectation_from_samples(observable)
    if "ncircuits" in readout.keys():
        val /= circuit_result_cal.expectation_from_samples(observable)
    mit_val = model(val, *optimal_params)

    # Return data
    if full_output == True:
        return mit_val, val, optimal_params, train_val
    else:
        return mit_val


def vnCDR(
    circuit,
    observable,
    noise_levels,
    noise_model,
    nshots: int = int(1e4),
    model=lambda x, *params: (x * np.array(params).reshape(-1, 1)).sum(0),
    n_training_samples: int = 100,
    insertion_gate: str = "CNOT",
    full_output: bool = False,
    readout: dict = {},
    backend=None,
):
    """Runs the variable-noise Clifford Data Regression error mitigation method.

    Args:
        circuit (:class:`qibo.models.Circuit`): input circuit decomposed in the
            primitive gates ``X``, ``CNOT``, ``RX(pi/2)``, ``RZ(theta)``.
        observable (numpy.ndarray): observable to be measured.
        noise_levels (numpy.ndarray): sequence of noise levels.
        noise_model (:class:`qibo.noise.NoiseModel`): noise model used for
            simulating noisy computation.
        nshots (int, optional): number of shots.
        model (callable, optional): model used for fitting. This should be a callable
            function object ``f(x, *params)``, taking as input the predictor variable
            and the parameters. Default is a simple linear model ``f(x,a,b) := a*x + b``.
        n_training_samples (int, optional): number of training circuits to sample.
        insertion_gate (str, optional): gate to be used in the insertion.
            If ``"RX"``, the gate used is :math:``RX(\\pi / 2)``.
            Default is ``"CNOT"``.
        full_output (bool, optional): if ``True``, this function returns additional
            information: ``val``, ``optimal_params``, ``train_val``.
        readout (dict, optional): It has the structure
            {'calibration_matrix': `numpy.ndarray`, 'ncircuits': `int`}.
            If passed, the calibration matrix or the randomized method is
            used to mitigate readout errors.
        backend (:class:`qibo.backends.abstract.Backend`, optional): calculation engine.

    Returns:
        mit_val (float): Mitigated expectation value of `observable`.
        val (list): Expectation value of `observable` with increased noise levels.
        optimal_params (list): Optimal values for `params`.
        train_val (dict): Contains the noise-free and noisy expectation values obtained
        with the training circuits.
    """

    # Set backend
    if backend is None:  # pragma: no cover
        backend = GlobalBackend()

    # Sample the training circuits
    training_circuits = [
        sample_training_circuit(circuit) for n in range(n_training_samples)
    ]
    train_val = {"noise-free": [], "noisy": []}

    # Add the different noise levels and run the circuits
    for c in training_circuits:
        val = c(nshots=nshots).expectation_from_samples(observable)
        train_val["noise-free"].append(val)
        for level in noise_levels:
            noisy_c = get_noisy_circuit(c, level, insertion_gate=insertion_gate)
            if "ncircuits" in readout.keys():
                (
                    circuit_result,
                    circuit_result_cal,
                ) = apply_randomized_readout_mitigation(
                    noisy_c, noise_model, nshots, readout["ncircuits"], backend
                )
            else:
                if noise_model is not None and backend.name != "qibolab":
                    noisy_c = noise_model.apply(noisy_c)
                circuit_result = backend.execute_circuit(noisy_c, nshots=nshots)
            if "calibration_matrix" in readout.keys():
                circuit_result = apply_readout_mitigation(
                    circuit_result, readout["calibration_matrix"]
                )
            val = circuit_result.expectation_from_samples(observable)
            if "ncircuits" in readout.keys():
                val /= circuit_result_cal.expectation_from_samples(observable)
            train_val["noisy"].append(val)

    # Repeat noise-free values for each noise level
    noisy_array = np.array(train_val["noisy"]).reshape(-1, len(noise_levels))

    # Fit the model
    params = np.random.rand(len(noise_levels))
    optimal_params = curve_fit(model, noisy_array.T, train_val["noise-free"], p0=params)

    # Run the input circuit
    val = []
    for level in noise_levels:
        noisy_c = get_noisy_circuit(circuit, level, insertion_gate=insertion_gate)
        if "ncircuits" in readout.keys():
            circuit_result, circuit_result_cal = apply_randomized_readout_mitigation(
                noisy_c, noise_model, nshots, readout["ncircuits"], backend
            )
        else:
            if noise_model is not None and backend.name != "qibolab":
                noisy_c = noise_model.apply(noisy_c)
            circuit_result = backend.execute_circuit(noisy_c, nshots=nshots)
        if "calibration_matrix" in readout.keys():
            circuit_result = apply_readout_mitigation(
                circuit_result, readout["calibration_matrix"]
            )
        expval = circuit_result.expectation_from_samples(observable)
        if "ncircuits" in readout.keys():
            expval /= circuit_result_cal.expectation_from_samples(observable)
        val.append(expval)

    mit_val = model(np.array(val).reshape(-1, 1), *optimal_params[0])[0]

    # Return data
    if full_output == True:
        return mit_val, val, optimal_params, train_val

    return mit_val


def calibration_matrix(nqubits, noise_model=None, nshots: int = 1000, backend=None):
    """Computes the calibration matrix for readout mitigation.

    Args:
        nqubits (int): Total number of qubits.
        noise_model (:class:`qibo.noise.NoiseModel`, optional): noise model used for simulating
            noisy computation. This matrix can be used to mitigate the effect of
            `qibo.noise.ReadoutError`.
        nshots (int, optional): number of shots.
        backend (:class:`qibo.backends.abstract.Backend`, optional): calculation engine.

    Returns:
        numpy.ndarray : The computed (`nqubits`, `nqubits`) calibration matrix for
            readout mitigation.
    """

    from qibo import Circuit  # pylint: disable=import-outside-toplevel

    if backend is None:  # pragma: no cover
        backend = GlobalBackend()

    matrix = np.zeros((2**nqubits, 2**nqubits))

    for i in range(2**nqubits):
        state = format(i, f"0{nqubits}b")

        circuit = Circuit(nqubits, density_matrix=True)
        for q, bit in enumerate(state):
            if bit == "1":
                circuit.add(gates.X(q))
        circuit.add(gates.M(*range(nqubits)))

        if noise_model is not None and backend.name != "qibolab":
            circuit = noise_model.apply(circuit)

        freq = backend.execute_circuit(circuit, nshots=nshots).frequencies()

        column = np.zeros(2**nqubits)
        for key in freq.keys():
            f = freq[key] / nshots
            column[int(key, 2)] = f
        matrix[:, i] = column

    return np.linalg.inv(matrix)


def apply_readout_mitigation(state, calibration_matrix):
    """Updates the frequencies of the input state with the mitigated ones obtained with
    ``calibration_matrix * state.frequencies()``.

    Args:
        state (:class:`qibo.measurements.CircuitResult`): input state to be updated.
        calibration_matrix (numpy.ndarray): calibration matrix for readout mitigation.

    Returns:
        :class:`qibo.measurements.CircuitResult`: the input state with the updated frequencies.
    """
    freq = np.zeros(2**state.nqubits)
    for k, v in state.frequencies().items():
        freq[int(k, 2)] = v

    freq = freq.reshape(-1, 1)

    for i, val in enumerate(calibration_matrix @ freq):
        state._frequencies[i] = float(val)

    return state


def apply_randomized_readout_mitigation(
    circuit, noise_model=None, nshots: int = int(1e3), ncircuits: int = 10, backend=None
):
    """Implements the readout mitigation method proposed in https://arxiv.org/abs/2012.09738.

    Args:
        circuit (:class:`qibo.models.Circuit`): input circuit.
        noise_model(:class:`qibo.noise.NoiseModel`, optional): noise model used for
            simulating noisy computation. This matrix can be used to mitigate the
            effects of :class:`qibo.noise.ReadoutError`.
        nshots (int, optional): number of shots.
        ncircuits (int, optional): number of randomized circuits. Each of them uses
            ``int(nshots / ncircuits)`` shots.
        backend (:class:`qibo.backends.abstract.Backend`): calculation engine.

    Return:
        :class:`qibo.measurements.CircuitResult`: the state of the input circuit with
            mitigated frequencies.

    """
    from qibo import Circuit  # pylint: disable=import-outside-toplevel
    from qibo.quantum_info import (  # pylint: disable=import-outside-toplevel
        random_pauli,
    )

    if backend is None:  # pragma: no cover
        backend = GlobalBackend()

    qubits = circuit.queue[-1].qubits
    nshots_r = int(nshots / ncircuits)
    freq = np.zeros((ncircuits, 2), object)
    for k in range(ncircuits):
        circuit_c = circuit.copy(True)
        circuit_c.queue.pop()
        cal_circuit = Circuit(circuit.nqubits, density_matrix=True)

        x_gate = random_pauli(len(qubits), 1, subset=["I", "X"]).queue

        error_map = {}
        for gate in x_gate:
            if gate.name == "x":
                error_map[gate.qubits[0]] = 1

        circuits = [circuit_c, cal_circuit]
        results = []
        freqs = []
        for circ in circuits:
            circ.add(x_gate)
            circ.add(gates.M(*qubits))
            if noise_model is not None and backend.name != "qibolab":
                circ = noise_model.apply(circ)
            result = backend.execute_circuit(circ, nshots=nshots_r)
            result._samples = result.apply_bitflips(error_map)
            results.append(result)
            freqs.append(result.frequencies(binary=False))
        freq[k, :] = freqs

    for j in range(2):
        results[j].nshots = nshots
        freq_sum = freq[0, j]
        for f in freq[1::, j]:
            freq_sum += f
        results[j]._frequencies = freq_sum

    return results


def compute_inv_noise_1qb(gjk_1qb, one_qb_tilde, one_qb_exact_operators):
    """Computes the inverse noise for 1 qubit operator.

        Args:
        gjk_1qb (numpy.matrix): Matrix with elements Tr(Q_j rho_k) for one qubit.
        one_qb_tilde (numpy.matrix): List of matrices. Each matrix with elements Tr(Q_j O_l rho_k) for one qubit.
            Here, O_l represents the l-th single-qubit operator.
        one_qb_exact_operators (numpy.matrix): List of matrices. Each matrix of the exact l-th single-qubit operator O_l.

    Returns:
        numpy.matrix: inverse noise of all single qubit operators.
    """

    import numpy as np

    from qibo.backends import GlobalBackend

    nqubits = 1
    no_of_operators_1qb = np.shape(one_qb_tilde)[0]
    T = np.matrix([[1, 1, 1, 1], [0, 0, 1, 0], [0, 0, 0, 1], [1, -1, 0, 0]])

    identity = np.matrix(gates.I(0).matrix(backend=GlobalBackend()))
    xgate = np.matrix(gates.X(0).matrix(backend=GlobalBackend()))
    ygate = np.matrix(gates.Y(0).matrix(backend=GlobalBackend()))
    zgate = np.matrix(gates.Z(0).matrix(backend=GlobalBackend()))
    Pauligates_1qubit = [identity, xgate, ygate, zgate]

    # Compute inverse noise for 1 qubit operator

    # Compute operator_hat
    # $\mathcal{\hat{O}}^{(l)} = T g^{-1} \mathcal{\tilde{O}}^{(l)} T^{-1}$
    one_qb_hat = np.zeros((no_of_operators_1qb, 4**nqubits, 4**nqubits))
    for idx_ops in range(0, no_of_operators_1qb):
        one_qb_hat[idx_ops, :, :] = (
            T * np.linalg.inv(gjk_1qb) * one_qb_tilde[idx_ops, :, :] * np.linalg.inv(T)
        )

    # Exact PTM of operator(s)
    # $\mathcal{{O}}_{\sigma, \tau}^{(l), exact} = \frac{1}{d} tr(\sigma \mathcal(O) \tau)$ (exact PTM of the operator(s)
    one_qb_PTM = np.zeros((no_of_operators_1qb, 4**nqubits, 4**nqubits))
    for idx_ops in range(0, no_of_operators_1qb):
        for ii in range(0, 4**nqubits):
            for jj in range(0, 4**nqubits):
                one_qb_PTM[idx_ops, ii, jj] = (1 / 2**nqubits) * np.trace(
                    Pauligates_1qubit[ii]
                    @ one_qb_exact_operators[idx_ops]
                    @ Pauligates_1qubit[jj]
                    @ np.conjugate(np.transpose(one_qb_exact_operators[idx_ops]))
                )

    # Compute inverse noise
    # $(\mathcal{N}^{(l)})^{-1} = \mathcal{{O}}^{(l), exact} (\mathcal{\hat{O}}^{(l)})^{-1}$
    invNoise_1qb = np.zeros((no_of_operators_1qb, 4**nqubits, 4**nqubits))

    for idx_ops in range(0, no_of_operators_1qb):
        invNoise_1qb[idx_ops, :, :] = np.matrix(one_qb_PTM[idx_ops, :, :]) @ (
            np.matrix(np.linalg.inv(one_qb_hat[idx_ops, :, :]))
        )

    return invNoise_1qb


def compute_inv_noise_2qb(gjk_2qb, two_qb_tilde, two_qb_exact_operators):
    """Computes the inverse noise for 1 qubit operator.

        Args:
        gjk_2qb (numpy.matrix): Matrix with elements Tr(Q_j rho_k) for two qubits.
        one_qb_tilde (numpy.matrix): List of matrices. Each matrix with elements Tr(Q_j O_l rho_k) for two qubits.
            Here, O_l represents the l-th two-qubit operator.
        one_qb_exact_operators (numpy.matrix): List of matrices. Each matrix of the exact l-th two-qubit operator O_l.

    Returns:
        numpy.matrix: inverse noise of all two-qubit operators.
    """
    import numpy as np

    from qibo.backends import GlobalBackend

    nqubits = 2
    no_of_operators_2qb = np.shape(two_qb_tilde)[0]
    T = np.matrix([[1, 1, 1, 1], [0, 0, 1, 0], [0, 0, 0, 1], [1, -1, 0, 0]])

    identity = np.matrix(gates.I(0).matrix(backend=GlobalBackend()))
    xgate = np.matrix(gates.X(0).matrix(backend=GlobalBackend()))
    ygate = np.matrix(gates.Y(0).matrix(backend=GlobalBackend()))
    zgate = np.matrix(gates.Z(0).matrix(backend=GlobalBackend()))
    Pauligates_1qubit = [identity, xgate, ygate, zgate]
    Pauligates_2qubits = []
    for ii in range(0, 4):
        for jj in range(0, 4):
            temp_matrix = np.kron(Pauligates_1qubit[jj], Pauligates_1qubit[ii])
            Pauligates_2qubits.append(temp_matrix)

    # Compute inverse noise for 2 qubit operators

    # Compute operator_hat
    # $\mathcal{\hat{O}}^{(l)} = T g^{-1} \mathcal{\tilde{O}}^{(l)} T^{-1}$
    two_qb_hat = np.zeros((no_of_operators_2qb, 4**nqubits, 4**nqubits))
    for idx_ops in range(0, no_of_operators_2qb):
        two_qb_hat[idx_ops, :, :] = (
            np.kron(T, T)
            * np.linalg.inv(gjk_2qb)
            * two_qb_tilde[idx_ops, :, :]
            * np.linalg.inv(np.kron(T, T))
        )

    # Exact PTM of operator(s)
    # $\mathcal{{O}}_{\sigma, \tau}^{(l), exact} = \frac{1}{d} tr(\sigma \mathcal(O) \tau)$ (exact PTM of the operator(s)
    two_qb_PTM = np.zeros((no_of_operators_2qb, 4**nqubits, 4**nqubits))
    for idx_ops in range(0, no_of_operators_2qb):
        for ii in range(0, 4**nqubits):
            for jj in range(0, 4**nqubits):
                two_qb_PTM[idx_ops, ii, jj] = (1 / 2**nqubits) * np.trace(
                    Pauligates_2qubits[ii]
                    @ two_qb_exact_operators[idx_ops]
                    @ Pauligates_2qubits[jj]
                    @ np.conjugate(np.transpose(two_qb_exact_operators[idx_ops]))
                )

    # Compute inverse noise
    # $(\mathcal{N}^{(l)})^{-1} = \mathcal{{O}}^{(l), exact} (\mathcal{\hat{O}}^{(l)})^{-1}$
    invNoise_2qb = np.zeros((no_of_operators_2qb, 4**nqubits, 4**nqubits))

    for idx_ops in range(0, no_of_operators_2qb):
        invNoise_2qb[idx_ops, :, :] = np.matrix(two_qb_PTM[idx_ops, :, :]) @ (
            np.matrix(np.linalg.inv(two_qb_hat[idx_ops, :, :]))
        )

    return invNoise_2qb


def get_quasiprobabilities_1qb(gjk_1qb, Bjk_hat_1qb_reshaped, invNoise_1qb=None):
    """Computes the quasiprobabilities,
                    normalized probabilities,
                    cumulative distribution functions,
                    indicative sampling costs
                for 1 qubit operator(s).

        Args:
        gjk_1qb (numpy.matrix): Matrix with elements Tr(Q_j rho_k) for one qubit.
        Bjk_hat_1qb_reshaped (numpy.matrix): Matrix representing reshaped Pauli Transfer Matrix noisy basis operations.
        invNoise_1qb (numpy.matrix): Inverse noise of all one-qubit operators. If None, it means no single qubit operators.

        Computes:
        operator_stats_1qb = [qOvector_1qb, qOprob_1qb, CDF_O_1qb, CO_1qb]
        states_stats_1qb   = [qrhovector_1qb, qrhoprob_1qb, CDF_rho_1qb, Crho_1qb]
        meas_stats_1qb     = [qQvector_1qb, qQprob_1qb, CDF_Q_1qb, CQ_1qb]

        qXvector_1qb: Quasi-probabilities.
        qXprob_1qb  : Normalized probabilities from quasi-probabilities.
        CDF_X_1qb   : Cumulative distribution function used for sampling.
        CX_1qb      : Indicative sampling cost.
        where X in {O, rho, Q}

    Returns:
        If single qubit operators present: numpy.list: operator_stats_1qb, states_stats_1qb, meas_stats_1qb.
        If no single qubit operators:      numpy.list: states_stats_1qb, meas_stats_1qb.
    """
    import numpy as np

    from qibo.backends import GlobalBackend

    nqubits = 1
    T = np.matrix([[1, 1, 1, 1], [0, 0, 1, 0], [0, 0, 0, 1], [1, -1, 0, 0]])

    identity = np.matrix(gates.I(0).matrix(backend=GlobalBackend()))
    xgate = np.matrix(gates.X(0).matrix(backend=GlobalBackend()))
    ygate = np.matrix(gates.Y(0).matrix(backend=GlobalBackend()))
    zgate = np.matrix(gates.Z(0).matrix(backend=GlobalBackend()))

    Pauligates_1qubit = [identity, xgate, ygate, zgate]

    if invNoise_1qb is not None:
        ###################################################################################
        ### Decompose inverse 1 qubit inverse noise in term of 1 qubit basis operations ###
        ###################################################################################

        no_of_operators_1qb = np.shape(invNoise_1qb)[0]

        ###########################################
        ### Reshape invNoise from 1qb operators ###
        ###########################################

        invNoise_reshaped_1qb = np.zeros((16, no_of_operators_1qb))
        for idx_ops in range(0, no_of_operators_1qb):
            invNoise_reshaped_1qb[:, idx_ops] = np.reshape(
                invNoise_1qb[idx_ops, :, :], [16, 1], order="F"
            ).reshape(
                16,
            )

        #######################################################################
        ### Get indicative sampling cost contribution from 1 qb operator(s) ###
        #######################################################################

        # Find coefficients
        qOvector_1qb = np.zeros((13, no_of_operators_1qb))
        for idx_ops in range(0, no_of_operators_1qb):
            qOvector_1qb[:, idx_ops], _, _, _ = np.linalg.lstsq(
                Bjk_hat_1qb_reshaped, invNoise_reshaped_1qb[:, idx_ops], rcond=None
            )

        # Find indicative sampling cost for operator(s)
        CO_1qb = np.zeros((1, no_of_operators_1qb))
        for idx_ops in range(0, no_of_operators_1qb):
            CO_1qb[0, idx_ops] = np.sum(np.abs(qOvector_1qb[:, idx_ops]))

        # print(CO_1qb)
        qOprob_1qb = np.zeros((13, no_of_operators_1qb))
        CDF_O_1qb = np.zeros((13, no_of_operators_1qb))
        for idx_ops in range(0, no_of_operators_1qb):
            qOprob_1qb[:, idx_ops] = (
                np.abs(qOvector_1qb[:, idx_ops]) / CO_1qb[0][idx_ops]
            )
            CDF_O_1qb[:, idx_ops] = np.cumsum(qOprob_1qb[:, idx_ops])

        operator_stats_1qb = [qOvector_1qb, qOprob_1qb, CDF_O_1qb, CO_1qb]

    #############################################################################
    ### Get indicative sampling cost contribution from 1 qubit initial states ###
    #############################################################################

    Qhat_row_1qb = gjk_1qb * np.linalg.inv(T)
    Qhat_col_1qb = np.transpose(Qhat_row_1qb)

    rhohat_col_1qb = T

    idealrho_Pauligates = []
    for ii in range(0, 4):
        for jj in range(0, 4):
            idealrho_Pauligates.append(
                np.kron(Pauligates_1qubit[ii], Pauligates_1qubit[jj])
            )

    ideal_state_1qb = np.matrix([[1], [0], [0], [0]])
    ideal_state_rho_1qb = ideal_state_1qb * np.transpose(ideal_state_1qb)

    ideal_rho_PTM_1qb = np.zeros((4, 1))
    for ii in range(0, 4):
        ideal_rho_PTM_1qb[ii, 0] = np.trace(
            idealrho_Pauligates[ii] * ideal_state_rho_1qb
        )

    qrhovector_1qb, _, _, _ = np.linalg.lstsq(
        rhohat_col_1qb, ideal_rho_PTM_1qb, rcond=None
    )

    Crho_1qb = np.sum(np.abs(qrhovector_1qb))
    qrhoprob_1qb = np.abs(qrhovector_1qb) / Crho_1qb
    CDF_rho_1qb = np.cumsum(qrhoprob_1qb)

    states_stats_1qb = [qrhovector_1qb, qrhoprob_1qb, CDF_rho_1qb, Crho_1qb]

    ###########################################################################
    ### Get indicative sampling cost contribution from 1 qubit measurements ###
    ###########################################################################

    idealQ_Pauligates = []
    for ii in range(0, 4):
        idealQ_Pauligates.append(Pauligates_1qubit[ii])

    ideal_state_1qb_Q = Pauligates_1qubit[3]

    ideal_Q_PTM_row_1qb = np.zeros((1, 4**nqubits))
    for ii in range(0, 4**nqubits):
        ideal_Q_PTM_row_1qb[0, ii] = (1 / (2**nqubits)) * np.trace(
            idealQ_Pauligates[ii] * ideal_state_1qb_Q
        )

    ideal_Q_PTM_col_1qb = np.transpose(ideal_Q_PTM_row_1qb)

    qQvector_1qb, _, _, _ = np.linalg.lstsq(
        Qhat_col_1qb, ideal_Q_PTM_col_1qb, rcond=None
    )
    CQ_1qb = np.sum(np.abs(qQvector_1qb))
    qQprob_1qb = np.abs(qQvector_1qb) / CQ_1qb
    CDF_Q_1qb = np.cumsum(qQprob_1qb)

    meas_stats_1qb = [qQvector_1qb, qQprob_1qb, CDF_Q_1qb, CQ_1qb]

    if invNoise_1qb is not None:
        print("     CQ_1qb =", CQ_1qb)
        print("   Crho_1qb =", Crho_1qb)
        print("     CO_1qb =", CO_1qb[0])
        return operator_stats_1qb, states_stats_1qb, meas_stats_1qb

    elif invNoise_1qb is None:
        print("     CQ_1qb =", CQ_1qb)
        print("   Crho_1qb =", Crho_1qb)
        return states_stats_1qb, meas_stats_1qb


def get_quasiprobabilities_2qb(gjk_2qb, Bjk_hat_2qb_reshaped, invNoise_2qb=None):
    """Computes the quasiprobabilities,
                    normalized probabilities,
                    cumulative distribution functions,
                    indicative sampling costs
                for 2 qubit operator(s).

        Args:
        gjk_2qb (numpy.matrix): Matrix with elements Tr(Q_j rho_k) for two qubits.
        Bjk_hat_2qb_reshaped (numpy.matrix): Matrix representing reshaped Pauli Transfer Matrix noisy basis operations.
        invNoise_2qb (numpy.matrix): Inverse noise of all two-qubit operators. If None, it means no two-qubit operators.

        Computes:
        operator_stats_2qb = [qOvector_2qb, qOprob_2qb, CDF_O_2qb, CO_2qb]

        qXvector_2qb: Quasi-probabilities.
        qXprob_2qb  : Normalized probabilities from quasi-probabilities.
        CDF_X_2qb   : Cumulative distribution function used for sampling.
        CX_2qb      : Indicative sampling cost.
        where X in {O}

        Initial states and measurement bases - use single qubits.
    Returns:
        If two-qubit operators present: numpy.list: operator_stats_2qb.
        If no two-qubit operators: 0.
    """

    import numpy as np

    from qibo.backends import GlobalBackend

    nqubits = 2
    T = np.matrix([[1, 1, 1, 1], [0, 0, 1, 0], [0, 0, 0, 1], [1, -1, 0, 0]])

    identity = np.matrix(gates.I(0).matrix(backend=GlobalBackend()))
    xgate = np.matrix(gates.X(0).matrix(backend=GlobalBackend()))
    ygate = np.matrix(gates.Y(0).matrix(backend=GlobalBackend()))
    zgate = np.matrix(gates.Z(0).matrix(backend=GlobalBackend()))

    Pauligates_1qubit = [identity, xgate, ygate, zgate]
    Pauligates_2qubits = []
    for ii in range(0, 4):
        for jj in range(0, 4):
            temp_matrix = np.kron(Pauligates_1qubit[jj], Pauligates_1qubit[ii])
            Pauligates_2qubits.append(temp_matrix)

    if invNoise_2qb is not None:
        ###################################################################################
        ### Decompose inverse 2 qubit inverse noise in term of 2 qubit basis operations ###
        ###################################################################################

        no_of_operators_2qb = np.shape(invNoise_2qb)[0]

        ###########################################
        ### Reshape invNoise from 2qb operators ###
        ###########################################

        invNoise_reshaped_2qb = np.zeros((256, no_of_operators_2qb))
        for idx_ops in range(0, no_of_operators_2qb):
            invNoise_reshaped_2qb[:, idx_ops] = np.reshape(
                invNoise_2qb[idx_ops, :, :], [256, 1], order="F"
            ).reshape(
                256,
            )

        ##########################################################################
        ### Get indicative sampling cost contribution from 2 qubit operator(s) ###
        ##########################################################################

        # Find coefficients
        qOvector_2qb = np.zeros((241, no_of_operators_2qb))
        for idx_ops in range(0, no_of_operators_2qb):
            qOvector_2qb[:, idx_ops], _, _, _ = np.linalg.lstsq(
                Bjk_hat_2qb_reshaped, invNoise_reshaped_2qb[:, idx_ops], rcond=None
            )

        # Find indicative sampling cost for operator(s)
        CO_2qb = np.zeros((1, no_of_operators_2qb))
        for idx_ops in range(0, no_of_operators_2qb):
            CO_2qb[0, idx_ops] = np.sum(np.abs(qOvector_2qb[:, idx_ops]))

        qOprob_2qb = np.zeros((241, no_of_operators_2qb))
        CDF_O_2qb = np.zeros((241, no_of_operators_2qb))
        for idx_ops in range(0, no_of_operators_2qb):
            qOprob_2qb[:, idx_ops] = (
                np.abs(qOvector_2qb[:, idx_ops]) / CO_2qb[0][idx_ops]
            )
            CDF_O_2qb[:, idx_ops] = np.cumsum(qOprob_2qb[:, idx_ops])

        operator_stats_2qb = [qOvector_2qb, qOprob_2qb, CDF_O_2qb, CO_2qb]

        print("     CO_2qb =", CO_2qb[0])

        return operator_stats_2qb

    elif invNoise_2qb is None:
        print("Error. There are no two qubit operators.")
        return 0


def monte_carlo_sampling(
    circuit,
    states_stats_1qb,
    meas_stats_1qb,
    BasisOps_13,
    NshotsMC=1e4,
    operator_stats_1qb=None,
    operator_stats_2qb=None,
    BasisOps_241=None,
    noise_model=None,
    save_data=None,
):
    """Performs Monte Carlo Sampling.

        Args:
        circuit (:class:`qibo.models.Circuit`): input circuit.
        states_stats_1qb (numpy.array): Array containing [qrhovector_1qb, qrhoprob_1qb, CDF_rho_1qb, Crho_1qb]
        meas_stats_1qb (numpy.array): Array containing [qQvector_1qb, qQprob_1qb, CDF_Q_1qb, CQ_1qb]
        BasisOps_13 (numpy.matrix): List of 13 basis operations in matrix form.
        NshotsMC (int, optional): Number of shots used for the Monte Carlo sampling.
        operator_stats_1qb (numpy.array, optional): Array containing [qOvector_1qb, qOprob_1qb, CDF_O_1qb, CO_1qb]
        operator_stats_2qb (nunpy.array, optional): Array containing [qOvector_2qb, qOprob_2qb, CDF_O_2qb, CO_2qb]
        BasisOps_241 (numpy.matrix, optional): List of 241 basis operations in matrix form.
        noise_model (:class:`qibo.noise.NoiseModel`, optional): Noise model applied
            to simulate noisy computation.
        save_data (optional): Flag to save gate set tomography data. If None, skip.

    Returns:
        numpy.matrix: prob_QEM containing error mitigated distribution.
    """

    import time

    import numpy as np

    tic_MC = time.time()

    gatenames = (
        "BasisOp00",
        "BasisOp01",
        "BasisOp02",
        "BasisOp03",
        "BasisOp04",
        "BasisOp05",
        "BasisOp06",
        "BasisOp07",
        "BasisOp08",
        "BasisOp09",
        "BasisOp10",
        "BasisOp11",
        "BasisOp12",
    )

    type_of_gates = count_qb_gates(circuit)

    import os

    if save_data is not None:
        MC_results = "MC_results"
        if not os.path.exists(MC_results):
            os.makedirs(MC_results)

    qrhovector_1qb = states_stats_1qb[0]
    qrhoprob_1qb = states_stats_1qb[1]
    CDF_rho_1qb = states_stats_1qb[2]
    Crho_1qb = states_stats_1qb[3]

    qQvector_1qb = meas_stats_1qb[0]
    qQprob_1qb = meas_stats_1qb[1]
    CDF_Q_1qb = meas_stats_1qb[2]
    CQ_1qb = meas_stats_1qb[3]

    if operator_stats_1qb is not None:
        # Extract operator_stats_1qb info
        qOvector_1qb = operator_stats_1qb[0]
        qOprob_1qb = operator_stats_1qb[1]
        CDF_O_1qb = operator_stats_1qb[2]
        CO_1qb = operator_stats_1qb[3]

    if operator_stats_2qb is not None:
        # Extract operator_stats_2qb info
        qOvector_2qb = operator_stats_2qb[0]
        qOprob_2qb = operator_stats_2qb[1]
        CDF_O_2qb = operator_stats_2qb[2]
        CO_2qb = operator_stats_2qb[3]

    ############################################################################################
    ###                                                                                      ###
    ###  MM     MM   OOOO   NN   N  TTTTTTT EEEEE    CCCC     AA    RRRRR    LL      OOOO    ###
    ###  MMM   MMM  OO   O  NNN  N    TT    EE      CC       AA A   RR  RR   LL     OO   O   ###
    ###  MM M MM M  OO   O  NN N N    TT    EEE     CC      AAAAA   RRRRR    LL     OO   O   ###
    ###  MM  MM  M  OO   O  NN  NN    TT    EE      CC     AA    A  RR   R   LL     OO   O   ###
    ###  MM      M   OOOO   NN   N    TT    EEEEE    CCCC  AA    A  RR   R   LLLLL   OOOO    ###
    ###                                                                                      ###
    ############################################################################################

    def state_prep_single_register(qc, index_of_rho, qreg):
        if index_of_rho == 0:  # |0>
            pass

        elif index_of_rho == 1:  # |1>
            qc.add(gates.X(qreg))

        elif index_of_rho == 2:  # |+>
            qc.add(gates.H(qreg))

        elif index_of_rho == 3:  # |y+>
            qc.add(gates.H(qreg))
            qc.add(gates.S(qreg))

        return qc

    def measurements_two_registers(qc, index_of_Q, top_register, bottom_register):
        # ===========================================================================================================
        if index_of_Q == 0:  # top = Identity basis # bottom = Identity basis
            pass

        elif index_of_Q == 1:  # top = Identity basis # bottom = X basis
            qc.add(gates.H(bottom_register))

        elif index_of_Q == 2:  # top = Identity basis # bottom = Y basis
            qc.add(gates.SDG(bottom_register))
            qc.add(gates.H(bottom_register))

        elif index_of_Q == 3:  # top = Identity basis # bottom = Z basis
            pass

        # ===========================================================================================================
        elif index_of_Q == 4:  # top = X basis # bottom = Identity basis
            qc.add(gates.H(top_register))

        elif index_of_Q == 5:  # top = X basis # bottom = X basis
            qc.add(gates.H(top_register))
            qc.add(gates.H(bottom_register))

        elif index_of_Q == 6:  # top = X basis # bottom = Y basis
            qc.add(gates.H(top_register))
            qc.add(gates.SDG(bottom_register))
            qc.add(gates.H(bottom_register))

        elif index_of_Q == 7:  # top = X basis # bottom = Z basis
            qc.add(gates.H(top_register))

        # ===========================================================================================================
        elif index_of_Q == 8:  # top = Y basis # bottom = Identity basis
            qc.add(gates.SDG(top_register))
            qc.add(gates.H(top_register))

        elif index_of_Q == 9:  # top = Y basis # bottom = X basis
            qc.add(gates.SDG(top_register))
            qc.add(gates.H(top_register))
            qc.add(gates.H(bottom_register))

        elif index_of_Q == 10:  # top = Y basis # bottom = Y basis
            qc.add(gates.SDG(top_register))
            qc.add(gates.H(top_register))
            qc.add(gates.SDG(bottom_register))
            qc.add(gates.H(bottom_register))

        elif index_of_Q == 11:  # top = Y basis # bottom = Z basis
            qc.add(gates.SDG(top_register))
            qc.add(gates.H(top_register))

        # ===========================================================================================================
        elif index_of_Q == 12:  # top = Z basis # bottom = Identity basis
            pass

        elif index_of_Q == 13:  # top = Z basis # bottom = X basis
            qc.add(gates.H(bottom_register))

        elif index_of_Q == 14:  # top = Z basis # bottom = Y basis
            qc.add(gates.SDG(bottom_register))
            qc.add(gates.H(bottom_register))

        elif index_of_Q == 15:  # top = Z basis # bottom = Z basis
            pass

        # ===========================================================================================================

        return qc

    def measurements_single_register(qc, index_of_Q, qreg):
        # ===========================================================================================================
        if index_of_Q == 0:  # top = Identity basis # bottom = Identity basis
            pass

        elif index_of_Q == 1:  # top = Identity basis # bottom = X basis
            qc.add(gates.H(qreg))

        elif index_of_Q == 2:  # top = Identity basis # bottom = Y basis
            qc.add(gates.SDG(qreg))
            qc.add(gates.H(qreg))

        elif index_of_Q == 3:  # top = Identity basis # bottom = Z basis
            pass

        return qc

    if operator_stats_1qb is not None:
        # 1 qubit Cumulative Distribution Function
        CDF_O_1qb_matrix = np.zeros((13, 2))
        CDF_O_1qb_matrix[:, 0] = np.arange(0, 13, 1)
        CDF_O_1qb_matrix[:, 1] = np.arange(0, 13, 1)
        CDF_O_1qb_matrix = np.hstack((CDF_O_1qb_matrix, CDF_O_1qb, qOvector_1qb))
        # pretty_print_matrix(CDF_O_1qb_matrix)

    if operator_stats_2qb is not None:
        # 2 qubit Cumulative Distribution Function
        CDF_O_2qb_matrix = np.zeros((241, 3))
        CDF_O_2qb_matrix[:, 0] = np.arange(0, 241, 1)
        count = 0
        for ii in range(0, 13):
            for jj in range(0, 13):
                CDF_O_2qb_matrix[count, 1] = ii
                CDF_O_2qb_matrix[count, 2] = jj
                count += 1

        CDF_O_2qb_matrix = np.hstack((CDF_O_2qb_matrix, CDF_O_2qb, qOvector_2qb))
        # pretty_print_matrix(CDF_O_2qb_matrix)

    if save_data is not None:
        if type_of_gates["1 qb gate"] >= 1 and type_of_gates["2 qb gate"] == 0:
            MC_file_path = f"MC_results/MC_results_{NshotsMC}.txt"
            with open(MC_file_path, "w") as file:
                print(
                    "Run, idx_1qb_rho, idx_1qb_BO, idx_1qb_Q, sgn_1qb_rho, sgn_1qb_BO, sgn_1qb_Q, sgn_total, measurement_outcome",
                    file=file,
                )

        elif type_of_gates["2 qb gate"] >= 1 and type_of_gates["1 qb gate"] == 0:
            MC_file_path = f"MC_results/MC_results_{NshotsMC}.txt"
            with open(MC_file_path, "w") as file:
                print(
                    "Run, idx_1qb_rho, idx_2qb_BO, idx_1qb_Q, sgn_1qb_rho, sgn_2qb_BO, sgn_1qb_Q, sgn_total, measurement_outcome",
                    file=file,
                )

        elif type_of_gates["2 qb gate"] >= 1 and type_of_gates["1 qb gate"] >= 1:
            MC_file_path = f"MC_results/MC_results_{NshotsMC}.txt"
            with open(MC_file_path, "w") as file:
                print(
                    "Run, idx_1qb_rho, idx_1qb_BO, idx_2qb_BO, idx_1qb_Q, sgn_1qb_rho, sgn_1qb_BO, sgn_2qb_BO, sgn_1qb_Q, sgn_total, measurement_outcome",
                    file=file,
                )

    MC_results = []
    all_data = []
    QEM_matrix = np.zeros((2**circuit.nqubits, 2))

    for idx_run in range(1, NshotsMC + 1):
        MC_circ = Circuit(circuit.nqubits, density_matrix=True)

        index_of_1qb_rho_vec = []
        sgn_of_1qb_rho_vec = []

        index_of_1qb_BO_vec = []
        sgn_of_1qb_BO_vec = []
        index_of_2qb_BO_vec = []
        sgn_of_2qb_BO_vec = []

        index_of_1qb_Q_vec = []
        sgn_of_1qb_Q_vec = []

        count_1qb_gate = 0
        count_2qb_gate = 0

        ## STATE PREPARATION
        # For simplicity, stick with single qubit initial states for all qubits.
        for register in range(0, circuit.nqubits):
            index_of_1qb_rho = np.where(CDF_rho_1qb >= np.random.rand())[0][0]
            sgn_of_1qb_rho = int(np.sign(qrhovector_1qb[index_of_1qb_rho]))
            index_of_1qb_rho_vec.append(index_of_1qb_rho)
            sgn_of_1qb_rho_vec.append(sgn_of_1qb_rho)

            MC_circ = state_prep_single_register(MC_circ, index_of_1qb_rho, register)

        ## ADD BASIS OPERATIONS
        # Do the 1qb and 2qb basis operations respectively.
        for data in circuit.raw["queue"]:
            # print(data)
            num_qubit_gate = len(data["init_args"])
            name_qubit_gate = data["name"]
            class_qubit_gate = data["_class"]
            ctrl_qb = data["_control_qubits"]
            targ_qb = data["_target_qubits"]
            theta = data.get("init_kwargs", {}).get("theta", None)

            ## SINGLE QUBIT OPERATOR + BASIS OPERATION
            if num_qubit_gate == 1:
                if class_qubit_gate != "M":
                    if theta is None:
                        # print(f'Sample {num_qubit_gate}-qubit gate CDF')
                        MC_circ.add(getattr(gates, class_qubit_gate)(targ_qb[0]))

                    if theta is not None:
                        MC_circ.add(getattr(gates, class_qubit_gate)(targ_qb[0], theta))

                    ## Add 1qb basis operation
                    index_of_1qb_BO = np.where(
                        CDF_O_1qb_matrix[:, 2 + count_1qb_gate] >= np.random.rand()
                    )[0][
                        0
                    ]  # Start from 2nd column (0th indexing)
                    index_of_1qb_BO_vec.append(index_of_1qb_BO)
                    MC_circ.add(
                        gates.Unitary(
                            BasisOps_13[index_of_1qb_BO],
                            targ_qb[0],
                            trainable=False,
                            name="%s" % (gatenames[index_of_1qb_BO]),
                        )
                    )

                    ## Concurrently extract sgn of basis operation
                    sgn_of_1qb_BO = int(
                        np.sign(qOvector_1qb[index_of_1qb_BO, count_1qb_gate])
                    )
                    sgn_of_1qb_BO_vec.append(sgn_of_1qb_BO)

                    count_1qb_gate += 1

            ## TWO QUBIT OPERATOR + BASIS OPERATION
            elif num_qubit_gate == 2:
                # print(f'Sample {num_qubit_gate}-qubit gate CDF')
                if theta is None:
                    MC_circ.add(
                        getattr(gates, class_qubit_gate)(ctrl_qb[0], targ_qb[0])
                    )
                if theta is not None:
                    MC_circ.add(
                        getattr(gates, class_qubit_gate)(ctrl_qb[0], targ_qb[0], theta)
                    )

                ## Add 2qb basis operation
                top_register = ctrl_qb[0]  # 0
                bottom_register = targ_qb[0]  # 1

                index_of_2qb_BO = np.where(
                    CDF_O_2qb_matrix[:, 3 + count_2qb_gate] >= np.random.rand()
                )[0][
                    0
                ]  # Start from 3rd column (0th indexing)
                index_of_2qb_BO_vec.append(index_of_2qb_BO)

                # sgn_of_2qb_BO = int(np.sign(qOvector_2qb[index_of_2qb_BO,idx_ops]))
                sgn_of_2qb_BO = int(
                    np.sign(qOvector_2qb[index_of_2qb_BO, count_2qb_gate])
                )
                sgn_of_2qb_BO_vec.append(sgn_of_2qb_BO)

                index_of_2qb_BO_top = int(CDF_O_2qb_matrix[index_of_2qb_BO, 1])
                index_of_2qb_BO_bottom = int(CDF_O_2qb_matrix[index_of_2qb_BO, 2])

                # print('index_of_2qb_BO =',index_of_2qb_BO)
                # print('top =',index_of_2qb_BO_top, 'bottom =', index_of_2qb_BO_bottom)
                if index_of_2qb_BO < 169:
                    # ===============#=========================#
                    # 16 SCENARIOS  #  13^2 BASIS OPERATIONS  #
                    # =========================================#
                    if (
                        index_of_2qb_BO_top < 10 and index_of_2qb_BO_bottom < 10
                    ):  ### SINGLE QUBIT BASIS OP ON TOP AND BOTTOM SEPARATELY
                        MC_circ.add(
                            gates.Unitary(
                                BasisOps_13[index_of_2qb_BO_top],
                                top_register,
                                trainable=False,
                                name="%s" % (gatenames[index_of_2qb_BO_top]),
                            )
                        )
                        MC_circ.add(
                            gates.Unitary(
                                BasisOps_13[index_of_2qb_BO_bottom],
                                bottom_register,
                                trainable=False,
                                name="%s" % (gatenames[index_of_2qb_BO_bottom]),
                            )
                        )

                    # ==================================================================================
                    elif (
                        index_of_2qb_BO_top < 10 and index_of_2qb_BO_bottom == 10
                    ):  ### PARTIALLY REPREPARE BOTTOM QUBIT in |+> state
                        MC_circ.add(
                            gates.Unitary(
                                BasisOps_13[index_of_2qb_BO_top],
                                top_register,
                                trainable=False,
                                name="%s" % (gatenames[index_of_2qb_BO_top]),
                            )
                        )
                        MC_circ.add(
                            gates.ResetChannel(bottom_register, [1, 0])
                        )  # ResetChannel(qubit, [prob0, prob1])
                        MC_circ.add(gates.H(bottom_register))

                    elif (
                        index_of_2qb_BO_top < 10 and index_of_2qb_BO_bottom == 11
                    ):  ### PARTIALLY REPREPARE BOTTOM QUBIT in |y+> state
                        MC_circ.add(
                            gates.Unitary(
                                BasisOps_13[index_of_2qb_BO_top],
                                top_register,
                                trainable=False,
                                name="%s" % (gatenames[index_of_2qb_BO_top]),
                            )
                        )
                        MC_circ.add(
                            gates.ResetChannel(bottom_register, [1, 0])
                        )  # ResetChannel(qubit, [prob0, prob1])
                        MC_circ.add(gates.H(bottom_register))
                        MC_circ.add(gates.S(bottom_register))

                    elif (
                        index_of_2qb_BO_top < 10 and index_of_2qb_BO_bottom == 12
                    ):  ### PARTIALLY REPREPARE BOTTOM QUBIT in |0> state
                        MC_circ.add(
                            gates.Unitary(
                                BasisOps_13[index_of_2qb_BO_top],
                                top_register,
                                trainable=False,
                                name="%s" % (gatenames[index_of_2qb_BO_top]),
                            )
                        )
                        MC_circ.add(
                            gates.ResetChannel(bottom_register, [1, 0])
                        )  # ResetChannel(qubit, [prob0, prob1])

                    # ==================================================================================
                    elif (
                        index_of_2qb_BO_top == 10 and index_of_2qb_BO_bottom < 10
                    ):  ### PARTIALLY REPREPARE TOP QUBIT in |+> state
                        MC_circ.add(
                            gates.Unitary(
                                BasisOps_13[index_of_2qb_BO_bottom],
                                bottom_register,
                                trainable=False,
                                name="%s" % (gatenames[index_of_2qb_BO_bottom]),
                            )
                        )
                        MC_circ.add(
                            gates.ResetChannel(top_register, [1, 0])
                        )  # ResetChannel(qubit, [prob0, prob1])
                        MC_circ.add(gates.H(top_register))

                    elif (
                        index_of_2qb_BO_top == 11 and index_of_2qb_BO_bottom < 10
                    ):  ### PARTIALLY REPREPARE TOP QUBIT in |y+> state
                        MC_circ.add(
                            gates.Unitary(
                                BasisOps_13[index_of_2qb_BO_bottom],
                                bottom_register,
                                trainable=False,
                                name="%s" % (gatenames[index_of_2qb_BO_bottom]),
                            )
                        )
                        MC_circ.add(
                            gates.ResetChannel(top_register, [1, 0])
                        )  # ResetChannel(qubit, [prob0, prob1])
                        MC_circ.add(gates.H(top_register))
                        MC_circ.add(gates.S(top_register))

                    elif (
                        index_of_2qb_BO_top == 12 and index_of_2qb_BO_bottom < 10
                    ):  ### PARTIALLY REPREPARE TOP QUBIT in |0> state
                        MC_circ.add(
                            gates.Unitary(
                                BasisOps_13[index_of_2qb_BO_bottom],
                                bottom_register,
                                trainable=False,
                                name="%s" % (gatenames[index_of_2qb_BO_bottom]),
                            )
                        )
                        MC_circ.add(
                            gates.ResetChannel(top_register, [1, 0])
                        )  # ResetChannel(qubit, [prob0, prob1])

                    # ==================================================================================
                    elif (
                        index_of_2qb_BO_top == 10 and index_of_2qb_BO_bottom == 10
                    ):  ### REPREPARE TOP QUBIT in |+> state, BOTTOM QUBIT in |+> state
                        MC_circ.add(
                            gates.ResetChannel(top_register, [1, 0])
                        )  # ResetChannel(qubit, [prob0, prob1])
                        MC_circ.add(
                            gates.ResetChannel(bottom_register, [1, 0])
                        )  # ResetChannel(qubit, [prob0, prob1])
                        MC_circ.add(gates.H(top_register))
                        MC_circ.add(gates.H(bottom_register))

                    elif (
                        index_of_2qb_BO_top == 10 and index_of_2qb_BO_bottom == 11
                    ):  ### REPREPARE TOP QUBIT in |+> state, BOTTOM QUBIT in |y+> state
                        MC_circ.add(
                            gates.ResetChannel(top_register, [1, 0])
                        )  # ResetChannel(qubit, [prob0, prob1])
                        MC_circ.add(
                            gates.ResetChannel(bottom_register, [1, 0])
                        )  # ResetChannel(qubit, [prob0, prob1])
                        MC_circ.add(gates.H(top_register))
                        MC_circ.add(gates.H(bottom_register))
                        MC_circ.add(gates.S(bottom_register))

                    elif (
                        index_of_2qb_BO_top == 10 and index_of_2qb_BO_bottom == 12
                    ):  ### REPREPARE TOP QUBIT in |+> state, BOTTOM QUBIT in |0> state
                        MC_circ.add(
                            gates.ResetChannel(top_register, [1, 0])
                        )  # ResetChannel(qubit, [prob0, prob1])
                        MC_circ.add(
                            gates.ResetChannel(bottom_register, [1, 0])
                        )  # ResetChannel(qubit, [prob0, prob1])
                        MC_circ.add(gates.H(top_register))

                    # ==================================================================================
                    elif (
                        index_of_2qb_BO_top == 11 and index_of_2qb_BO_bottom == 10
                    ):  ### REPREPARE TOP QUBIT in |y+> state, BOTTOM QUBIT in |+> state
                        MC_circ.add(
                            gates.ResetChannel(top_register, [1, 0])
                        )  # ResetChannel(qubit, [prob0, prob1])
                        MC_circ.add(
                            gates.ResetChannel(bottom_register, [1, 0])
                        )  # ResetChannel(qubit, [prob0, prob1])
                        MC_circ.add(gates.H(top_register))
                        MC_circ.add(gates.S(top_register))
                        MC_circ.add(gates.H(bottom_register))

                    elif (
                        index_of_2qb_BO_top == 11 and index_of_2qb_BO_bottom == 11
                    ):  ### REPREPARE TOP QUBIT in |y+> state, BOTTOM QUBIT in |y+> state
                        MC_circ.add(
                            gates.ResetChannel(top_register, [1, 0])
                        )  # ResetChannel(qubit, [prob0, prob1])
                        MC_circ.add(
                            gates.ResetChannel(bottom_register, [1, 0])
                        )  # ResetChannel(qubit, [prob0, prob1])
                        MC_circ.add(gates.H(top_register))
                        MC_circ.add(gates.S(top_register))
                        MC_circ.add(gates.H(bottom_register))
                        MC_circ.add(gates.S(bottom_register))

                    elif (
                        index_of_2qb_BO_top == 11 and index_of_2qb_BO_bottom == 12
                    ):  ### REPREPARE TOP QUBIT in |y+> state, BOTTOM QUBIT in |0> state
                        MC_circ.add(
                            gates.ResetChannel(top_register, [1, 0])
                        )  # ResetChannel(qubit, [prob0, prob1])
                        MC_circ.add(
                            gates.ResetChannel(bottom_register, [1, 0])
                        )  # ResetChannel(qubit, [prob0, prob1])
                        MC_circ.add(gates.H(top_register))
                        MC_circ.add(gates.S(top_register))

                    # ==================================================================================
                    elif (
                        index_of_2qb_BO_top == 12 and index_of_2qb_BO_bottom == 10
                    ):  ### REPREPARE TOP QUBIT in |0> state, BOTTOM QUBIT in |+> state
                        MC_circ.add(
                            gates.ResetChannel(top_register, [1, 0])
                        )  # ResetChannel(qubit, [prob0, prob1])
                        MC_circ.add(
                            gates.ResetChannel(bottom_register, [1, 0])
                        )  # ResetChannel(qubit, [prob0, prob1])
                        MC_circ.add(gates.H(bottom_register))

                    elif (
                        index_of_2qb_BO_top == 12 and index_of_2qb_BO_bottom == 11
                    ):  ### REPREPARE TOP QUBIT in |0> state, BOTTOM QUBIT in |y+> state
                        MC_circ.add(
                            gates.ResetChannel(top_register, [1, 0])
                        )  # ResetChannel(qubit, [prob0, prob1])
                        MC_circ.add(
                            gates.ResetChannel(bottom_register, [1, 0])
                        )  # ResetChannel(qubit, [prob0, prob1])
                        MC_circ.add(gates.H(bottom_register))
                        MC_circ.add(gates.S(bottom_register))

                    elif (
                        index_of_2qb_BO_top == 12 and index_of_2qb_BO_bottom == 12
                    ):  ### REPREPARE TOP QUBIT in |0> state, BOTTOM QUBIT in |0> state
                        MC_circ.add(
                            gates.ResetChannel(top_register, [1, 0])
                        )  # ResetChannel(qubit, [prob0, prob1])
                        MC_circ.add(
                            gates.ResetChannel(bottom_register, [1, 0])
                        )  # ResetChannel(qubit, [prob0, prob1])

                if index_of_2qb_BO >= 169:
                    MC_circ.add(
                        gates.Unitary(
                            BasisOps_241[index_of_2qb_BO],
                            top_register,
                            bottom_register,
                            trainable=False,
                            name="BasisOp %d" % (index_of_2qb_BO),
                        )
                    )

                count_2qb_gate += 1

        ## MEASUREMENT
        # For simplicity, stick with single qubit measurement bases for all qubits.
        for register in range(0, circuit.nqubits):
            index_of_1qb_Q = np.where(CDF_Q_1qb >= np.random.rand())[0][0]
            sgn_of_1qb_Q = int(np.sign(qQvector_1qb[index_of_1qb_Q]))
            index_of_1qb_Q_vec.append(index_of_1qb_Q)
            sgn_of_1qb_Q_vec.append(sgn_of_1qb_Q)

            MC_circ = measurements_single_register(MC_circ, index_of_1qb_Q, register)

        ## COMPUTE TOTAL SGN
        if type_of_gates["2 qb gate"] >= 1 and type_of_gates["1 qb gate"] >= 1:
            # print('case1: no of 2qb gates >= 1 and no of 1qb gates >= 1:')
            total_sgns = [
                sgn_of_1qb_rho_vec,
                sgn_of_1qb_BO_vec,
                sgn_of_2qb_BO_vec,
                sgn_of_1qb_Q_vec,
            ]
        elif type_of_gates["1 qb gate"] >= 1 and type_of_gates["2 qb gate"] == 0:
            # print('case2: no of 2qb gates == 0 and no of 1qb gates >= 1:')
            total_sgns = [sgn_of_1qb_rho_vec, sgn_of_1qb_BO_vec, sgn_of_1qb_Q_vec]
        elif type_of_gates["2 qb gate"] >= 1 and type_of_gates["1 qb gate"] == 0:
            # print('case1: no of 2qb gates >= 1 and no of 1qb gates == 0:')
            total_sgns = [sgn_of_1qb_rho_vec, sgn_of_2qb_BO_vec, sgn_of_1qb_Q_vec]

        total_sgns = [
            item
            for sublist in total_sgns
            for item in (sublist if isinstance(sublist, list) else [sublist])
        ]
        sgn_tot = np.prod(total_sgns)

        if noise_model is not None and backend.name != "qibolab":
            MC_circ = noise_model.apply(MC_circ)

        for register in range(circuit.nqubits):
            MC_circ.add(gates.M(register))

        # PERFORM 1 SHOT
        result = MC_circ.execute(nshots=1)
        counts = dict(result.frequencies(binary=True))

        MC_output = int(list(counts)[0], 2)
        MC_output_binary = format(MC_output, "02b")

        if type_of_gates["1 qb gate"] >= 1 and type_of_gates["2 qb gate"] == 0:
            MC_run = [
                idx_run,
                *index_of_1qb_rho_vec,
                *index_of_1qb_BO_vec,
                *index_of_1qb_Q_vec,
                MC_output_binary,
            ]
        elif type_of_gates["1 qb gate"] == 0 and type_of_gates["2 qb gate"] >= 1:
            MC_run = [
                idx_run,
                *index_of_1qb_rho_vec,
                *index_of_2qb_BO_vec,
                *index_of_1qb_Q_vec,
                MC_output_binary,
            ]
        elif type_of_gates["1 qb gate"] >= 1 and type_of_gates["2 qb gate"] >= 1:
            MC_run = [
                idx_run,
                *index_of_1qb_rho_vec,
                *index_of_1qb_BO_vec,
                *index_of_2qb_BO_vec,
                *index_of_1qb_Q_vec,
                MC_output_binary,
            ]

        MC_results.append(MC_run)

        # Append data
        if save_data is not None:
            if type_of_gates["1 qb gate"] >= 1 and type_of_gates["2 qb gate"] == 0:
                # print(idx_run, *index_of_1qb_rho_vec, *index_of_1qb_BO_vec, *index_of_1qb_Q_vec, *sgn_of_1qb_rho_vec, *sgn_of_1qb_BO_vec, *sgn_of_1qb_Q_vec, sgn_tot, MC_output_binary)
                MC_file_path = f"MC_results/MC_results_{NshotsMC}.txt"
                with open(MC_file_path, "a") as file:
                    print(
                        idx_run,
                        *index_of_1qb_rho_vec,
                        *index_of_1qb_BO_vec,
                        *index_of_1qb_Q_vec,
                        *sgn_of_1qb_rho_vec,
                        *sgn_of_1qb_BO_vec,
                        *sgn_of_1qb_Q_vec,
                        sgn_tot,
                        MC_output_binary,
                        file=file,
                    )

            elif type_of_gates["2 qb gate"] >= 1 and type_of_gates["1 qb gate"] == 0:
                # print(idx_run, *index_of_1qb_rho_vec, *index_of_2qb_BO_vec, *index_of_1qb_Q_vec, *sgn_of_1qb_rho_vec, *sgn_of_2qb_BO_vec, *sgn_of_1qb_Q_vec, sgn_tot, MC_output_binary)
                MC_file_path = f"MC_results/MC_results_{NshotsMC}.txt"
                with open(MC_file_path, "a") as file:
                    print(
                        idx_run,
                        *index_of_1qb_rho_vec,
                        *index_of_2qb_BO_vec,
                        *index_of_1qb_Q_vec,
                        *sgn_of_1qb_rho_vec,
                        *sgn_of_2qb_BO_vec,
                        *sgn_of_1qb_Q_vec,
                        sgn_tot,
                        MC_output_binary,
                        file=file,
                    )

            elif type_of_gates["2 qb gate"] >= 1 and type_of_gates["1 qb gate"] >= 1:
                # print(idx_run, *index_of_1qb_rho_vec, *index_of_1qb_BO_vec, *index_of_2qb_BO_vec, *index_of_1qb_Q_vec, *sgn_of_1qb_rho_vec, *sgn_of_1qb_BO_vec, *sgn_of_2qb_BO_vec, *sgn_of_1qb_Q_vec, sgn_tot, MC_output_binary)
                MC_file_path = f"MC_results/MC_results_{NshotsMC}.txt"
                with open(MC_file_path, "a") as file:
                    print(
                        idx_run,
                        *index_of_1qb_rho_vec,
                        *index_of_1qb_BO_vec,
                        *index_of_2qb_BO_vec,
                        *index_of_1qb_Q_vec,
                        *sgn_of_1qb_rho_vec,
                        *sgn_of_1qb_BO_vec,
                        *sgn_of_2qb_BO_vec,
                        *sgn_of_1qb_Q_vec,
                        sgn_tot,
                        MC_output_binary,
                        file=file,
                    )

        ####################
        ### POST PROCESS ###
        ####################

        row = int(MC_output_binary, 2)
        if sgn_tot == -1:
            QEM_matrix[row, 1] += 1
        elif sgn_tot == 1:
            QEM_matrix[row, 0] += 1

        if np.remainder(idx_run, 5000) == 0:
            QEM_eff_counts = np.zeros((2**circuit.nqubits, 1))
            QEM_eff_prob = np.zeros((2**circuit.nqubits, 1))
            QEM_eff_counts[:, 0] = QEM_matrix[:, 0] - QEM_matrix[:, 1]

            QEM_eff_prob = QEM_eff_counts / np.sum(QEM_eff_counts[:, 0])
            print(f"prob_QEM with {idx_run} MC shots\n", QEM_eff_prob)

    QEM_eff_counts = np.zeros((2**circuit.nqubits, 1))
    QEM_eff_prob = np.zeros((2**circuit.nqubits, 1))
    QEM_eff_counts[:, 0] = QEM_matrix[:, 0] - QEM_matrix[:, 1]

    QEM_eff_prob = QEM_eff_counts / np.sum(QEM_eff_counts[:, 0])

    print(f"Final prob_QEM with {NshotsMC} Monte Carlo samples:\n", QEM_eff_prob)
    if (QEM_eff_prob < 0).any():
        print("Insufficient counts, negative values still exist in prob_QEM.")

    prob_QEM = {}
    for ii in range(0, 2**circuit.nqubits):
        key = bin(ii)[2:].zfill(circuit.nqubits)
        prob_QEM[key] = QEM_eff_prob[ii, 0]

    toc_MC = time.time() - tic_MC
    print("Total time: %.4f seconds" % (toc_MC))

    return prob_QEM


def count_qb_gates(circuit):
    """Small function to count the number of n-qubit gates.

        Args:
        circuit (:class:`qibo.models.Circuit`): input circuit.

    Returns:
        dict: Dictionary containing with keys '1 qb gates', '2 qb gates', '>2 qb gates'.
    """
    # Count the number of 1 and 2 qubit gates.
    type_of_gates = {}
    type_of_gates["1 qb gate"] = 0
    type_of_gates["2 qb gate"] = 0
    type_of_gates[">2 qb gate"] = 0

    for data in circuit.raw["queue"]:
        num_qubit_gate = len(data["init_args"])
        if num_qubit_gate == 1:
            type_of_gates["1 qb gate"] += 1
        elif num_qubit_gate == 2:
            type_of_gates["2 qb gate"] += 1
        else:
            type_of_gates[">2 qb gate"] += 1

    return type_of_gates


def PEC(
    circuit,
    NshotsGST=int(1e4),
    NshotsMC=int(1e4),
    noise_model=None,
    backend=None,
    save_data=None,
):
    """Runs the Probabilistic Error Cancellation method for error mitigation.

        Args:
        circuit (:class:`qibo.models.Circuit`): input circuit.
        noise_model (:class:`qibo.noise.NoiseModel`, optional): Noise model applied
            to simulate noisy computation.
        NshotsGST (int, optional): Number of shots used in Gate Set Tomography.
        NshotsMC  (int, optional): Number of shots used for the Monte Carlo sampling.
        backend (:class:`qibo.backends.abstract.Backend`, optional): Calculation engine.
        save_data: Flag to save data. If True, mkdir and save data. If None, skip.

    Returns:
        numpy.matrix: PEC's error-mitigated probability distribution (with sufficient NshotMC shots).
    """

    from qibo.backends import GlobalBackend

    if backend is None:  # pragma: no cover
        backend = GlobalBackend()

    import time

    import numpy as np

    tic_PEC = time.time()

    print("Circuit in question:")
    print(circuit.draw())

    type_of_gates = count_qb_gates(circuit)

    # Gate set tomography for empty circuit
    gjk_1qb = GST_1qb(NshotsGST, noise_model, save_data=save_data)
    if type_of_gates["2 qb gate"] >= 1:
        gjk_2qb = GST_2qb(NshotsGST, noise_model, save_data=save_data)

    # Gate set tomography for basis operations
    Bjk_hat_1qb, Bjk_hat_1qb_reshaped, BasisOps_13 = GST_1qb_basis_operations(
        NshotsGST, noise_model, gjk_1qb=gjk_1qb, save_data=None
    )
    if type_of_gates["2 qb gate"] >= 1:
        Bjk_hat_2qb, Bjk_hat_2qb_reshaped, BasisOps_241 = GST_2qb_basis_operations(
            NshotsGST, noise_model, gjk_2qb=gjk_2qb, save_data=None
        )

    # Gate set tomography for single/two qubit gate(s)
    one_qb_tilde = []
    two_qb_tilde = []
    one_qb_exact_operators = []
    two_qb_exact_operators = []
    one_qb_operatorname = []
    two_qb_operatorname = []
    for data in circuit.raw["queue"]:
        num_qubit_gate = len(data["init_args"])
        name_qubit_gate = data["name"]
        class_qubit_gate = data["_class"]
        ctrl_qb = data["_control_qubits"]
        targ_qb = data["_target_qubits"]
        theta = data.get("init_kwargs", {}).get("theta", None)

        if num_qubit_gate == 1:
            if class_qubit_gate != "M":
                # Gate set tomography for single qubit gate(s)
                hola = GST_1qb(
                    NshotsGST,
                    noise_model,
                    save_data=save_data,
                    class_qubit_gate=class_qubit_gate,
                    theta=theta,
                )
                one_qb_tilde.append(hola)

                if theta is None:
                    matrix_form = getattr(gates, class_qubit_gate)(targ_qb[0]).matrix()
                elif theta is not None:
                    matrix_form = getattr(gates, class_qubit_gate)(
                        targ_qb[0], theta
                    ).matrix()
                one_qb_exact_operators.append(matrix_form)
                one_qb_operatorname.append(name_qubit_gate)

        elif num_qubit_gate == 2:
            # Gate set tomography for two qubit gate(s)
            gracias = GST_2qb(
                NshotsGST,
                noise_model,
                save_data=save_data,
                class_qubit_gate=class_qubit_gate,
                ctrl_qb=ctrl_qb,
                targ_qb=targ_qb,
                theta=theta,
            )
            two_qb_tilde.append(gracias)

            if theta is None:
                matrix_form = getattr(gates, class_qubit_gate)(
                    ctrl_qb[0], targ_qb[0]
                ).matrix()
            elif theta is not None:
                matrix_form = getattr(gates, class_qubit_gate)(
                    ctrl_qb[0], targ_qb[0], theta
                ).matrix()
            two_qb_exact_operators.append(matrix_form)
            two_qb_operatorname.append(name_qubit_gate)

    one_qb_tilde = np.reshape(one_qb_tilde, [len(one_qb_tilde), 4, 4])
    one_qb_exact_operators = np.reshape(
        one_qb_exact_operators, [len(one_qb_exact_operators), 2, 2]
    )

    # Compute inverse noise
    invNoise_1qb = compute_inv_noise_1qb(gjk_1qb, one_qb_tilde, one_qb_exact_operators)
    if type_of_gates["2 qb gate"] >= 1:
        two_qb_tilde = np.reshape(two_qb_tilde, [len(two_qb_tilde), 16, 16])
        two_qb_exact_operators = np.reshape(
            two_qb_exact_operators, [len(two_qb_exact_operators), 4, 4]
        )

        invNoise_2qb = compute_inv_noise_2qb(
            gjk_2qb, two_qb_tilde, two_qb_exact_operators
        )

    # Compute quasi probabilities
    operator_stats_1qb, states_stats_1qb, meas_stats_1qb = get_quasiprobabilities_1qb(
        gjk_1qb, Bjk_hat_1qb_reshaped, invNoise_1qb=invNoise_1qb
    )

    if type_of_gates["2 qb gate"] >= 1:
        operator_stats_2qb = get_quasiprobabilities_2qb(
            gjk_2qb, Bjk_hat_2qb_reshaped, invNoise_2qb=invNoise_2qb
        )

    # Compute Csample for 1 and 2 qubits
    Csample_1qb = (
        np.product(operator_stats_1qb[3])
        * states_stats_1qb[3] ** circuit.nqubits
        * meas_stats_1qb[3] ** circuit.nqubits
    )

    if type_of_gates["2 qb gate"] >= 1:
        Csample_2qb = np.product(operator_stats_2qb[3])
        Csample_total = Csample_1qb * Csample_2qb
    else:
        Csample_total = Csample_1qb

    # Compute total indicative sampling cost
    width = 12
    estimated_shots = (Csample_total / 0.01) ** 2
    print(
        f"Estimated shots needed for Monte Carlo sampling to have variance of 0.01: {int(estimated_shots):>{width},}"
    )
    estimated_shots = (Csample_total / 0.02) ** 2
    print(
        f"Estimated shots needed for Monte Carlo sampling to have variance of 0.02: {int(estimated_shots):>{width},}"
    )
    estimated_shots = (Csample_total / 0.05) ** 2
    print(
        f"Estimated shots needed for Monte Carlo sampling to have variance of 0.05: {int(estimated_shots):>{width},}"
    )
    estimated_shots = (Csample_total / 0.1) ** 2
    print(
        f"Estimated shots needed for Monte Carlo sampling to have variance of 0.10: {int(estimated_shots):>{width},}"
    )
    estimated_shots = (Csample_total / 0.2) ** 2
    print(
        f"Estimated shots needed for Monte Carlo sampling to have variance of 0.20: {int(estimated_shots):>{width},}"
    )

    # Perform Monte Carlo sampling
    print(
        f"########################### MONTE CARLO SAMPLING USING {NshotsMC} SHOTS ###########################"
    )
    if type_of_gates["2 qb gate"] >= 1:
        error_mitigated_distribution = monte_carlo_sampling(
            circuit,
            states_stats_1qb,
            meas_stats_1qb,
            BasisOps_13,
            NshotsMC,
            operator_stats_1qb,
            operator_stats_2qb,
            BasisOps_241,
            noise_model=None,
            save_data=save_data,
        )
    else:
        error_mitigated_distribution = monte_carlo_sampling(
            circuit,
            states_stats_1qb,
            meas_stats_1qb,
            BasisOps_13,
            NshotsMC,
            operator_stats_1qb,
            operator_stats_2qb=None,
            BasisOps_241=None,
            noise_model=None,
            save_data=save_data,
        )

    # Compute variance
    variance = Csample_total / np.sqrt(NshotsMC)

    return error_mitigated_distribution, variance, Csample_total
