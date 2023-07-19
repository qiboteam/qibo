"""Error Mitigation Methods."""

import numpy as np
from scipy.optimize import curve_fit

from qibo import gates
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
                / (2 ** (2 * cmax) * np.math.factorial(i))
                * (-1) ** i
                / (1 + 2 * i)
                * np.math.factorial(1 + 2 * cmax)
                / (np.math.factorial(cmax) * np.math.factorial(cmax - i))
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

    if backend == None:  # pragma: no cover
        from qibo.backends import GlobalBackend

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

    Returns:
        :class:`qibo.models.Circuit`: The sampled circuit.
    """
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
                gate.matrix - [rep_gate.matrix for rep_gate in rep_gates],
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
    if backend == None:  # pragma: no cover
        from qibo.backends import GlobalBackend

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
    if backend == None:  # pragma: no cover
        from qibo.backends import GlobalBackend

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
        from qibo.backends import (  # pylint: disable=import-outside-toplevel
            GlobalBackend,
        )

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
        state (:class:`qibo.states.CircuitResult`): input state to be updated.
        calibration_matrix (numpy.ndarray): calibration matrix for readout mitigation.

    Returns:
        :class:`qibo.states.CircuitResult`: the input state with the updated frequencies.
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
        :class:`qibo.states.CircuitResult`: the state of the input circuit with
            mitigated frequencies.

    """
    from qibo import Circuit  # pylint: disable=import-outside-toplevel
    from qibo.quantum_info import (  # pylint: disable=import-outside-toplevel
        random_pauli,
    )

    if backend is None:  # pragma: no cover
        from qibo.backends import GlobalBackend

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
