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


def PEC(
    circuit, nshots_GST=int(1e4), nshots_MC=int(1e4), noise_model=None, backend=None
):
    """Runs the Probabilistic Error Cancellation method for error mitigation.

        Args:
        circuit (:class:`qibo.models.Circuit`): input circuit.
        #### observable (numpy.ndarray): Observable to measure.
        noise_model (:class:`qibo.noise.NoiseModel`, optional): Noise model applied
            to simulate noisy computation.
        nshots_GST (int, optional): Number of shots used in Gate Set Tomography.
        nshots_MC  (int, optional): Number of shots used for the Monte Carlo sampling.
        backend (:class:`qibo.backends.abstract.Backend`, optional): Calculation engine.
        #### Flag to save data. If True, mkdir and save data. If None, skip.
        #### Check input circuit for 1 and/or 2 qubit gates. Do the necessary Gate Set Tomography for 1 and/or 2 qubits.

    Returns:
        numpy.matrix: PEC-error-mitigated probability distribution.
    """

    from qibo.backends import GlobalBackend

    if backend is None:  # pragma: no cover
        backend = GlobalBackend()

    import time

    tic_PEC = time.time()

    #######################################################
    ### Use os to create the directories if not present ###
    #######################################################
    # The file directories needed are:
    # GST_basisops_qibo_1qb
    # GST_basisops_qibo_2qb
    # GST_gates_qibo_1qb
    # GST_gates_qibo_2qb
    # GST_nogates_qibo_1qb
    # GST_nogates_qibo_2qb
    # MC_data

    import os

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

    # print(type_of_gates)

    if type_of_gates[">2 qb gate"] > 0:
        print("This code does not do PEC for >2 qubits")
        return np.NaN, np.NaN

    elif type_of_gates[">2 qb gate"] == 0:
        if type_of_gates["1 qb gate"] >= 1 and type_of_gates["2 qb gate"] == 0:
            print("Do 1qb GST only")

            GST_nogates_qibo_1qb = "GST_nogates_qibo_1qb"
            if not os.path.exists(GST_nogates_qibo_1qb):
                os.makedirs(GST_nogates_qibo_1qb)

            GST_gates_qibo_1qb = "GST_gates_qibo_1qb"
            if not os.path.exists(GST_gates_qibo_1qb):
                os.makedirs(GST_gates_qibo_1qb)

            GST_basisops_qibo_1qb = "GST_basisops_qibo_1qb"
            if not os.path.exists(GST_basisops_qibo_1qb):
                os.makedirs(GST_basisops_qibo_1qb)

        elif type_of_gates["2 qb gate"] >= 1 and type_of_gates["1 qb gate"] == 0:
            print("Do 2qb GST only")

            GST_nogates_qibo_1qb = "GST_nogates_qibo_1qb"
            if not os.path.exists(GST_nogates_qibo_1qb):
                os.makedirs(GST_nogates_qibo_1qb)

            GST_nogates_qibo_2qb = "GST_nogates_qibo_2qb"
            if not os.path.exists(GST_nogates_qibo_2qb):
                os.makedirs(GST_nogates_qibo_2qb)

            GST_gates_qibo_2qb = "GST_gates_qibo_2qb"
            if not os.path.exists(GST_gates_qibo_2qb):
                os.makedirs(GST_gates_qibo_2qb)

            GST_basisops_qibo_2qb = "GST_basisops_qibo_2qb"
            if not os.path.exists(GST_basisops_qibo_2qb):
                os.makedirs(GST_basisops_qibo_2qb)

        elif type_of_gates["2 qb gate"] >= 1 and type_of_gates["1 qb gate"] >= 1:
            print("Do 1qb and 2qb GST")

            GST_nogates_qibo_1qb = "GST_nogates_qibo_1qb"
            if not os.path.exists(GST_nogates_qibo_1qb):
                os.makedirs(GST_nogates_qibo_1qb)

            GST_gates_qibo_1qb = "GST_gates_qibo_1qb"
            if not os.path.exists(GST_gates_qibo_1qb):
                os.makedirs(GST_gates_qibo_1qb)

            GST_basisops_qibo_1qb = "GST_basisops_qibo_1qb"
            if not os.path.exists(GST_basisops_qibo_1qb):
                os.makedirs(GST_basisops_qibo_1qb)

            GST_nogates_qibo_2qb = "GST_nogates_qibo_2qb"
            if not os.path.exists(GST_nogates_qibo_2qb):
                os.makedirs(GST_nogates_qibo_2qb)

            GST_gates_qibo_2qb = "GST_gates_qibo_2qb"
            if not os.path.exists(GST_gates_qibo_2qb):
                os.makedirs(GST_gates_qibo_2qb)

            GST_basisops_qibo_2qb = "GST_basisops_qibo_2qb"
            if not os.path.exists(GST_basisops_qibo_2qb):
                os.makedirs(GST_basisops_qibo_2qb)

    MC_results = "MC_results"
    if not os.path.exists(MC_results):
        os.makedirs(MC_results)

    ############################
    ### Some extra functions ###
    ############################
    from IPython.display import HTML, display

    # Function to apply custom formatting to matrices
    def pretty_print_matrix(matrix):
        matrix = np.array(matrix)
        html = "<table>"
        for row in matrix:
            html += "<tr>"
            for val in row:
                html += f"<td>{val:.4f}</td>"
            html += "</tr>"
        html += "</table>"
        display(HTML(html))

    #####################
    ### Load packages ###
    #####################

    import matplotlib.pyplot as plt
    import numpy as np

    from qibo import gates
    from qibo.models import Circuit

    ################################
    ### Prepare basis operations ###
    ################################

    def rx(theta):
        gate = [
            [np.cos(theta / 2), -1j * np.sin(theta / 2)],
            [-1j * np.sin(theta / 2), np.cos(theta / 2)],
        ]
        return np.matrix(gate)

    def ry(theta):
        gate = [
            [np.cos(theta / 2), -np.sin(theta / 2)],
            [np.sin(theta / 2), np.cos(theta / 2)],
        ]
        return np.matrix(gate)

    def rz(theta):
        gate = [[np.exp(-1j * theta / 2), 0], [0, np.exp(1j * theta / 2)]]
        return np.matrix(gate)

    hadamard = (1 / np.sqrt(2)) * np.matrix([[1, 1], [1, -1]])
    identity = np.matrix([[1, 0], [0, 1]])
    xgate = np.matrix([[0, 1], [1, 0]])
    ygate = np.matrix([[0, -1j], [1j, 0]])
    zgate = np.matrix([[1, 0], [0, -1]])
    sgate = np.matrix([[1, 0], [0, np.exp(1j * np.pi / 2)]])
    sdggate = np.matrix([[1, 0], [0, np.exp(-1j * np.pi / 2)]])
    tgate = np.matrix([[1, 0], [0, np.exp(1j * np.pi / 4)]])
    tdggate = np.matrix([[1, 0], [0, np.exp(-1j * np.pi / 4)]])
    CNOT = np.matrix([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])
    SWAP = np.matrix([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])

    Pauligates_1qubit = [identity, xgate, ygate, zgate]

    Pauligates_2qubits = []
    for ii in range(0, 4):
        for jj in range(0, 4):
            temp_matrix = np.kron(Pauligates_1qubit[jj], Pauligates_1qubit[ii])
            Pauligates_2qubits.append(temp_matrix)

    # One qubit basis operations
    BO_1qubit = []
    BO_1qubit.append(identity)
    BO_1qubit.append(xgate)
    BO_1qubit.append(ygate)
    BO_1qubit.append(zgate)
    BO_1qubit.append((1 / np.sqrt(2)) * np.matrix([[1, 1j], [1j, 1]]))
    BO_1qubit.append((1 / np.sqrt(2)) * np.matrix([[1, 1], [-1, 1]]))
    BO_1qubit.append((1 / np.sqrt(2)) * np.matrix([[(1 + 1j), 0], [0, (1 - 1j)]]))
    BO_1qubit.append((1 / np.sqrt(2)) * np.matrix([[1, -1j], [1j, -1]]))
    BO_1qubit.append((1 / np.sqrt(2)) * np.matrix([[1, 1], [1, -1]]))
    BO_1qubit.append((1 / np.sqrt(2)) * np.matrix([[0, (1 - 1j)], [(1 + 1j), 0]]))
    BO_1qubit.append(identity)
    BO_1qubit.append(identity)
    BO_1qubit.append(identity)

    # Intermediate operations: "interm_ops"
    interm_ops = []
    interm_ops.append(CNOT)
    interm_ops.append(np.kron(xgate, identity) * CNOT * np.kron(xgate, identity))
    interm_ops.append(
        np.matrix(np.block([[identity, np.zeros((2, 2))], [np.zeros((2, 2)), sgate]]))
    )
    interm_ops.append(
        np.matrix(
            np.block([[identity, np.zeros((2, 2))], [np.zeros((2, 2)), hadamard]])
        )
    )
    interm_ops.append(
        np.kron(ry(np.pi / 4), identity) * CNOT * np.kron(ry(np.pi / 4), identity)
    )
    interm_ops.append(CNOT * np.kron(hadamard, identity))
    interm_ops.append(SWAP)
    interm_ops.append(
        np.matrix([[1, 0, 0, 0], [0, 0, 1j, 0], [0, 1j, 0, 0], [0, 0, 0, 1]])
    )
    interm_ops.append(SWAP * np.kron(hadamard, identity))

    # Two qubit basis operations
    # "Easy"
    BO_2qubits_easy = []
    for ii in range(0, 13):
        for jj in range(0, 13):
            temp_matrix = np.kron(BO_1qubit[ii], BO_1qubit[jj])
            BO_2qubits_easy.append(temp_matrix)

    # "Hard"
    BO_2qubits_hard = []
    for ii in range(0, 9):
        if ii == 6:
            # print(f'interm_op {ii}, 3 combinations')
            temp_matrix1 = (
                np.kron(identity, sgate * hadamard)
                * interm_ops[ii]
                * np.kron(identity, hadamard * sdggate)
            )
            temp_matrix2 = (
                np.kron(identity, hadamard * sdggate)
                * interm_ops[ii]
                * np.kron(identity, sgate * hadamard)
            )
            temp_matrix3 = (
                np.kron(identity, identity)
                * interm_ops[ii]
                * np.kron(identity, identity)
            )

            BO_2qubits_hard.append(temp_matrix1)
            BO_2qubits_hard.append(temp_matrix2)
            BO_2qubits_hard.append(temp_matrix3)

        elif ii == 7:
            # print(f'interm_op {ii}, 6 combinations')
            temp_matrix1 = (
                np.kron(sgate * hadamard, sgate * hadamard)
                * interm_ops[ii]
                * np.kron(hadamard * sdggate, hadamard * sdggate)
            )
            temp_matrix2 = (
                np.kron(sgate * hadamard, hadamard * sdggate)
                * interm_ops[ii]
                * np.kron(hadamard * sdggate, sgate * hadamard)
            )
            temp_matrix3 = (
                np.kron(sgate * hadamard, identity)
                * interm_ops[ii]
                * np.kron(hadamard * sdggate, identity)
            )
            temp_matrix4 = (
                np.kron(identity * hadamard, sgate * hadamard)
                * interm_ops[ii]
                * np.kron(identity * sdggate, hadamard * sdggate)
            )
            temp_matrix5 = (
                np.kron(identity * hadamard, hadamard * sdggate)
                * interm_ops[ii]
                * np.kron(identity * sdggate, sgate * hadamard)
            )
            temp_matrix6 = (
                np.kron(identity, identity)
                * interm_ops[ii]
                * np.kron(identity, identity)
            )

            BO_2qubits_hard.append(temp_matrix1)
            BO_2qubits_hard.append(temp_matrix2)
            BO_2qubits_hard.append(temp_matrix3)
            BO_2qubits_hard.append(temp_matrix4)
            BO_2qubits_hard.append(temp_matrix5)
            BO_2qubits_hard.append(temp_matrix6)
        else:
            # print(f'interm_op {ii}, 9 combinations')
            temp_matrix1 = (
                np.kron(sgate * hadamard, sgate * hadamard)
                * interm_ops[ii]
                * np.kron(hadamard * sdggate, hadamard * sdggate)
            )
            temp_matrix2 = (
                np.kron(sgate * hadamard, hadamard * sdggate)
                * interm_ops[ii]
                * np.kron(hadamard * sdggate, sgate * hadamard)
            )
            temp_matrix3 = (
                np.kron(sgate * hadamard, identity)
                * interm_ops[ii]
                * np.kron(hadamard * sdggate, identity)
            )

            temp_matrix4 = (
                np.kron(hadamard * sdggate, sgate * hadamard)
                * interm_ops[ii]
                * np.kron(sgate * hadamard, hadamard * sdggate)
            )
            temp_matrix5 = (
                np.kron(hadamard * sdggate, hadamard * sdggate)
                * interm_ops[ii]
                * np.kron(sgate * hadamard, sgate * hadamard)
            )
            temp_matrix6 = (
                np.kron(hadamard * sdggate, identity)
                * interm_ops[ii]
                * np.kron(sgate * hadamard, identity)
            )

            temp_matrix7 = (
                np.kron(identity, sgate * hadamard)
                * interm_ops[ii]
                * np.kron(identity, hadamard * sdggate)
            )
            temp_matrix8 = (
                np.kron(identity, hadamard * sdggate)
                * interm_ops[ii]
                * np.kron(identity, sgate * hadamard)
            )
            temp_matrix9 = (
                np.kron(identity, identity)
                * interm_ops[ii]
                * np.kron(identity, identity)
            )

            BO_2qubits_hard.append(temp_matrix1)
            BO_2qubits_hard.append(temp_matrix2)
            BO_2qubits_hard.append(temp_matrix3)
            BO_2qubits_hard.append(temp_matrix4)
            BO_2qubits_hard.append(temp_matrix5)
            BO_2qubits_hard.append(temp_matrix6)
            BO_2qubits_hard.append(temp_matrix7)
            BO_2qubits_hard.append(temp_matrix8)
            BO_2qubits_hard.append(temp_matrix9)

    BO_2qubits = BO_2qubits_easy + BO_2qubits_hard

    ##########################################
    ###                                    ###
    ###      GGGGG    SSSSS   TTTTTTTT     ###
    ###     GG       SS          TTT       ###
    ###     GG  GG    SSSS       TTT       ###
    ###     GG  GG       SS      TTT       ###
    ###      GGGGG   SSSSS       TTT       ###
    ###                                    ###
    ##########################################

    NshotsGST = nshots_GST

    def sort_counts_1qb(matrix):
        counts_matrix = np.zeros((2, 2))

        for ii in range(0, 2):
            word = bin(ii)[2:]
            counts_matrix[ii, 0] = word

        for row in matrix:
            row_number = int(str(int(row[0])), 2)
            counts_matrix[row_number, 1] = row[1]

        return counts_matrix

    def sort_counts_2qb(matrix):
        counts_matrix = np.zeros((4, 2))

        for ii in range(0, 4):
            word = bin(ii)[2:]
            counts_matrix[ii, 0] = word

        for row in matrix:
            row_number = int(str(int(row[0])), 2)
            counts_matrix[row_number, 1] = row[1]

        return counts_matrix

    #########################################
    ### Single register blank circuit GST ###
    #########################################

    def GST_measure_1qb(circuit, k, j):
        idx_basis = ["I", "X", "Y", "Z"]
        filedirectory = "GST_nogates_qibo_1qb/"

        result = circuit.execute(nshots=NshotsGST)
        counts = dict(result.frequencies(binary=True))

        # Save counts into a matrix. 0th column: word; 1st column: count value
        # --------------------------------------------------------------------
        fout = filedirectory + "No gates (k=%.2d,j=%.2d) %s_basis.txt" % (
            k,
            j,
            idx_basis[j],
        )
        fo = open(fout, "w")
        for v1, v2 in counts.items():
            fo.write(str(v1) + "  " + str(v2) + "\n")
        fo.close()

        # Find value for gjk matrix
        # -------------------------

        expectation_signs = np.matrix([[1, 1, 1, 1], [1, -1, -1, -1]])

        matrix = np.zeros((2, 2))
        for ii in range(0, 2):
            word = bin(ii)[2:]
            matrix[ii, 0] = word

        for v1, v2 in counts.items():
            # print(v1, v2)
            row = int(v1, 2)
            col = v2
            matrix[row, 1] = col

        sorted_matrix = sort_counts_1qb(matrix)
        probs = sorted_matrix[:, 1] / np.sum(sorted_matrix[:, 1])
        sorted_matrix = np.column_stack((sorted_matrix, probs))

        k = k + 1
        j = j + 1

        if j == 1:
            sorted_matrix = np.column_stack((sorted_matrix, expectation_signs[:, 0]))
        elif j == 2:
            sorted_matrix = np.column_stack((sorted_matrix, expectation_signs[:, 1]))
        elif j == 3:
            sorted_matrix = np.column_stack((sorted_matrix, expectation_signs[:, 1]))
        elif j == 4:
            sorted_matrix = np.column_stack((sorted_matrix, expectation_signs[:, 1]))
        elif j == 5:
            sorted_matrix = np.column_stack((sorted_matrix, expectation_signs[:, 2]))
        elif j == 9:
            sorted_matrix = np.column_stack((sorted_matrix, expectation_signs[:, 2]))
        elif j == 13:
            sorted_matrix = np.column_stack((sorted_matrix, expectation_signs[:, 2]))
        else:
            sorted_matrix = np.column_stack((sorted_matrix, expectation_signs[:, 3]))

        temp = np.array(sorted_matrix[:, 2]) * np.array(sorted_matrix[:, 3])
        sorted_matrix = np.column_stack((sorted_matrix, temp))

        expectation_val = np.sum(temp)

        return expectation_val

    nqubits = 1
    gjk_1qb = np.zeros((4**nqubits, 4**nqubits))
    for k in range(0, 4):
        if k == 0:  # |0>
            circ = Circuit(nqubits, density_matrix=True)

        elif k == 1:  # |1>
            circ = Circuit(nqubits, density_matrix=True)
            circ.add(gates.X(0))

        elif k == 2:  # |+>
            circ = Circuit(nqubits, density_matrix=True)
            circ.add(gates.H(0))

        elif k == 3:  # |y+>
            circ = Circuit(nqubits, density_matrix=True)
            circ.add(gates.H(0))
            circ.add(gates.S(0))

        # NO INPUT GATE

        initial_state = np.zeros(2**nqubits)
        initial_state[0] = 1
        # ================= START OF 16 MEASUREMENT BASES =================|||
        for j in range(0, 4):
            # print('GST 1qb blank circ: initial state k=%d, measurement basis %d' %(k,j))

            if j == 0:  # Identity basis
                circ0 = Circuit(nqubits, density_matrix=True)
                NC = circ + circ0
                if noise_model is not None and backend.name != "qibolab":
                    NC = noise_model.apply(NC)
                NC.add(gates.M(0))

                expectation_val = GST_measure_1qb(NC, k, j)

            elif j == 1:  # X basis
                circ1 = Circuit(nqubits, density_matrix=True)
                NC = circ + circ1
                NC.add(gates.H(0))
                if noise_model is not None and backend.name != "qibolab":
                    NC = noise_model.apply(NC)
                NC.add(gates.M(0))

                expectation_val = GST_measure_1qb(NC, k, j)

            elif j == 2:  # Y basis
                circ2 = Circuit(nqubits, density_matrix=True)
                NC = circ + circ2
                NC.add(gates.SDG(0))
                NC.add(gates.H(0))
                if noise_model is not None and backend.name != "qibolab":
                    NC = noise_model.apply(NC)
                NC.add(gates.M(0))

                expectation_val = GST_measure_1qb(NC, k, j)

            elif j == 3:  # Z basis
                circ3 = Circuit(nqubits, density_matrix=True)
                NC = circ + circ3
                if noise_model is not None and backend.name != "qibolab":
                    NC = noise_model.apply(NC)
                NC.add(gates.M(0))

                expectation_val = GST_measure_1qb(NC, k, j)

            gjk_1qb[j, k] = expectation_val

        # ================= END OF 4 MEASUREMENT BASES ===================|||
    # =================================================================================================================

    gjk_1qb = np.matrix(gjk_1qb)
    pretty_print_matrix(gjk_1qb)
    print("1 register blank circuit tomography done. ")

    #######################################
    ### Two registers blank circuit GST ###
    #######################################

    if type_of_gates["2 qb gate"] >= 1:

        def GST_measure_2qb(circuit, k, j):
            idx_basis = [
                "II",
                "IX",
                "IY",
                "IZ",
                "XI",
                "XX",
                "XY",
                "XZ",
                "YI",
                "YX",
                "YY",
                "YZ",
                "ZI",
                "ZX",
                "ZY",
                "ZZ",
            ]
            filedirectory = "GST_nogates_qibo_2qb/"

            result = circuit.execute(nshots=NshotsGST)
            counts = dict(result.frequencies(binary=True))

            # Save counts into a matrix. 0th column: word; 1st column: count value
            # --------------------------------------------------------------------
            fout = filedirectory + "No gates (k=%.2d,j=%.2d) %s_basis.txt" % (
                k,
                j,
                idx_basis[j],
            )
            fo = open(fout, "w")
            for v1, v2 in counts.items():
                fo.write(str(v1) + "  " + str(v2) + "\n")
            fo.close()

            # Find value for gjk matrix
            # -------------------------

            expectation_signs = np.matrix(
                [[1, 1, 1, 1], [1, -1, 1, -1], [1, 1, -1, -1], [1, -1, -1, 1]]
            )

            matrix = np.zeros((2**2, 2))
            for ii in range(0, 2**2):
                word = bin(ii)[2:]
                matrix[ii, 0] = word

            for v1, v2 in counts.items():
                # print(v1, v2)
                row = int(v1, 2)
                col = v2
                matrix[row, 1] = col

            sorted_matrix = sort_counts_2qb(matrix)
            probs = sorted_matrix[:, 1] / np.sum(sorted_matrix[:, 1])
            sorted_matrix = np.column_stack((sorted_matrix, probs))

            k = k + 1
            j = j + 1

            if j == 1:
                sorted_matrix = np.column_stack(
                    (sorted_matrix, expectation_signs[:, 0])
                )
            elif j == 2:
                sorted_matrix = np.column_stack(
                    (sorted_matrix, expectation_signs[:, 1])
                )
            elif j == 3:
                sorted_matrix = np.column_stack(
                    (sorted_matrix, expectation_signs[:, 1])
                )
            elif j == 4:
                sorted_matrix = np.column_stack(
                    (sorted_matrix, expectation_signs[:, 1])
                )
            elif j == 5:
                sorted_matrix = np.column_stack(
                    (sorted_matrix, expectation_signs[:, 2])
                )
            elif j == 9:
                sorted_matrix = np.column_stack(
                    (sorted_matrix, expectation_signs[:, 2])
                )
            elif j == 13:
                sorted_matrix = np.column_stack(
                    (sorted_matrix, expectation_signs[:, 2])
                )
            else:
                sorted_matrix = np.column_stack(
                    (sorted_matrix, expectation_signs[:, 3])
                )

            temp = np.array(sorted_matrix[:, 2]) * np.array(sorted_matrix[:, 3])
            sorted_matrix = np.column_stack((sorted_matrix, temp))

            expectation_val = np.sum(temp)

            return expectation_val

        nqubits = 2
        gjk_2qb = np.zeros((4**nqubits, 4**nqubits))
        for k in range(0, 16):
            if k == 0:  # |0> |0>
                circ = Circuit(nqubits, density_matrix=True)

            elif k == 1:  # |0> |1>
                circ = Circuit(nqubits, density_matrix=True)
                circ.add(gates.X(1))

            elif k == 2:  # |0> |+>
                circ = Circuit(nqubits, density_matrix=True)
                circ.add(gates.H(1))

            elif k == 3:  # |0> |y+>
                circ = Circuit(nqubits, density_matrix=True)
                circ.add(gates.H(1))
                circ.add(gates.S(1))
            # --------------------------------------------

            elif k == 4:  # |1> |0>
                circ = Circuit(nqubits, density_matrix=True)
                circ.add(gates.X(0))

            elif k == 5:  # |1> |1>
                circ = Circuit(nqubits, density_matrix=True)
                circ.add(gates.X(0))
                circ.add(gates.X(1))

            elif k == 6:  # |1> |+>
                circ = Circuit(nqubits, density_matrix=True)
                circ.add(gates.X(0))
                circ.add(gates.H(1))

            elif k == 7:  # |1> |y+>
                circ = Circuit(nqubits, density_matrix=True)
                circ.add(gates.X(0))
                circ.add(gates.H(1))
                circ.add(gates.S(1))

            # --------------------------------------------

            elif k == 8:  # |+> |0>
                circ = Circuit(nqubits, density_matrix=True)
                circ.add(gates.H(0))

            elif k == 9:  # |+> |1>
                circ = Circuit(nqubits, density_matrix=True)
                circ.add(gates.H(0))
                circ.add(gates.X(1))

            elif k == 10:  # |+> |+>
                circ = Circuit(nqubits, density_matrix=True)
                circ.add(gates.H(0))
                circ.add(gates.H(1))

            elif k == 11:  # |+> |y+>
                circ = Circuit(nqubits, density_matrix=True)
                circ.add(gates.H(0))
                circ.add(gates.H(1))
                circ.add(gates.S(1))

            # --------------------------------------------

            elif k == 12:  # |y+> |0>
                circ = Circuit(nqubits, density_matrix=True)
                circ.add(gates.H(0))
                circ.add(gates.S(0))

            elif k == 13:  # |y+> |1>
                circ = Circuit(nqubits, density_matrix=True)
                circ.add(gates.H(0))
                circ.add(gates.S(0))
                circ.add(gates.X(1))

            elif k == 14:  # |y+> |+>
                circ = Circuit(nqubits, density_matrix=True)
                circ.add(gates.H(0))
                circ.add(gates.S(0))
                circ.add(gates.H(1))

            elif k == 15:  # |y+> |y+>
                circ = Circuit(nqubits, density_matrix=True)
                circ.add(gates.H(0))
                circ.add(gates.S(0))
                circ.add(gates.H(1))
                circ.add(gates.S(1))

            # --------------------------------------------

            # NO INPUT GATE

            initial_state = np.zeros(2**nqubits)
            initial_state[0] = 1
            # ================= START OF 16 MEASUREMENT BASES =================|||
            for j in range(0, 16):
                # print('GST 2qb blank circ: initial state k=%d, measurement basis %d' %(k,j))

                if j == 0:  # Identity basis  # Identity basis
                    circ0 = Circuit(nqubits, density_matrix=True)
                    NC = circ + circ0
                    if noise_model is not None and backend.name != "qibolab":
                        NC = noise_model.apply(NC)
                    NC.add(gates.M(0))
                    NC.add(gates.M(1))

                    expectation_val = GST_measure_2qb(NC, k, j)

                elif j == 1:  # Identity basis # X basis
                    circ1 = Circuit(nqubits, density_matrix=True)
                    NC = circ + circ1
                    NC.add(gates.H(1))
                    if noise_model is not None and backend.name != "qibolab":
                        NC = noise_model.apply(NC)
                    NC.add(gates.M(0))
                    NC.add(gates.M(1))

                    expectation_val = GST_measure_2qb(NC, k, j)

                elif j == 2:  # Identity basis # Y basis
                    circ2 = Circuit(nqubits, density_matrix=True)
                    NC = circ + circ2
                    NC.add(gates.SDG(1))
                    NC.add(gates.H(1))
                    if noise_model is not None and backend.name != "qibolab":
                        NC = noise_model.apply(NC)
                    NC.add(gates.M(0))
                    NC.add(gates.M(1))

                    expectation_val = GST_measure_2qb(NC, k, j)

                elif j == 3:  # Identity basis # Z basis
                    circ3 = Circuit(nqubits, density_matrix=True)
                    NC = circ + circ3
                    if noise_model is not None and backend.name != "qibolab":
                        NC = noise_model.apply(NC)
                    NC.add(gates.M(0))
                    NC.add(gates.M(1))

                    expectation_val = GST_measure_2qb(NC, k, j)

                # =================================================================================================================

                elif j == 4:  # X basis# Identity basis
                    circ4 = Circuit(nqubits, density_matrix=True)
                    NC = circ + circ4
                    NC.add(gates.H(0))
                    if noise_model is not None and backend.name != "qibolab":
                        NC = noise_model.apply(NC)
                    NC.add(gates.M(0))
                    NC.add(gates.M(1))

                    expectation_val = GST_measure_2qb(NC, k, j)

                elif j == 5:  # X basis # X basis
                    circ5 = Circuit(nqubits, density_matrix=True)
                    NC = circ + circ5
                    NC.add(gates.H(0))
                    NC.add(gates.H(1))
                    if noise_model is not None and backend.name != "qibolab":
                        NC = noise_model.apply(NC)
                    NC.add(gates.M(0))
                    NC.add(gates.M(1))

                    expectation_val = GST_measure_2qb(NC, k, j)

                elif j == 6:  # X basis # Y basis
                    circ6 = Circuit(nqubits, density_matrix=True)
                    NC = circ + circ6
                    NC.add(gates.H(0))
                    NC.add(gates.SDG(1))
                    NC.add(gates.H(1))
                    if noise_model is not None and backend.name != "qibolab":
                        NC = noise_model.apply(NC)
                    NC.add(gates.M(0))
                    NC.add(gates.M(1))

                    expectation_val = GST_measure_2qb(NC, k, j)

                elif j == 7:  # X basis  # Z basis
                    circ7 = Circuit(nqubits, density_matrix=True)
                    NC = circ + circ7
                    NC.add(gates.H(0))
                    if noise_model is not None and backend.name != "qibolab":
                        NC = noise_model.apply(NC)
                    NC.add(gates.M(0))
                    NC.add(gates.M(1))

                    expectation_val = GST_measure_2qb(NC, k, j)

                # =================================================================================================================

                elif j == 8:  # Y basis # Identity basis
                    circ8 = Circuit(nqubits, density_matrix=True)
                    NC = circ + circ8
                    NC.add(gates.SDG(0))
                    NC.add(gates.H(0))
                    if noise_model is not None and backend.name != "qibolab":
                        NC = noise_model.apply(NC)
                    NC.add(gates.M(0))
                    NC.add(gates.M(1))

                    expectation_val = GST_measure_2qb(NC, k, j)

                elif j == 9:  # Y basis  # X basis
                    circ9 = Circuit(nqubits, density_matrix=True)
                    NC = circ + circ9
                    NC.add(gates.SDG(0))
                    NC.add(gates.H(0))
                    NC.add(gates.H(1))
                    if noise_model is not None and backend.name != "qibolab":
                        NC = noise_model.apply(NC)
                    NC.add(gates.M(0))
                    NC.add(gates.M(1))

                    expectation_val = GST_measure_2qb(NC, k, j)

                elif j == 10:  # Y basis  # Y basis
                    circ10 = Circuit(nqubits, density_matrix=True)
                    NC = circ + circ10
                    NC.add(gates.SDG(0))
                    NC.add(gates.H(0))
                    NC.add(gates.SDG(1))
                    NC.add(gates.H(1))
                    if noise_model is not None and backend.name != "qibolab":
                        NC = noise_model.apply(NC)
                    NC.add(gates.M(0))
                    NC.add(gates.M(1))

                    expectation_val = GST_measure_2qb(NC, k, j)

                elif j == 11:  # Y basis  # Z basis
                    circ11 = Circuit(nqubits, density_matrix=True)
                    NC = circ + circ11
                    NC.add(gates.SDG(0))
                    NC.add(gates.H(0))
                    if noise_model is not None and backend.name != "qibolab":
                        NC = noise_model.apply(NC)
                    NC.add(gates.M(0))
                    NC.add(gates.M(1))

                    expectation_val = GST_measure_2qb(NC, k, j)

                # =================================================================================================================

                elif j == 12:  # Z basis# Identity basis
                    circ12 = Circuit(nqubits, density_matrix=True)
                    NC = circ + circ12
                    if noise_model is not None and backend.name != "qibolab":
                        NC = noise_model.apply(NC)
                    NC.add(gates.M(0))
                    NC.add(gates.M(1))

                    expectation_val = GST_measure_2qb(NC, k, j)

                elif j == 13:  # Z basis  # X basis
                    circ13 = Circuit(nqubits, density_matrix=True)
                    NC = circ + circ13
                    NC.add(gates.H(1))
                    if noise_model is not None and backend.name != "qibolab":
                        NC = noise_model.apply(NC)
                    NC.add(gates.M(0))
                    NC.add(gates.M(1))

                    expectation_val = GST_measure_2qb(NC, k, j)

                elif j == 14:  # Z basis  # Y basis
                    circ14 = Circuit(nqubits, density_matrix=True)
                    NC = circ + circ14
                    NC.add(gates.SDG(1))
                    NC.add(gates.H(1))
                    if noise_model is not None and backend.name != "qibolab":
                        NC = noise_model.apply(NC)
                    NC.add(gates.M(0))
                    NC.add(gates.M(1))

                    expectation_val = GST_measure_2qb(NC, k, j)

                elif j == 15:  # Z basis # Z basis
                    circ15 = Circuit(nqubits, density_matrix=True)
                    NC = circ + circ15
                    if noise_model is not None and backend.name != "qibolab":
                        NC = noise_model.apply(NC)
                    NC.add(gates.M(0))
                    NC.add(gates.M(1))

                    expectation_val = GST_measure_2qb(NC, k, j)

                # print(expectation_val)
                gjk_2qb[j, k] = expectation_val

            # ================= END OF 16 MEASUREMENT BASES ===================|||
        # =================================================================================================================

        gjk_2qb = np.matrix(gjk_2qb)
        pretty_print_matrix(gjk_2qb)
        print("2 registers blank circuit tomography done. ")

    Bjk_tilde_1qb = np.zeros((13, 4, 4))
    Bjk_tilde_2qb = np.zeros((241, 16, 16))

    ##################################################################
    ### 13 BASIS OPERATIONS (To also use for 169 Basis Operations) ###
    ##################################################################

    BasisOps_13 = [
        np.matrix([[1, 0], [0, 1]]),
        np.matrix([[0, 1], [1, 0]]),
        np.matrix([[0, -1j], [1j, 0]]),
        np.matrix([[1, 0], [0, -1]]),
        np.matrix(
            [
                [1 / np.sqrt(2) * 1, 1 / np.sqrt(2) * 1j],
                [1 / np.sqrt(2) * 1j, 1 / np.sqrt(2) * 1],
            ]
        ),
        np.matrix(
            [
                [1 / np.sqrt(2) * 1, 1 / np.sqrt(2) * 1],
                [1 / np.sqrt(2) * (-1), 1 / np.sqrt(2) * 1],
            ]
        ),
        np.matrix([[1 / np.sqrt(2) * (1 + 1j), 0], [0, 1 / np.sqrt(2) * (1 - 1j)]]),
        np.matrix(
            [
                [1 / np.sqrt(2) * 1, 1 / np.sqrt(2) * (-1j)],
                [1 / np.sqrt(2) * 1j, 1 / np.sqrt(2) * (-1)],
            ]
        ),
        np.matrix(
            [
                [1 / np.sqrt(2) * 1, 1 / np.sqrt(2) * 1],
                [1 / np.sqrt(2) * 1, 1 / np.sqrt(2) * (-1)],
            ]
        ),
        np.matrix([[0, 1 / np.sqrt(2) * (1 - 1j)], [1 / np.sqrt(2) * (1 + 1j), 0]]),
        np.matrix([[1, 0], [0, 0]]),
        np.matrix([[0.5, 0.5], [0.5, 0.5]]),
        np.matrix([[0.5, -0.5j], [0.5j, 0.5]]),
    ]

    BasisOps_169 = BasisOps_13

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

    #########################################
    ### THE REMAINING 72 BASIS OPERATIONS ###
    #########################################

    BasisOps_241 = BO_2qubits_hard

    ################################
    ### Setting up the functions ###
    ################################

    def GST_measure_13(circuit, idx_gatenames_1, k, j):
        nqubits = 1
        idx_basis = ["I", "X", "Y", "Z"]
        filedirectory = "GST_basisops_qibo_1qb/"

        result = circuit.execute(nshots=NshotsGST)
        counts = dict(result.frequencies(binary=True))

        # Save counts into a matrix. 0th column: word; 1st column: count value
        # --------------------------------------------------------------------
        fout = filedirectory + "Q_1_BasisOp%.2d (k=%.2d,j=%.2d) %s_basis.txt" % (
            idx_gatenames_1,
            k,
            j,
            idx_basis[j],
        )
        fo = open(fout, "w")
        for v1, v2 in counts.items():
            fo.write(str(v1) + "  " + str(v2) + "\n")
        fo.close()

        # Find value for gjk matrix
        # -------------------------

        expectation_signs = np.matrix([[1, 1, 1, 1], [1, -1, -1, -1]])

        matrix = np.zeros((2**nqubits, 2**nqubits))
        for ii in range(0, 2):
            word = bin(ii)[2:]
            matrix[ii, 0] = word

        for v1, v2 in counts.items():
            # print(v1, v2)
            row = int(v1, 2)
            col = v2
            matrix[row, 1] = col

        sorted_matrix = sort_counts_1qb(matrix)
        probs = sorted_matrix[:, 1] / np.sum(sorted_matrix[:, 1])
        sorted_matrix = np.column_stack((sorted_matrix, probs))

        k = k + 1
        j = j + 1

        if j == 1:
            sorted_matrix = np.column_stack((sorted_matrix, expectation_signs[:, 0]))
        elif j == 2:
            sorted_matrix = np.column_stack((sorted_matrix, expectation_signs[:, 1]))
        elif j == 3:
            sorted_matrix = np.column_stack((sorted_matrix, expectation_signs[:, 1]))
        elif j == 4:
            sorted_matrix = np.column_stack((sorted_matrix, expectation_signs[:, 1]))
        elif j == 5:
            sorted_matrix = np.column_stack((sorted_matrix, expectation_signs[:, 2]))
        elif j == 9:
            sorted_matrix = np.column_stack((sorted_matrix, expectation_signs[:, 2]))
        elif j == 13:
            sorted_matrix = np.column_stack((sorted_matrix, expectation_signs[:, 2]))
        else:
            sorted_matrix = np.column_stack((sorted_matrix, expectation_signs[:, 3]))

        temp = np.array(sorted_matrix[:, 2]) * np.array(sorted_matrix[:, 3])
        sorted_matrix = np.column_stack((sorted_matrix, temp))

        expectation_val = np.sum(temp)

        return expectation_val

    def GSTBasisOpMeasurements_13(circ):
        nqubits = 1
        initial_state = np.zeros(2**nqubits)
        initial_state[0] = 1

        column_of_js = np.zeros((4**nqubits, 1))

        # ================= START OF 16 MEASUREMENT BASES =================|||
        for j in range(0, 4**nqubits):
            # print('[def GSTBasisOpMeasurements]: initial state k=%d, idx_gatenames_1 = %d, idx_gatenames_2 = %d, measurement basis %d' %(k,idx_gatenames_1,idx_gatenames_2,j))

            if j == 0:  # Identity basis
                circ0 = Circuit(nqubits, density_matrix=True)
                NC = circ + circ0
                if noise_model is not None and backend.name != "qibolab":
                    NC = noise_model.apply(NC)
                NC.add(gates.M(0))

                expectation_val = GST_measure_13(NC, idx_gatenames_1, k, j)

            elif j == 1:  # X basis
                circ1 = Circuit(nqubits, density_matrix=True)
                NC = circ + circ1
                NC.add(gates.H(0))
                if noise_model is not None and backend.name != "qibolab":
                    NC = noise_model.apply(NC)
                NC.add(gates.M(0))

                expectation_val = GST_measure_13(NC, idx_gatenames_1, k, j)

            elif j == 2:  # Y basis
                circ2 = Circuit(nqubits, density_matrix=True)
                NC = circ + circ2
                NC.add(gates.SDG(0))
                NC.add(gates.H(0))
                if noise_model is not None and backend.name != "qibolab":
                    NC = noise_model.apply(NC)
                NC.add(gates.M(0))

                expectation_val = GST_measure_13(NC, idx_gatenames_1, k, j)

            elif j == 3:  # Z basis
                circ3 = Circuit(nqubits, density_matrix=True)
                NC = circ + circ3
                if noise_model is not None and backend.name != "qibolab":
                    NC = noise_model.apply(NC)
                NC.add(gates.M(0))

                expectation_val = GST_measure_13(NC, idx_gatenames_1, k, j)

            column_of_js[j, :] = expectation_val

        return column_of_js

        # ================= END OF 4 MEASUREMENT BASES ===================|||

    def GST_measure_169(circuit, idx_gatenames_1, idx_gatenames_2, k, j):
        nqubits = 2
        idx_basis = [
            "II",
            "IX",
            "IY",
            "IZ",
            "XI",
            "XX",
            "XY",
            "XZ",
            "YI",
            "YX",
            "YY",
            "YZ",
            "ZI",
            "ZX",
            "ZY",
            "ZZ",
        ]
        filedirectory = "GST_basisops_qibo_2qb/"

        result = circuit.execute(nshots=NshotsGST)
        counts = dict(result.frequencies(binary=True))

        # Save counts into a matrix. 0th column: word; 1st column: count value
        # --------------------------------------------------------------------
        fout = (
            filedirectory
            + "Q_1_BasisOp%.2d Q_2_BasisOp%.2d (k=%.2d,j=%.2d) %s_basis.txt"
            % (idx_gatenames_1, idx_gatenames_2, k, j, idx_basis[j])
        )
        fo = open(fout, "w")
        for v1, v2 in counts.items():
            fo.write(str(v1) + "  " + str(v2) + "\n")
        fo.close()

        # Find value for gjk matrix
        # -------------------------

        expectation_signs = np.matrix(
            [[1, 1, 1, 1], [1, -1, 1, -1], [1, 1, -1, -1], [1, -1, -1, 1]]
        )

        matrix = np.zeros((2**nqubits, 2))
        for ii in range(0, 2**nqubits):
            word = bin(ii)[2:]
            matrix[ii, 0] = word

        for v1, v2 in counts.items():
            # print(v1, v2)
            row = int(v1, 2)
            col = v2
            matrix[row, 1] = col

        sorted_matrix = sort_counts_2qb(matrix)
        probs = sorted_matrix[:, 1] / np.sum(sorted_matrix[:, 1])
        sorted_matrix = np.column_stack((sorted_matrix, probs))

        k = k + 1
        j = j + 1

        if j == 1:
            sorted_matrix = np.column_stack((sorted_matrix, expectation_signs[:, 0]))
        elif j == 2:
            sorted_matrix = np.column_stack((sorted_matrix, expectation_signs[:, 1]))
        elif j == 3:
            sorted_matrix = np.column_stack((sorted_matrix, expectation_signs[:, 1]))
        elif j == 4:
            sorted_matrix = np.column_stack((sorted_matrix, expectation_signs[:, 1]))
        elif j == 5:
            sorted_matrix = np.column_stack((sorted_matrix, expectation_signs[:, 2]))
        elif j == 9:
            sorted_matrix = np.column_stack((sorted_matrix, expectation_signs[:, 2]))
        elif j == 13:
            sorted_matrix = np.column_stack((sorted_matrix, expectation_signs[:, 2]))
        else:
            sorted_matrix = np.column_stack((sorted_matrix, expectation_signs[:, 3]))

        temp = np.array(sorted_matrix[:, 2]) * np.array(sorted_matrix[:, 3])
        sorted_matrix = np.column_stack((sorted_matrix, temp))

        expectation_val = np.sum(temp)

        return expectation_val

    def GSTBasisOpMeasurements_169(circ):
        nqubits = 2
        initial_state = np.zeros(2**nqubits)
        initial_state[0] = 1

        column_of_js = np.zeros((4**nqubits, 1))

        # ================= START OF 16 MEASUREMENT BASES =================|||
        for j in range(0, 4**nqubits):
            # print('[def GSTBasisOpMeasurements]: initial state k=%d, idx_gatenames_1 = %d, idx_gatenames_2 = %d, measurement basis %d' %(k,idx_gatenames_1,idx_gatenames_2,j))

            if j == 0:  # Identity basis  # Identity basis
                circ0 = Circuit(nqubits, density_matrix=True)
                NC = circ + circ0
                if noise_model is not None and backend.name != "qibolab":
                    NC = noise_model.apply(NC)
                NC.add(gates.M(0))
                NC.add(gates.M(1))

                expectation_val = GST_measure_169(
                    NC, idx_gatenames_1, idx_gatenames_2, k, j
                )

            elif j == 1:  # Identity basis # X basis
                circ1 = Circuit(nqubits, density_matrix=True)
                NC = circ + circ1
                NC.add(gates.H(1))
                if noise_model is not None and backend.name != "qibolab":
                    NC = noise_model.apply(NC)
                NC.add(gates.M(0))
                NC.add(gates.M(1))

                expectation_val = GST_measure_169(
                    NC, idx_gatenames_1, idx_gatenames_2, k, j
                )

            elif j == 2:  # Identity basis # Y basis
                circ2 = Circuit(nqubits, density_matrix=True)
                NC = circ + circ2
                NC.add(gates.SDG(1))
                NC.add(gates.H(1))
                if noise_model is not None and backend.name != "qibolab":
                    NC = noise_model.apply(NC)
                NC.add(gates.M(0))
                NC.add(gates.M(1))

                expectation_val = GST_measure_169(
                    NC, idx_gatenames_1, idx_gatenames_2, k, j
                )

            elif j == 3:  # Identity basis # Z basis
                circ3 = Circuit(nqubits, density_matrix=True)
                NC = circ + circ3
                if noise_model is not None and backend.name != "qibolab":
                    NC = noise_model.apply(NC)
                NC.add(gates.M(0))
                NC.add(gates.M(1))

                expectation_val = GST_measure_169(
                    NC, idx_gatenames_1, idx_gatenames_2, k, j
                )

            # =================================================================================================================

            elif j == 4:  # X basis# Identity basis
                circ4 = Circuit(nqubits, density_matrix=True)
                NC = circ + circ4
                NC.add(gates.H(0))
                if noise_model is not None and backend.name != "qibolab":
                    NC = noise_model.apply(NC)
                NC.add(gates.M(0))
                NC.add(gates.M(1))

                expectation_val = GST_measure_169(
                    NC, idx_gatenames_1, idx_gatenames_2, k, j
                )

            elif j == 5:  # X basis # X basis
                circ5 = Circuit(nqubits, density_matrix=True)
                NC = circ + circ5
                NC.add(gates.H(0))
                NC.add(gates.H(1))
                if noise_model is not None and backend.name != "qibolab":
                    NC = noise_model.apply(NC)
                NC.add(gates.M(0))
                NC.add(gates.M(1))

                expectation_val = GST_measure_169(
                    NC, idx_gatenames_1, idx_gatenames_2, k, j
                )

            elif j == 6:  # X basis # Y basis
                circ6 = Circuit(nqubits, density_matrix=True)
                NC = circ + circ6
                NC.add(gates.H(0))
                NC.add(gates.SDG(1))
                NC.add(gates.H(1))
                if noise_model is not None and backend.name != "qibolab":
                    NC = noise_model.apply(NC)
                NC.add(gates.M(0))
                NC.add(gates.M(1))

                expectation_val = GST_measure_169(
                    NC, idx_gatenames_1, idx_gatenames_2, k, j
                )

            elif j == 7:  # X basis  # Z basis
                circ7 = Circuit(nqubits, density_matrix=True)
                NC = circ + circ7
                NC.add(gates.H(0))
                if noise_model is not None and backend.name != "qibolab":
                    NC = noise_model.apply(NC)
                NC.add(gates.M(0))
                NC.add(gates.M(1))

                expectation_val = GST_measure_169(
                    NC, idx_gatenames_1, idx_gatenames_2, k, j
                )

            # =================================================================================================================

            elif j == 8:  # Y basis # Identity basis
                circ8 = Circuit(nqubits, density_matrix=True)
                NC = circ + circ8
                NC.add(gates.SDG(0))
                NC.add(gates.H(0))
                if noise_model is not None and backend.name != "qibolab":
                    NC = noise_model.apply(NC)
                NC.add(gates.M(0))
                NC.add(gates.M(1))

                expectation_val = GST_measure_169(
                    NC, idx_gatenames_1, idx_gatenames_2, k, j
                )

            elif j == 9:  # Y basis  # X basis
                circ9 = Circuit(nqubits, density_matrix=True)
                NC = circ + circ9
                NC.add(gates.SDG(0))
                NC.add(gates.H(0))
                NC.add(gates.H(1))
                if noise_model is not None and backend.name != "qibolab":
                    NC = noise_model.apply(NC)
                NC.add(gates.M(0))
                NC.add(gates.M(1))

                expectation_val = GST_measure_169(
                    NC, idx_gatenames_1, idx_gatenames_2, k, j
                )

            elif j == 10:  # Y basis  # Y basis
                circ10 = Circuit(nqubits, density_matrix=True)
                NC = circ + circ10
                NC.add(gates.SDG(0))
                NC.add(gates.H(0))
                NC.add(gates.SDG(1))
                NC.add(gates.H(1))
                if noise_model is not None and backend.name != "qibolab":
                    NC = noise_model.apply(NC)
                NC.add(gates.M(0))
                NC.add(gates.M(1))

                expectation_val = GST_measure_169(
                    NC, idx_gatenames_1, idx_gatenames_2, k, j
                )

            elif j == 11:  # Y basis  # Z basis
                circ11 = Circuit(nqubits, density_matrix=True)
                NC = circ + circ11
                NC.add(gates.SDG(0))
                NC.add(gates.H(0))
                if noise_model is not None and backend.name != "qibolab":
                    NC = noise_model.apply(NC)
                NC.add(gates.M(0))
                NC.add(gates.M(1))

                expectation_val = GST_measure_169(
                    NC, idx_gatenames_1, idx_gatenames_2, k, j
                )

            # =================================================================================================================

            elif j == 12:  # Z basis# Identity basis
                circ12 = Circuit(nqubits, density_matrix=True)
                NC = circ + circ12
                if noise_model is not None and backend.name != "qibolab":
                    NC = noise_model.apply(NC)
                NC.add(gates.M(0))
                NC.add(gates.M(1))

                expectation_val = GST_measure_169(
                    NC, idx_gatenames_1, idx_gatenames_2, k, j
                )

            elif j == 13:  # Z basis  # X basis
                circ13 = Circuit(nqubits, density_matrix=True)
                NC = circ + circ13
                NC.add(gates.H(1))
                if noise_model is not None and backend.name != "qibolab":
                    NC = noise_model.apply(NC)
                NC.add(gates.M(0))
                NC.add(gates.M(1))

                expectation_val = GST_measure_169(
                    NC, idx_gatenames_1, idx_gatenames_2, k, j
                )

            elif j == 14:  # Z basis  # Y basis
                circ14 = Circuit(nqubits, density_matrix=True)
                NC = circ + circ14
                NC.add(gates.SDG(1))
                NC.add(gates.H(1))
                if noise_model is not None and backend.name != "qibolab":
                    NC = noise_model.apply(NC)
                NC.add(gates.M(0))
                NC.add(gates.M(1))

                expectation_val = GST_measure_169(
                    NC, idx_gatenames_1, idx_gatenames_2, k, j
                )

            elif j == 15:  # Z basis # Z basis
                circ15 = Circuit(nqubits, density_matrix=True)
                NC = circ + circ15
                if noise_model is not None and backend.name != "qibolab":
                    NC = noise_model.apply(NC)
                NC.add(gates.M(0))
                NC.add(gates.M(1))

                expectation_val = GST_measure_169(
                    NC, idx_gatenames_1, idx_gatenames_2, k, j
                )

            column_of_js[j, :] = expectation_val

        return column_of_js

        # ================= END OF 16 MEASUREMENT BASES ===================|||

    def GST_measure_72(circuit, idx_basis_ops_72, k, j):
        nqubits = 2
        idx_basis = [
            "II",
            "IX",
            "IY",
            "IZ",
            "XI",
            "XX",
            "XY",
            "XZ",
            "YI",
            "YX",
            "YY",
            "YZ",
            "ZI",
            "ZX",
            "ZY",
            "ZZ",
        ]
        filedirectory = "GST_basisops_qibo_2qb/"

        result = circuit.execute(nshots=NshotsGST)
        counts = dict(result.frequencies(binary=True))

        # Save counts into a matrix. 0th column: word; 1st column: count value
        # --------------------------------------------------------------------
        fout = filedirectory + "Q_1_Q_2_BasisOp%.2d (k=%.2d,j=%.2d) II_basis.txt" % (
            idx_basis_ops_72,
            k,
            j,
        )
        fo = open(fout, "w")
        for v1, v2 in counts.items():
            fo.write(str(v1) + "  " + str(v2) + "\n")
        fo.close()

        # Find value for gjk matrix
        # -------------------------

        expectation_signs = np.matrix(
            [[1, 1, 1, 1], [1, -1, 1, -1], [1, 1, -1, -1], [1, -1, -1, 1]]
        )

        matrix = np.zeros((2**nqubits, 2))
        for ii in range(0, 2**nqubits):
            word = bin(ii)[2:]
            matrix[ii, 0] = word

        for v1, v2 in counts.items():
            # print(v1, v2)
            row = int(v1, 2)
            col = v2
            matrix[row, 1] = col

        sorted_matrix = sort_counts_2qb(matrix)
        probs = sorted_matrix[:, 1] / np.sum(sorted_matrix[:, 1])
        sorted_matrix = np.column_stack((sorted_matrix, probs))

        k = k + 1
        j = j + 1

        if j == 1:
            sorted_matrix = np.column_stack((sorted_matrix, expectation_signs[:, 0]))
        elif j == 2:
            sorted_matrix = np.column_stack((sorted_matrix, expectation_signs[:, 1]))
        elif j == 3:
            sorted_matrix = np.column_stack((sorted_matrix, expectation_signs[:, 1]))
        elif j == 4:
            sorted_matrix = np.column_stack((sorted_matrix, expectation_signs[:, 1]))
        elif j == 5:
            sorted_matrix = np.column_stack((sorted_matrix, expectation_signs[:, 2]))
        elif j == 9:
            sorted_matrix = np.column_stack((sorted_matrix, expectation_signs[:, 2]))
        elif j == 13:
            sorted_matrix = np.column_stack((sorted_matrix, expectation_signs[:, 2]))
        else:
            sorted_matrix = np.column_stack((sorted_matrix, expectation_signs[:, 3]))

        temp = np.array(sorted_matrix[:, 2]) * np.array(sorted_matrix[:, 3])
        sorted_matrix = np.column_stack((sorted_matrix, temp))

        expectation_val = np.sum(temp)

        return expectation_val

    def GSTBasisOpMeasurements_72(circ, idx_basis_ops_72):
        nqubits = 2
        initial_state = np.zeros(2**nqubits)
        initial_state[0] = 1

        column_of_js = np.zeros((4**nqubits, 1))

        # ================= START OF 16 MEASUREMENT BASES =================|||
        for j in range(0, 4**nqubits):
            # print('[def GSTBasisOpMeasurements]: initial state k=%d, idx_gatenames = %d, measurement basis %d' %(k,idx_basis_ops_72,j))

            if j == 0:  # Identity basis  # Identity basis
                circ0 = Circuit(nqubits, density_matrix=True)
                NC = circ + circ0
                if noise_model is not None and backend.name != "qibolab":
                    NC = noise_model.apply(NC)
                NC.add(gates.M(0))
                NC.add(gates.M(1))

                expectation_val = GST_measure_72(NC, idx_basis_ops_72, k, j)

            elif j == 1:  # Identity basis # X basis
                circ1 = Circuit(nqubits, density_matrix=True)
                NC = circ + circ1
                NC.add(gates.H(1))
                if noise_model is not None and backend.name != "qibolab":
                    NC = noise_model.apply(NC)
                NC.add(gates.M(0))
                NC.add(gates.M(1))

                expectation_val = GST_measure_72(NC, idx_basis_ops_72, k, j)

            elif j == 2:  # Identity basis # Y basis
                circ2 = Circuit(nqubits, density_matrix=True)
                NC = circ + circ2
                NC.add(gates.SDG(1))
                NC.add(gates.H(1))
                if noise_model is not None and backend.name != "qibolab":
                    NC = noise_model.apply(NC)
                NC.add(gates.M(0))
                NC.add(gates.M(1))

                expectation_val = GST_measure_72(NC, idx_basis_ops_72, k, j)

            elif j == 3:  # Identity basis # Z basis
                circ3 = Circuit(nqubits, density_matrix=True)
                NC = circ + circ3
                if noise_model is not None and backend.name != "qibolab":
                    NC = noise_model.apply(NC)
                NC.add(gates.M(0))
                NC.add(gates.M(1))

                expectation_val = GST_measure_72(NC, idx_basis_ops_72, k, j)

            # =================================================================================================================

            elif j == 4:  # X basis# Identity basis
                circ4 = Circuit(nqubits, density_matrix=True)
                NC = circ + circ4
                NC.add(gates.H(0))
                if noise_model is not None and backend.name != "qibolab":
                    NC = noise_model.apply(NC)
                NC.add(gates.M(0))
                NC.add(gates.M(1))

                expectation_val = GST_measure_72(NC, idx_basis_ops_72, k, j)

            elif j == 5:  # X basis # X basis
                circ5 = Circuit(nqubits, density_matrix=True)
                NC = circ + circ5
                NC.add(gates.H(0))
                NC.add(gates.H(1))
                if noise_model is not None and backend.name != "qibolab":
                    NC = noise_model.apply(NC)
                NC.add(gates.M(0))
                NC.add(gates.M(1))

                expectation_val = GST_measure_72(NC, idx_basis_ops_72, k, j)

            elif j == 6:  # X basis # Y basis
                circ6 = Circuit(nqubits, density_matrix=True)
                NC = circ + circ6
                NC.add(gates.H(0))
                NC.add(gates.SDG(1))
                NC.add(gates.H(1))
                if noise_model is not None and backend.name != "qibolab":
                    NC = noise_model.apply(NC)
                NC.add(gates.M(0))
                NC.add(gates.M(1))

                expectation_val = GST_measure_72(NC, idx_basis_ops_72, k, j)

            elif j == 7:  # X basis  # Z basis
                circ7 = Circuit(nqubits, density_matrix=True)
                NC = circ + circ7
                NC.add(gates.H(0))
                if noise_model is not None and backend.name != "qibolab":
                    NC = noise_model.apply(NC)
                NC.add(gates.M(0))
                NC.add(gates.M(1))

                expectation_val = GST_measure_72(NC, idx_basis_ops_72, k, j)

            # =================================================================================================================

            elif j == 8:  # Y basis # Identity basis
                circ8 = Circuit(nqubits, density_matrix=True)
                NC = circ + circ8
                NC.add(gates.SDG(0))
                NC.add(gates.H(0))
                if noise_model is not None and backend.name != "qibolab":
                    NC = noise_model.apply(NC)
                NC.add(gates.M(0))
                NC.add(gates.M(1))

                expectation_val = GST_measure_72(NC, idx_basis_ops_72, k, j)

            elif j == 9:  # Y basis  # X basis
                circ9 = Circuit(nqubits, density_matrix=True)
                NC = circ + circ9
                NC.add(gates.SDG(0))
                NC.add(gates.H(0))
                NC.add(gates.H(1))
                if noise_model is not None and backend.name != "qibolab":
                    NC = noise_model.apply(NC)
                NC.add(gates.M(0))
                NC.add(gates.M(1))

                expectation_val = GST_measure_72(NC, idx_basis_ops_72, k, j)

            elif j == 10:  # Y basis  # Y basis
                circ10 = Circuit(nqubits, density_matrix=True)
                NC = circ + circ10
                NC.add(gates.SDG(0))
                NC.add(gates.H(0))
                NC.add(gates.SDG(1))
                NC.add(gates.H(1))
                if noise_model is not None and backend.name != "qibolab":
                    NC = noise_model.apply(NC)
                NC.add(gates.M(0))
                NC.add(gates.M(1))

                expectation_val = GST_measure_72(NC, idx_basis_ops_72, k, j)

            elif j == 11:  # Y basis  # Z basis
                circ11 = Circuit(nqubits, density_matrix=True)
                NC = circ + circ11
                NC.add(gates.SDG(0))
                NC.add(gates.H(0))
                if noise_model is not None and backend.name != "qibolab":
                    NC = noise_model.apply(NC)
                NC.add(gates.M(0))
                NC.add(gates.M(1))

                expectation_val = GST_measure_72(NC, idx_basis_ops_72, k, j)

            # =================================================================================================================

            elif j == 12:  # Z basis# Identity basis
                circ12 = Circuit(nqubits, density_matrix=True)
                NC = circ + circ12
                if noise_model is not None and backend.name != "qibolab":
                    NC = noise_model.apply(NC)
                NC.add(gates.M(0))
                NC.add(gates.M(1))

                expectation_val = GST_measure_72(NC, idx_basis_ops_72, k, j)

            elif j == 13:  # Z basis  # X basis
                circ13 = Circuit(nqubits, density_matrix=True)
                NC = circ + circ13
                NC.add(gates.H(1))
                if noise_model is not None and backend.name != "qibolab":
                    NC = noise_model.apply(NC)
                NC.add(gates.M(0))
                NC.add(gates.M(1))

                expectation_val = GST_measure_72(NC, idx_basis_ops_72, k, j)

            elif j == 14:  # Z basis  # Y basis
                circ14 = Circuit(nqubits, density_matrix=True)
                NC = circ + circ14
                NC.add(gates.SDG(1))
                NC.add(gates.H(1))
                if noise_model is not None and backend.name != "qibolab":
                    NC = noise_model.apply(NC)
                NC.add(gates.M(0))
                NC.add(gates.M(1))

                expectation_val = GST_measure_72(NC, idx_basis_ops_72, k, j)

            elif j == 15:  # Z basis # Z basis
                circ15 = Circuit(nqubits, density_matrix=True)
                NC = circ + circ15
                if noise_model is not None and backend.name != "qibolab":
                    NC = noise_model.apply(NC)
                NC.add(gates.M(0))
                NC.add(gates.M(1))

                expectation_val = GST_measure_72(NC, idx_basis_ops_72, k, j)

            column_of_js[j, :] = expectation_val

        return column_of_js

        # ================= END OF 16 MEASUREMENT BASES ===================|||

    ###################################
    ### 13 scenarios # (2023-11-06) ###
    ###################################

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
    import time

    tic = time.time()
    idx_BO = 0
    nqubits = 1
    for idx_gatenames_1 in range(0, 13):
        print("GST 1qb 13 Basis Op. Basis Op idx = %d" % (idx_gatenames_1))

        time.sleep(0)

        # ===============#=========================#
        # 13 SCENARIOS -- BIG ENDIAN NOTATION FOR THE 13 BASIS GATES TENSORED PRODUCTS
        # ==================================================================================

        # Initialise 16 different initial states
        for k in range(0, 4):
            #     print('Tomography of initial state %.2d' %(k))
            if k == 0:  # |0>
                circ = Circuit(nqubits, density_matrix=True)

            elif k == 1:  # |1>
                circ = Circuit(nqubits, density_matrix=True)
                circ.add(gates.X(0))

            elif k == 2:  # |+>
                circ = Circuit(nqubits, density_matrix=True)
                circ.add(gates.H(0))

            elif k == 3:  # |y+>
                circ = Circuit(nqubits, density_matrix=True)
                circ.add(gates.H(0))
                circ.add(gates.S(0))
            # --------------------------------------------

            if idx_gatenames_1 < 10:
                circ.add(
                    gates.Unitary(
                        BasisOps_13[idx_gatenames_1],
                        0,
                        trainable=False,
                        name="%s" % (gatenames[idx_gatenames_1]),
                    )
                )

                column_of_js = GSTBasisOpMeasurements_13(circ)

            # ==================================================================================

            ### REPREPARE QUBIT_1

            elif idx_gatenames_1 == 10:
                circ.add(
                    gates.ResetChannel(0, [1, 0])
                )  # ResetChannel(qubit, [prob0, prob1])
                circ.add(gates.H(0))

                circ.add(
                    gates.Unitary(
                        BasisOps_13[idx_gatenames_1],
                        0,
                        trainable=False,
                        name="%s" % (gatenames[idx_gatenames_1]),
                    )
                )

                column_of_js = GSTBasisOpMeasurements_13(circ)

            elif idx_gatenames_1 == 11:
                circ.add(
                    gates.ResetChannel(0, [1, 0])
                )  # ResetChannel(qubit, [prob0, prob1])
                circ.add(gates.H(0))
                circ.add(gates.S(0))

                circ.add(
                    gates.Unitary(
                        BasisOps_13[idx_gatenames_1],
                        0,
                        trainable=False,
                        name="%s" % (gatenames[idx_gatenames_1]),
                    )
                )

                column_of_js = GSTBasisOpMeasurements_13(circ)

            elif idx_gatenames_1 == 12:
                circ.add(
                    gates.ResetChannel(0, [1, 0])
                )  # ResetChannel(qubit, [prob0, prob1])

                circ.add(
                    gates.Unitary(
                        BasisOps_13[idx_gatenames_1],
                        0,
                        trainable=False,
                        name="%s" % (gatenames[idx_gatenames_1]),
                    )
                )

                column_of_js = GSTBasisOpMeasurements_13(circ)

            Bjk_tilde_1qb[idx_BO, :, k] = column_of_js.reshape(
                4,
            )
        idx_BO += 1

    toc = time.time() - tic
    print("13 basis operations tomography done. %.4f seconds\n" % (toc))

    #######################################################################
    ### Get noisy Pauli Transfer Matrix for each of 13 basis operations ###
    #######################################################################

    nqubits = 1
    T = np.matrix([[1, 1, 1, 1], [0, 0, 1, 0], [0, 0, 0, 1], [1, -1, 0, 0]])
    Bjk_hat_1qb = np.zeros((13, 4**nqubits, 4**nqubits))
    for idx_BO in range(0, 13):
        Bjk_hat_1qb[idx_BO, :, :] = (
            T * np.linalg.inv(gjk_1qb) * Bjk_tilde_1qb[idx_BO, :, :] * np.linalg.inv(T)
        )

    # Reshape
    Bjk_hat_1qb_reshaped = np.zeros((16, 13))
    for idx_basis_op in range(0, 13):
        temp = np.reshape(
            Bjk_hat_1qb[idx_basis_op, :, :], [16, 1], order="F"
        )  # 'F' so that it stacks columns.
        Bjk_hat_1qb_reshaped[:, idx_basis_op] = temp[:, 0].flatten()

    if type_of_gates["2 qb gate"] >= 1:
        ####################################
        ### 169 scenarios # (2020-12-05) ###
        ####################################

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
        import time

        tic = time.time()
        idx_BO = 0
        nqubits = 2
        for idx_gatenames_1 in range(0, 13):
            for idx_gatenames_2 in range(0, 13):
                # print('GST 2qb 241 Basis Op. Basis Op idx = %d (top reg idx = %d, bottom reg idx = %d)' %(idx_BO, idx_gatenames_1, idx_gatenames_2))
                print("GST 2qb 241 Basis Op. Basis Op idx = %d" % (idx_BO))

                # ===============#=========================#
                # 16 SCENARIOS  #  13^2 BASIS OPERATIONS  #  -- BIG ENDIAN NOTATION FOR THE 13 BASIS GATES TENSORED PRODUCTS
                # ==================================================================================

                # Initialise 16 different initial states
                for k in range(0, 16):
                    #     print('Tomography of initial state %.2d' %(k))
                    if k == 0:  # |0> |0>
                        circ = Circuit(nqubits, density_matrix=True)

                    elif k == 1:  # |0> |1>
                        circ = Circuit(nqubits, density_matrix=True)
                        circ.add(gates.X(1))

                    elif k == 2:  # |0> |+>
                        circ = Circuit(nqubits, density_matrix=True)
                        circ.add(gates.H(1))

                    elif k == 3:  # |0> |y+>
                        circ = Circuit(nqubits, density_matrix=True)
                        circ.add(gates.H(1))
                        circ.add(gates.S(1))
                    # --------------------------------------------

                    elif k == 4:  # |1> |0>
                        circ = Circuit(nqubits, density_matrix=True)
                        circ.add(gates.X(0))

                    elif k == 5:  # |1> |1>
                        circ = Circuit(nqubits, density_matrix=True)
                        circ.add(gates.X(0))
                        circ.add(gates.X(1))

                    elif k == 6:  # |1> |+>
                        circ = Circuit(nqubits, density_matrix=True)
                        circ.add(gates.X(0))
                        circ.add(gates.H(1))

                    elif k == 7:  # |1> |y+>
                        circ = Circuit(nqubits, density_matrix=True)
                        circ.add(gates.X(0))
                        circ.add(gates.H(1))
                        circ.add(gates.S(1))

                    # --------------------------------------------

                    elif k == 8:  # |+> |0>
                        circ = Circuit(nqubits, density_matrix=True)
                        circ.add(gates.H(0))

                    elif k == 9:  # |+> |1>
                        circ = Circuit(nqubits, density_matrix=True)
                        circ.add(gates.H(0))
                        circ.add(gates.X(1))

                    elif k == 10:  # |+> |+>
                        circ = Circuit(nqubits, density_matrix=True)
                        circ.add(gates.H(0))
                        circ.add(gates.H(1))

                    elif k == 11:  # |+> |y+>
                        circ = Circuit(nqubits, density_matrix=True)
                        circ.add(gates.H(0))
                        circ.add(gates.H(1))
                        circ.add(gates.S(1))

                    # --------------------------------------------

                    elif k == 12:  # |y+> |0>
                        circ = Circuit(nqubits, density_matrix=True)
                        circ.add(gates.H(0))
                        circ.add(gates.S(0))

                    elif k == 13:  # |y+> |1>
                        circ = Circuit(nqubits, density_matrix=True)
                        circ.add(gates.H(0))
                        circ.add(gates.S(0))
                        circ.add(gates.X(1))

                    elif k == 14:  # |y+> |+>
                        circ = Circuit(nqubits, density_matrix=True)
                        circ.add(gates.H(0))
                        circ.add(gates.S(0))
                        circ.add(gates.H(1))

                    elif k == 15:  # |y+> |y+>
                        circ = Circuit(nqubits, density_matrix=True)
                        circ.add(gates.H(0))
                        circ.add(gates.S(0))
                        circ.add(gates.H(1))
                        circ.add(gates.S(1))

                    # --------------------------------------------

                    if (
                        idx_gatenames_1 < 10 and idx_gatenames_2 < 10
                    ):  ### NEW SCENARIO 1
                        circ.add(
                            gates.Unitary(
                                BasisOps_169[idx_gatenames_1],
                                0,
                                trainable=False,
                                name="%s" % (gatenames[idx_gatenames_1]),
                            )
                        )
                        circ.add(
                            gates.Unitary(
                                BasisOps_169[idx_gatenames_2],
                                1,
                                trainable=False,
                                name="%s" % (gatenames[idx_gatenames_2]),
                            )
                        )

                        column_of_js = GSTBasisOpMeasurements_169(circ)

                    # ==================================================================================

                    ### PARTIALLY REPREPARE QUBIT_1

                    elif idx_gatenames_1 == 10 and idx_gatenames_2 < 10:
                        # print('Partially reprepare Qubit_1: idx_gatenames_1 = %d, idx_gatenames_2 = %d ' %(idx_gatenames_1, idx_gatenames_1))

                        circ.add(
                            gates.ResetChannel(0, [1, 0])
                        )  # ResetChannel(qubit, [prob0, prob1])
                        circ.add(gates.H(0))

                        circ.add(
                            gates.Unitary(
                                BasisOps_169[idx_gatenames_2],
                                1,
                                trainable=False,
                                name="%s" % (gatenames[idx_gatenames_2]),
                            )
                        )

                        column_of_js = GSTBasisOpMeasurements_169(circ)

                    elif idx_gatenames_1 == 11 and idx_gatenames_2 < 10:
                        # print('Partially reprepare Qubit_1: idx_gatenames_1 = %d, idx_gatenames_2 = %d ' %(idx_gatenames_1, idx_gatenames_2))

                        circ.add(
                            gates.ResetChannel(0, [1, 0])
                        )  # ResetChannel(qubit, [prob0, prob1])
                        circ.add(gates.H(0))
                        circ.add(gates.S(0))

                        circ.add(
                            gates.Unitary(
                                BasisOps_169[idx_gatenames_2],
                                1,
                                trainable=False,
                                name="%s" % (gatenames[idx_gatenames_2]),
                            )
                        )

                        column_of_js = GSTBasisOpMeasurements_169(circ)

                    elif idx_gatenames_1 == 12 and idx_gatenames_2 < 10:
                        # print('Partially reprepare Qubit_1: idx_gatenames_1 = %d, idx_gatenames_2 = %d ' %(idx_gatenames_1,idx_gatenames_2))

                        circ.add(
                            gates.ResetChannel(0, [1, 0])
                        )  # ResetChannel(qubit, [prob0, prob1])

                        circ.add(
                            gates.Unitary(
                                BasisOps_169[idx_gatenames_2],
                                1,
                                trainable=False,
                                name="%s" % (gatenames[idx_gatenames_2]),
                            )
                        )

                        column_of_js = GSTBasisOpMeasurements_169(circ)

                    # ==================================================================================

                    ### PARTIALLY REPREPARE QUBIT_2

                    elif idx_gatenames_1 < 10 and idx_gatenames_2 == 10:
                        # print('Partially reprepare Qubit_2: idx_gatenames_1 = %d, idx_gatenames_2 = %d ' %(idx_gatenames_1, idx_gatenames_2))

                        circ.add(
                            gates.Unitary(
                                BasisOps_169[idx_gatenames_1],
                                0,
                                trainable=False,
                                name="%s" % (gatenames[idx_gatenames_1]),
                            )
                        )
                        circ.add(
                            gates.ResetChannel(1, [1, 0])
                        )  # ResetChannel(qubit, [prob0, prob1])
                        circ.add(gates.H(1))

                        column_of_js = GSTBasisOpMeasurements_169(circ)

                    elif idx_gatenames_1 < 10 and idx_gatenames_2 == 11:
                        # print('Partially reprepare Qubit_2: idx_gatenames_1 = %d, idx_gatenames_2 = %d ' %(idx_gatenames_1,idx_gatenames_2))

                        circ.add(
                            gates.Unitary(
                                BasisOps_169[idx_gatenames_1],
                                0,
                                trainable=False,
                                name="%s" % (gatenames[idx_gatenames_1]),
                            )
                        )
                        circ.add(
                            gates.ResetChannel(1, [1, 0])
                        )  # ResetChannel(qubit, [prob0, prob1])
                        circ.add(gates.H(1))
                        circ.add(gates.S(1))

                        column_of_js = GSTBasisOpMeasurements_169(circ)

                    elif idx_gatenames_1 < 10 and idx_gatenames_2 == 12:
                        # print('Partially reprepare Qubit_2: idx_gatenames_1 = %d, idx_gatenames_2 = %d ' %(idx_gatenames_1,idx_gatenames_2))

                        circ.add(
                            gates.Unitary(
                                BasisOps_169[idx_gatenames_1],
                                0,
                                trainable=False,
                                name="%s" % (gatenames[idx_gatenames_1]),
                            )
                        )
                        circ.add(
                            gates.ResetChannel(1, [1, 0])
                        )  # ResetChannel(qubit, [prob0, prob1])

                        column_of_js = GSTBasisOpMeasurements_169(circ)

                    # ==================================================================================

                    ### FULLY REPREPARE QUBIT_1 AND QUBIT_2

                    elif idx_gatenames_1 == 10 and idx_gatenames_2 == 10:
                        # qubit_1 = reinitialised |+> state
                        # qubit_2 = reinitialised |+> state
                        circ.add(
                            gates.ResetChannel(0, [1, 0])
                        )  # ResetChannel(qubit, [prob0, prob1])
                        circ.add(
                            gates.ResetChannel(1, [1, 0])
                        )  # ResetChannel(qubit, [prob0, prob1])
                        circ.add(gates.H(0))
                        circ.add(gates.H(1))

                        column_of_js = GSTBasisOpMeasurements_169(circ)

                    elif idx_gatenames_1 == 10 and idx_gatenames_2 == 11:
                        # qubit_1 = reinitialised |+> state
                        # qubit_2 = reinitialised |y+> state
                        circ.add(
                            gates.ResetChannel(0, [1, 0])
                        )  # ResetChannel(qubit, [prob0, prob1])
                        circ.add(
                            gates.ResetChannel(1, [1, 0])
                        )  # ResetChannel(qubit, [prob0, prob1])
                        circ.add(gates.H(0))
                        circ.add(gates.H(1))
                        circ.add(gates.S(1))

                        column_of_js = GSTBasisOpMeasurements_169(circ)

                    elif idx_gatenames_1 == 10 and idx_gatenames_2 == 12:
                        # qubit_1 = reinitialised |+> state
                        # qubit_2 = reinitialised |0> state
                        circ.add(
                            gates.ResetChannel(0, [1, 0])
                        )  # ResetChannel(qubit, [prob0, prob1])
                        circ.add(
                            gates.ResetChannel(1, [1, 0])
                        )  # ResetChannel(qubit, [prob0, prob1])
                        circ.add(gates.H(0))

                        column_of_js = GSTBasisOpMeasurements_169(circ)

                    # ==================================================================================

                    elif idx_gatenames_1 == 11 and idx_gatenames_2 == 10:
                        # qubit_1 = reinitialised |y+> state
                        # qubit_2 = reinitialised |+> state
                        circ.add(
                            gates.ResetChannel(0, [1, 0])
                        )  # ResetChannel(qubit, [prob0, prob1])
                        circ.add(
                            gates.ResetChannel(1, [1, 0])
                        )  # ResetChannel(qubit, [prob0, prob1])
                        circ.add(gates.H(0))
                        circ.add(gates.S(0))
                        circ.add(gates.H(1))

                        column_of_js = GSTBasisOpMeasurements_169(circ)

                    elif idx_gatenames_1 == 11 and idx_gatenames_2 == 11:
                        # qubit_1 = reinitialised |y+> state
                        # qubit_2 = reinitialised |y+> state
                        circ.add(
                            gates.ResetChannel(0, [1, 0])
                        )  # ResetChannel(qubit, [prob0, prob1])
                        circ.add(
                            gates.ResetChannel(1, [1, 0])
                        )  # ResetChannel(qubit, [prob0, prob1])
                        circ.add(gates.H(0))
                        circ.add(gates.S(0))
                        circ.add(gates.H(1))
                        circ.add(gates.S(1))

                        column_of_js = GSTBasisOpMeasurements_169(circ)

                    elif idx_gatenames_1 == 11 and idx_gatenames_2 == 12:
                        # qubit_1 = reinitialised |y+> state
                        # qubit_2 = reinitialised |0> state
                        circ.add(
                            gates.ResetChannel(0, [1, 0])
                        )  # ResetChannel(qubit, [prob0, prob1])
                        circ.add(
                            gates.ResetChannel(1, [1, 0])
                        )  # ResetChannel(qubit, [prob0, prob1])
                        circ.add(gates.H(0))
                        circ.add(gates.S(0))

                        column_of_js = GSTBasisOpMeasurements_169(circ)

                    # ==================================================================================

                    elif idx_gatenames_1 == 12 and idx_gatenames_2 == 10:
                        # qubit_1 = reinitialised |0> state
                        # qubit_2 = reinitialised |+> state
                        circ.add(
                            gates.ResetChannel(0, [1, 0])
                        )  # ResetChannel(qubit, [prob0, prob1])
                        circ.add(
                            gates.ResetChannel(1, [1, 0])
                        )  # ResetChannel(qubit, [prob0, prob1])
                        circ.add(gates.H(1))

                        column_of_js = GSTBasisOpMeasurements_169(circ)

                    elif idx_gatenames_1 == 12 and idx_gatenames_2 == 11:
                        # qubit_1 = reinitialised |0> state
                        # qubit_2 = reinitialised |y+> state
                        circ.add(
                            gates.ResetChannel(0, [1, 0])
                        )  # ResetChannel(qubit, [prob0, prob1])
                        circ.add(
                            gates.ResetChannel(1, [1, 0])
                        )  # ResetChannel(qubit, [prob0, prob1])
                        circ.add(gates.H(1))
                        circ.add(gates.S(1))

                        column_of_js = GSTBasisOpMeasurements_169(circ)

                    elif idx_gatenames_1 == 12 and idx_gatenames_2 == 12:
                        # qubit_1 = reinitialised |0> state
                        # qubit_2 = reinitialised |0> state
                        circ.add(
                            gates.ResetChannel(0, [1, 0])
                        )  # ResetChannel(qubit, [prob0, prob1])
                        circ.add(
                            gates.ResetChannel(1, [1, 0])
                        )  # ResetChannel(qubit, [prob0, prob1])

                        column_of_js = GSTBasisOpMeasurements_169(circ)

                    # ==================================================================================

                    Bjk_tilde_2qb[idx_BO, :, k] = column_of_js.reshape(
                        16,
                    )

                idx_BO += 1

        toc = time.time() - tic
        print("0-168 basis operations tomography done. %.4f seconds\n" % (toc))

        ###################################
        ### 72 scenarios # (2020-12-05) ###
        ###################################

        tic = time.time()
        idx_BO = 169
        for idx_basis_ops_72 in range(169, 241):  # range(169,241):
            # print('GST 2qb 241 Basis Op. Basis Op idx = %d [%d]' %(idx_basis_ops_72, idx_basis_ops_72-169))
            print("GST 2qb 241 Basis Op. Basis Op idx = %d" % (idx_basis_ops_72))

            for k in range(0, 16):
                if k == 0:  # |0> |0>
                    circ = Circuit(nqubits, density_matrix=True)

                elif k == 1:  # |0> |1>
                    circ = Circuit(nqubits, density_matrix=True)
                    circ.add(gates.X(1))

                elif k == 2:  # |0> |+>
                    circ = Circuit(nqubits, density_matrix=True)
                    circ.add(gates.H(1))

                elif k == 3:  # |0> |y+>
                    circ = Circuit(nqubits, density_matrix=True)
                    circ.add(gates.H(1))
                    circ.add(gates.S(1))
                # --------------------------------------------

                elif k == 4:  # |1> |0>
                    circ = Circuit(nqubits, density_matrix=True)
                    circ.add(gates.X(0))

                elif k == 5:  # |1> |1>
                    circ = Circuit(nqubits, density_matrix=True)
                    circ.add(gates.X(0))
                    circ.add(gates.X(1))

                elif k == 6:  # |1> |+>
                    circ = Circuit(nqubits, density_matrix=True)
                    circ.add(gates.X(0))
                    circ.add(gates.H(1))

                elif k == 7:  # |1> |y+>
                    circ = Circuit(nqubits, density_matrix=True)
                    circ.add(gates.X(0))
                    circ.add(gates.H(1))
                    circ.add(gates.S(1))

                # --------------------------------------------

                elif k == 8:  # |+> |0>
                    circ = Circuit(nqubits, density_matrix=True)
                    circ.add(gates.H(0))

                elif k == 9:  # |+> |1>
                    circ = Circuit(nqubits, density_matrix=True)
                    circ.add(gates.H(0))
                    circ.add(gates.X(1))

                elif k == 10:  # |+> |+>
                    circ = Circuit(nqubits, density_matrix=True)
                    circ.add(gates.H(0))
                    circ.add(gates.H(1))

                elif k == 11:  # |+> |y+>
                    circ = Circuit(nqubits, density_matrix=True)
                    circ.add(gates.H(0))
                    circ.add(gates.H(1))
                    circ.add(gates.S(1))

                # --------------------------------------------

                elif k == 12:  # |y+> |0>
                    circ = Circuit(nqubits, density_matrix=True)
                    circ.add(gates.H(0))
                    circ.add(gates.S(0))

                elif k == 13:  # |y+> |1>
                    circ = Circuit(nqubits, density_matrix=True)
                    circ.add(gates.H(0))
                    circ.add(gates.S(0))
                    circ.add(gates.X(1))

                elif k == 14:  # |y+> |+>
                    circ = Circuit(nqubits, density_matrix=True)
                    circ.add(gates.H(0))
                    circ.add(gates.S(0))
                    circ.add(gates.H(1))

                elif k == 15:  # |y+> |y+>
                    circ = Circuit(nqubits, density_matrix=True)
                    circ.add(gates.H(0))
                    circ.add(gates.S(0))
                    circ.add(gates.H(1))
                    circ.add(gates.S(1))

                # --------------------------------------------

                # INPUT BASIS GATE
                circ.add(
                    gates.Unitary(
                        BasisOps_241[idx_basis_ops_72 - 169],
                        0,
                        1,
                        trainable=False,
                        name="%s" % (idx_basis_ops_72),
                    )
                )
                # print('TEst!')
                column_of_js = GSTBasisOpMeasurements_72(circ, idx_basis_ops_72)

                Bjk_tilde_2qb[idx_BO, :, k] = column_of_js.reshape(
                    16,
                )

            idx_BO += 1

        toc = time.time() - tic
        print("169-240 basis operations tomography done. %.4f seconds\n" % (toc))

        ########################################################################
        ### Get noisy Pauli Transfer Matrix for each of 241 basis operations ###
        ########################################################################

        nqubits = 2
        T = np.matrix([[1, 1, 1, 1], [0, 0, 1, 0], [0, 0, 0, 1], [1, -1, 0, 0]])
        Bjk_hat_2qb = np.zeros((241, 4**nqubits, 4**nqubits))
        for idx_BO in range(0, 241):
            Bjk_hat_2qb[idx_BO, :, :] = (
                np.kron(T, T)
                * np.linalg.inv(gjk_2qb)
                * Bjk_tilde_2qb[idx_BO, :, :]
                * np.linalg.inv(np.kron(T, T))
            )

        # Reshape
        Bjk_hat_2qb_reshaped = np.zeros((256, 241))
        for idx_basis_op in range(0, 241):
            temp = np.reshape(
                Bjk_hat_2qb[idx_basis_op, :, :], [256, 1], order="F"
            )  # 'F' so that it stacks columns.
            Bjk_hat_2qb_reshaped[:, idx_basis_op] = temp[:, 0].flatten()

    def GST_measure_op_1qb(circuit, operatorname, k, j):
        idx_basis = ["I", "X", "Y", "Z"]
        filedirectory = "GST_gates_qibo_1qb/"

        result = circuit.execute(nshots=NshotsGST)
        counts = dict(result.frequencies(binary=True))

        # Save counts into a matrix. 0th column: word; 1st column: count value
        # --------------------------------------------------------------------
        fout = filedirectory + "OPERATOR %s (k=%.2d,j=%.2d) %s_basis.txt" % (
            operatorname,
            k,
            j,
            idx_basis[j],
        )
        fo = open(fout, "w")
        for v1, v2 in counts.items():
            fo.write(str(v1) + "  " + str(v2) + "\n")
        fo.close()

        # Find value for gjk matrix
        # -------------------------

        expectation_signs = np.matrix([[1, 1, 1, 1], [1, -1, -1, -1]])

        matrix = np.zeros((2, 2))
        for ii in range(0, 2):
            word = bin(ii)[2:]
            matrix[ii, 0] = word

        for v1, v2 in counts.items():
            # print(v1, v2)
            row = int(v1, 2)
            col = v2
            matrix[row, 1] = col

        sorted_matrix = sort_counts_1qb(matrix)
        probs = sorted_matrix[:, 1] / np.sum(sorted_matrix[:, 1])
        sorted_matrix = np.column_stack((sorted_matrix, probs))

        k = k + 1
        j = j + 1

        if j == 1:
            sorted_matrix = np.column_stack((sorted_matrix, expectation_signs[:, 0]))
        elif j == 2:
            sorted_matrix = np.column_stack((sorted_matrix, expectation_signs[:, 1]))
        elif j == 3:
            sorted_matrix = np.column_stack((sorted_matrix, expectation_signs[:, 1]))
        elif j == 4:
            sorted_matrix = np.column_stack((sorted_matrix, expectation_signs[:, 1]))
        elif j == 5:
            sorted_matrix = np.column_stack((sorted_matrix, expectation_signs[:, 2]))
        elif j == 9:
            sorted_matrix = np.column_stack((sorted_matrix, expectation_signs[:, 2]))
        elif j == 13:
            sorted_matrix = np.column_stack((sorted_matrix, expectation_signs[:, 2]))
        else:
            sorted_matrix = np.column_stack((sorted_matrix, expectation_signs[:, 3]))

        temp = np.array(sorted_matrix[:, 2]) * np.array(sorted_matrix[:, 3])
        sorted_matrix = np.column_stack((sorted_matrix, temp))

        expectation_val = np.sum(temp)

        return expectation_val

    def GST_1qb_operator(class_qubit_gate, operatorname):
        nqubits = 1
        Ojk_tilde_1qb = np.zeros((4**nqubits, 4**nqubits))

        for k in range(0, 4):
            if k == 0:  # |0>
                circ = Circuit(nqubits, density_matrix=True)

            elif k == 1:  # |1>
                circ = Circuit(nqubits, density_matrix=True)
                circ.add(gates.X(0))

            elif k == 2:  # |+>
                circ = Circuit(nqubits, density_matrix=True)
                circ.add(gates.H(0))

            elif k == 3:  # |y+>
                circ = Circuit(nqubits, density_matrix=True)
                circ.add(gates.H(0))
                circ.add(gates.S(0))

            # --------------------------------------------

            # INPUT OPERATOR
            circ.add(getattr(gates, class_qubit_gate)(0))

            initial_state = np.zeros(2**nqubits)
            initial_state[0] = 1
            # ================= START OF 16 MEASUREMENT BASES =================|||
            for j in range(0, 4):
                # print('initial state k=%d, measurement basis %d' %(k,j))

                if j == 0:  # Identity basis  # Identity basis
                    circ0 = Circuit(nqubits, density_matrix=True)
                    NC = circ + circ0
                    if noise_model is not None and backend.name != "qibolab":
                        NC = noise_model.apply(NC)
                    NC.add(gates.M(0))

                    expectation_val = GST_measure_op_1qb(NC, operatorname, k, j)

                elif j == 1:  # Identity basis # X basis
                    circ1 = Circuit(nqubits, density_matrix=True)
                    NC = circ + circ1
                    NC.add(gates.H(0))
                    if noise_model is not None and backend.name != "qibolab":
                        NC = noise_model.apply(NC)
                    NC.add(gates.M(0))

                    expectation_val = GST_measure_op_1qb(NC, operatorname, k, j)

                elif j == 2:  # Identity basis # Y basis
                    circ2 = Circuit(nqubits, density_matrix=True)
                    NC = circ + circ2
                    NC.add(gates.SDG(0))
                    NC.add(gates.H(0))
                    if noise_model is not None and backend.name != "qibolab":
                        NC = noise_model.apply(NC)
                    NC.add(gates.M(0))

                    expectation_val = GST_measure_op_1qb(NC, operatorname, k, j)

                elif j == 3:  # Identity basis # Z basis
                    circ3 = Circuit(nqubits, density_matrix=True)
                    NC = circ + circ3
                    if noise_model is not None and backend.name != "qibolab":
                        NC = noise_model.apply(NC)
                    NC.add(gates.M(0))

                    expectation_val = GST_measure_op_1qb(NC, operatorname, k, j)

                Ojk_tilde_1qb[j, k] = expectation_val

        return Ojk_tilde_1qb

    def GST_measure_op_2qb(circuit, operatorname, k, j):
        idx_basis = [
            "II",
            "IX",
            "IY",
            "IZ",
            "XI",
            "XX",
            "XY",
            "XZ",
            "YI",
            "YX",
            "YY",
            "YZ",
            "ZI",
            "ZX",
            "ZY",
            "ZZ",
        ]
        filedirectory = "GST_gates_qibo_2qb/"

        result = circuit.execute(nshots=NshotsGST)
        counts = dict(result.frequencies(binary=True))

        # Save counts into a matrix. 0th column: word; 1st column: count value
        # --------------------------------------------------------------------
        fout = filedirectory + "Q_1 Q_2 OPERATOR_%s (k=%.2d,j=%.2d) %s_basis.txt" % (
            operatorname,
            k,
            j,
            idx_basis[j],
        )
        fo = open(fout, "w")
        for v1, v2 in counts.items():
            fo.write(str(v1) + "  " + str(v2) + "\n")
        fo.close()

        # Find value for gjk matrix
        # -------------------------

        expectation_signs = np.matrix(
            [[1, 1, 1, 1], [1, -1, 1, -1], [1, 1, -1, -1], [1, -1, -1, 1]]
        )

        matrix = np.zeros((2**2, 2))
        for ii in range(0, 2**2):
            word = bin(ii)[2:]
            matrix[ii, 0] = word

        for v1, v2 in counts.items():
            # print(v1, v2)
            row = int(v1, 2)
            col = v2
            matrix[row, 1] = col

        sorted_matrix = sort_counts_2qb(matrix)
        probs = sorted_matrix[:, 1] / np.sum(sorted_matrix[:, 1])
        sorted_matrix = np.column_stack((sorted_matrix, probs))

        k = k + 1
        j = j + 1

        if j == 1:
            sorted_matrix = np.column_stack((sorted_matrix, expectation_signs[:, 0]))
        elif j == 2:
            sorted_matrix = np.column_stack((sorted_matrix, expectation_signs[:, 1]))
        elif j == 3:
            sorted_matrix = np.column_stack((sorted_matrix, expectation_signs[:, 1]))
        elif j == 4:
            sorted_matrix = np.column_stack((sorted_matrix, expectation_signs[:, 1]))
        elif j == 5:
            sorted_matrix = np.column_stack((sorted_matrix, expectation_signs[:, 2]))
        elif j == 9:
            sorted_matrix = np.column_stack((sorted_matrix, expectation_signs[:, 2]))
        elif j == 13:
            sorted_matrix = np.column_stack((sorted_matrix, expectation_signs[:, 2]))
        else:
            sorted_matrix = np.column_stack((sorted_matrix, expectation_signs[:, 3]))

        temp = np.array(sorted_matrix[:, 2]) * np.array(sorted_matrix[:, 3])
        sorted_matrix = np.column_stack((sorted_matrix, temp))

        expectation_val = np.sum(temp)

        return expectation_val

    def GST_2qb_operator(class_qubit_gate, operatorname, ctrl_qb, targ_qb):
        nqubits = 2
        Ojk_tilde_2qb = np.zeros((4**nqubits, 4**nqubits))

        for k in range(0, 16):
            if k == 0:  # |0> |0>
                circ = Circuit(nqubits, density_matrix=True)

            elif k == 1:  # |0> |1>
                circ = Circuit(nqubits, density_matrix=True)
                circ.add(gates.X(1))

            elif k == 2:  # |0> |+>
                circ = Circuit(nqubits, density_matrix=True)
                circ.add(gates.H(1))

            elif k == 3:  # |0> |y+>
                circ = Circuit(nqubits, density_matrix=True)
                circ.add(gates.H(1))
                circ.add(gates.S(1))
            # --------------------------------------------

            elif k == 4:  # |1> |0>
                circ = Circuit(nqubits, density_matrix=True)
                circ.add(gates.X(0))

            elif k == 5:  # |1> |1>
                circ = Circuit(nqubits, density_matrix=True)
                circ.add(gates.X(0))
                circ.add(gates.X(1))

            elif k == 6:  # |1> |+>
                circ = Circuit(nqubits, density_matrix=True)
                circ.add(gates.X(0))
                circ.add(gates.H(1))

            elif k == 7:  # |1> |y+>
                circ = Circuit(nqubits, density_matrix=True)
                circ.add(gates.X(0))
                circ.add(gates.H(1))
                circ.add(gates.S(1))

            # --------------------------------------------

            elif k == 8:  # |+> |0>
                circ = Circuit(nqubits, density_matrix=True)
                circ.add(gates.H(0))

            elif k == 9:  # |+> |1>
                circ = Circuit(nqubits, density_matrix=True)
                circ.add(gates.H(0))
                circ.add(gates.X(1))

            elif k == 10:  # |+> |+>
                circ = Circuit(nqubits, density_matrix=True)
                circ.add(gates.H(0))
                circ.add(gates.H(1))

            elif k == 11:  # |+> |y+>
                circ = Circuit(nqubits, density_matrix=True)
                circ.add(gates.H(0))
                circ.add(gates.H(1))
                circ.add(gates.S(1))

            # --------------------------------------------

            elif k == 12:  # |y+> |0>
                circ = Circuit(nqubits, density_matrix=True)
                circ.add(gates.H(0))
                circ.add(gates.S(0))

            elif k == 13:  # |y+> |1>
                circ = Circuit(nqubits, density_matrix=True)
                circ.add(gates.H(0))
                circ.add(gates.S(0))
                circ.add(gates.X(1))

            elif k == 14:  # |y+> |+>
                circ = Circuit(nqubits, density_matrix=True)
                circ.add(gates.H(0))
                circ.add(gates.S(0))
                circ.add(gates.H(1))

            elif k == 15:  # |y+> |y+>
                circ = Circuit(nqubits, density_matrix=True)
                circ.add(gates.H(0))
                circ.add(gates.S(0))
                circ.add(gates.H(1))
                circ.add(gates.S(1))

            # --------------------------------------------

            # INPUT OPERATOR
            circ.add(
                getattr(gates, class_qubit_gate)(targ_qb, ctrl_qb)
            )  # <-- (targ_qb, ctrl_qb) gives the correct results. Probably the residual little Endian Qiskit issue. Not necessary to sort out.
            # circ.add(getattr(gates, class_qubit_gate)(ctrl_qb, targ_qb))

            initial_state = np.zeros(2**nqubits)
            initial_state[0] = 1
            # ================= START OF 16 MEASUREMENT BASES =================|||
            for j in range(0, 16):
                # print('initial state k=%d, measurement basis %d' %(k,j))

                if j == 0:  # Identity basis  # Identity basis
                    circ0 = Circuit(nqubits, density_matrix=True)
                    NC = circ + circ0
                    if noise_model is not None and backend.name != "qibolab":
                        NC = noise_model.apply(NC)
                    NC.add(gates.M(0))
                    NC.add(gates.M(1))

                    expectation_val = GST_measure_op_2qb(NC, operatorname, k, j)

                elif j == 1:  # Identity basis # X basis
                    circ1 = Circuit(nqubits, density_matrix=True)
                    NC = circ + circ1
                    NC.add(gates.H(1))
                    if noise_model is not None and backend.name != "qibolab":
                        NC = noise_model.apply(NC)
                    NC.add(gates.M(0))
                    NC.add(gates.M(1))

                    expectation_val = GST_measure_op_2qb(NC, operatorname, k, j)

                elif j == 2:  # Identity basis # Y basis
                    circ2 = Circuit(nqubits, density_matrix=True)
                    NC = circ + circ2
                    NC.add(gates.SDG(1))
                    NC.add(gates.H(1))
                    if noise_model is not None and backend.name != "qibolab":
                        NC = noise_model.apply(NC)
                    NC.add(gates.M(0))
                    NC.add(gates.M(1))

                    expectation_val = GST_measure_op_2qb(NC, operatorname, k, j)

                elif j == 3:  # Identity basis # Z basis
                    circ3 = Circuit(nqubits, density_matrix=True)
                    NC = circ + circ3
                    if noise_model is not None and backend.name != "qibolab":
                        NC = noise_model.apply(NC)
                    NC.add(gates.M(0))
                    NC.add(gates.M(1))

                    expectation_val = GST_measure_op_2qb(NC, operatorname, k, j)

                # =================================================================================================================

                elif j == 4:  # X basis# Identity basis
                    circ4 = Circuit(nqubits, density_matrix=True)
                    NC = circ + circ4
                    NC.add(gates.H(0))
                    if noise_model is not None and backend.name != "qibolab":
                        NC = noise_model.apply(NC)
                    NC.add(gates.M(0))
                    NC.add(gates.M(1))

                    expectation_val = GST_measure_op_2qb(NC, operatorname, k, j)

                elif j == 5:  # X basis # X basis
                    circ5 = Circuit(nqubits, density_matrix=True)
                    NC = circ + circ5
                    NC.add(gates.H(0))
                    NC.add(gates.H(1))
                    if noise_model is not None and backend.name != "qibolab":
                        NC = noise_model.apply(NC)
                    NC.add(gates.M(0))
                    NC.add(gates.M(1))

                    expectation_val = GST_measure_op_2qb(NC, operatorname, k, j)

                elif j == 6:  # X basis # Y basis
                    circ6 = Circuit(nqubits, density_matrix=True)
                    NC = circ + circ6
                    NC.add(gates.H(0))
                    NC.add(gates.SDG(1))
                    NC.add(gates.H(1))
                    if noise_model is not None and backend.name != "qibolab":
                        NC = noise_model.apply(NC)
                    NC.add(gates.M(0))
                    NC.add(gates.M(1))

                    expectation_val = GST_measure_op_2qb(NC, operatorname, k, j)

                elif j == 7:  # X basis  # Z basis
                    circ7 = Circuit(nqubits, density_matrix=True)
                    NC = circ + circ7
                    NC.add(gates.H(0))
                    if noise_model is not None and backend.name != "qibolab":
                        NC = noise_model.apply(NC)
                    NC.add(gates.M(0))
                    NC.add(gates.M(1))

                    expectation_val = GST_measure_op_2qb(NC, operatorname, k, j)

                # =================================================================================================================

                elif j == 8:  # Y basis # Identity basis
                    circ8 = Circuit(nqubits, density_matrix=True)
                    NC = circ + circ8
                    NC.add(gates.SDG(0))
                    NC.add(gates.H(0))
                    if noise_model is not None and backend.name != "qibolab":
                        NC = noise_model.apply(NC)
                    NC.add(gates.M(0))
                    NC.add(gates.M(1))

                    expectation_val = GST_measure_op_2qb(NC, operatorname, k, j)

                elif j == 9:  # Y basis  # X basis
                    circ9 = Circuit(nqubits, density_matrix=True)
                    NC = circ + circ9
                    NC.add(gates.SDG(0))
                    NC.add(gates.H(0))
                    NC.add(gates.H(1))
                    if noise_model is not None and backend.name != "qibolab":
                        NC = noise_model.apply(NC)
                    NC.add(gates.M(0))
                    NC.add(gates.M(1))

                    expectation_val = GST_measure_op_2qb(NC, operatorname, k, j)

                elif j == 10:  # Y basis  # Y basis
                    circ10 = Circuit(nqubits, density_matrix=True)
                    NC = circ + circ10
                    NC.add(gates.SDG(0))
                    NC.add(gates.H(0))
                    NC.add(gates.SDG(1))
                    NC.add(gates.H(1))
                    if noise_model is not None and backend.name != "qibolab":
                        NC = noise_model.apply(NC)
                    NC.add(gates.M(0))
                    NC.add(gates.M(1))

                    expectation_val = GST_measure_op_2qb(NC, operatorname, k, j)

                elif j == 11:  # Y basis  # Z basis
                    circ11 = Circuit(nqubits, density_matrix=True)
                    NC = circ + circ11
                    NC.add(gates.SDG(0))
                    NC.add(gates.H(0))
                    if noise_model is not None and backend.name != "qibolab":
                        NC = noise_model.apply(NC)
                    NC.add(gates.M(0))
                    NC.add(gates.M(1))

                    expectation_val = GST_measure_op_2qb(NC, operatorname, k, j)

                # =================================================================================================================

                elif j == 12:  # Z basis# Identity basis
                    circ12 = Circuit(nqubits, density_matrix=True)
                    NC = circ + circ12
                    if noise_model is not None and backend.name != "qibolab":
                        NC = noise_model.apply(NC)
                    NC.add(gates.M(0))
                    NC.add(gates.M(1))

                    expectation_val = GST_measure_op_2qb(NC, operatorname, k, j)

                elif j == 13:  # Z basis  # X basis
                    circ13 = Circuit(nqubits, density_matrix=True)
                    NC = circ + circ13
                    NC.add(gates.H(1))
                    if noise_model is not None and backend.name != "qibolab":
                        NC = noise_model.apply(NC)
                    NC.add(gates.M(0))
                    NC.add(gates.M(1))

                    expectation_val = GST_measure_op_2qb(NC, operatorname, k, j)

                elif j == 14:  # Z basis  # Y basis
                    circ14 = Circuit(nqubits, density_matrix=True)
                    NC = circ + circ14
                    NC.add(gates.SDG(1))
                    NC.add(gates.H(1))
                    if noise_model is not None and backend.name != "qibolab":
                        NC = noise_model.apply(NC)
                    NC.add(gates.M(0))
                    NC.add(gates.M(1))

                    expectation_val = GST_measure_op_2qb(NC, operatorname, k, j)

                elif j == 15:  # Z basis # Z basis
                    circ15 = Circuit(nqubits, density_matrix=True)
                    NC = circ + circ15
                    if noise_model is not None and backend.name != "qibolab":
                        NC = noise_model.apply(NC)
                    NC.add(gates.M(0))
                    NC.add(gates.M(1))

                    expectation_val = GST_measure_op_2qb(NC, operatorname, k, j)

                Ojk_tilde_2qb[j, k] = expectation_val

        return Ojk_tilde_2qb

    # for idx_ops in range(0,no_of_operators_1qb):
    #     print(f'Operator: {operatornames_1qb[idx_ops]}')
    #     pretty_print_matrix(Ojk_tilde_1qb[idx_ops,:,:])
    #     print(' ')

    # for idx_ops in range(0,no_of_operators_2qb):
    #     print(f'Operator: {operatornames_2qb[idx_ops]}')
    #     pretty_print_matrix(Ojk_tilde_2qb[idx_ops,:,:])
    #     print(' ')

    one_qb_tilde = []
    two_qb_tilde = []
    one_qb_exact_operators = []
    two_qb_exact_operators = []
    one_qb_operatorname = []
    two_qb_operatorname = []

    for data in circuit.raw["queue"]:
        # print(data)
        num_qubit_gate = len(data["init_args"])
        name_qubit_gate = data["name"]
        class_qubit_gate = data["_class"]
        ctrl_qb = data["_control_qubits"]
        targ_qb = data["_target_qubits"]

        # Do GST for one qubit and two qubit operators
        # print(f'{name_qubit_gate} is a {num_qubit_gate}-qubit gate, _class={class_qubit_gate}')

        if num_qubit_gate == 1:
            if class_qubit_gate != "M":
                print(
                    f"Perform {num_qubit_gate}-qubit gate GST for {name_qubit_gate}, _class={class_qubit_gate}"
                )
                hola = GST_1qb_operator(
                    class_qubit_gate, name_qubit_gate
                )  # Perform GST here
                one_qb_tilde.append(hola)

                matrix_form = getattr(gates, class_qubit_gate)(targ_qb[0]).matrix()
                one_qb_exact_operators.append(matrix_form)
                one_qb_operatorname.append(name_qubit_gate)

        elif num_qubit_gate == 2:
            print(
                f"Perform {num_qubit_gate}-qubit gate GST for {name_qubit_gate}, _class={class_qubit_gate}"
            )
            gracias = GST_2qb_operator(
                class_qubit_gate, name_qubit_gate, ctrl_qb[0], targ_qb[0]
            )  # Perform GST here
            two_qb_tilde.append(gracias)

            matrix_form = getattr(gates, class_qubit_gate)(
                ctrl_qb[0], targ_qb[0]
            ).matrix()
            two_qb_exact_operators.append(matrix_form)
            two_qb_operatorname.append(name_qubit_gate)

    if type_of_gates["1 qb gate"] >= 1:
        one_qb_tilde = np.reshape(one_qb_tilde, [len(one_qb_tilde), 4, 4])
        one_qb_exact_operators = np.reshape(
            one_qb_exact_operators, [len(one_qb_exact_operators), 2, 2]
        )
        print(one_qb_exact_operators)

    if type_of_gates["2 qb gate"] >= 1:
        two_qb_tilde = np.reshape(two_qb_tilde, [len(two_qb_tilde), 16, 16])
        two_qb_exact_operators = np.reshape(
            two_qb_exact_operators, [len(two_qb_exact_operators), 4, 4]
        )
        print(two_qb_exact_operators)

    if type_of_gates["1 qb gate"] >= 1:
        ########################################################
        ### Compute Noisy Pauli Transfer Matrix of operators ###
        ########################################################
        # $\mathcal{\hat{O}}^{(l)} = T g^{-1} \mathcal{\tilde{O}}^{(l)} T^{-1}$

        no_of_operators_1qb = np.shape(one_qb_tilde)[0]
        nqubits = 1

        T = np.matrix([[1, 1, 1, 1], [0, 0, 1, 0], [0, 0, 0, 1], [1, -1, 0, 0]])

        one_qb_hat = np.zeros((no_of_operators_1qb, 4**nqubits, 4**nqubits))
        for idx_ops in range(0, no_of_operators_1qb):
            one_qb_hat[idx_ops, :, :] = (
                T
                * np.linalg.inv(gjk_1qb)
                * one_qb_tilde[idx_ops, :, :]
                * np.linalg.inv(T)
            )

            # print(f'Noisy Pauli Transfer Matrix for for 1qb {one_qb_operatorname[idx_ops]} operator')
            # pretty_print_matrix(one_qb_hat[idx_ops,:,:])

        ########################################################
        ### Compute exact Pauli Transfer Matrix of operators ###
        ########################################################
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

            # print(f'Exact Pauli Transfer Matrix for 1qb {one_qb_operatorname[idx_ops]} operator')
            # pretty_print_matrix(one_qb_PTM[idx_ops, :, :])

        ##########################################
        ### Compute inverse noise of operators ###
        ##########################################
        # $(\mathcal{N}^{(l)})^{-1} = \mathcal{{O}}^{(l), exact} (\mathcal{\hat{O}}^{(l)})^{-1}$

        invNoise_1qb = np.zeros((no_of_operators_1qb, 4**nqubits, 4**nqubits))

        for idx_ops in range(0, no_of_operators_1qb):
            invNoise_1qb[idx_ops, :, :] = np.matrix(one_qb_PTM[idx_ops, :, :]) @ (
                np.matrix(np.linalg.inv(one_qb_hat[idx_ops, :, :]))
            )

        for idx_ops in range(0, no_of_operators_1qb):
            print(f"Inverse noise for 1qb {one_qb_operatorname[idx_ops]} operator")
            pretty_print_matrix(invNoise_1qb[idx_ops, :, :])

            print(
                f"Print diagonal elements of {one_qb_operatorname[idx_ops]} invNoise_1qb:"
            )
            for ii in range(0, np.shape(invNoise_1qb[idx_ops, :, :])[0]):
                print("%.4f" % (np.around(invNoise_1qb[idx_ops, ii, ii], 4)))

    if type_of_gates["2 qb gate"] >= 1:
        ########################################################
        ### Compute Noisy Pauli Transfer Matrix of operators ###
        ########################################################
        # $\mathcal{\hat{O}}^{(l)} = T g^{-1} \mathcal{\tilde{O}}^{(l)} T^{-1}$

        no_of_operators_2qb = np.shape(two_qb_tilde)[0]
        print("Number of two qubit operators:", no_of_operators_2qb)
        nqubits = 2

        T = np.matrix([[1, 1, 1, 1], [0, 0, 1, 0], [0, 0, 0, 1], [1, -1, 0, 0]])

        two_qb_hat = np.zeros((no_of_operators_2qb, 4**nqubits, 4**nqubits))
        for idx_ops in range(0, no_of_operators_2qb):
            two_qb_hat[idx_ops, :, :] = (
                np.kron(T, T)
                * np.linalg.inv(gjk_2qb)
                * two_qb_tilde[idx_ops, :, :]
                * np.linalg.inv(np.kron(T, T))
            )

            # print(f'Noisy Pauli Transfer Matrix for 2qb {two_qb_operatorname[idx_ops]} operator')
            # pretty_print_matrix(two_qb_hat[idx_ops,:,:])

        ########################################################
        ### Compute exact Pauli Transfer Matrix of operators ###
        ########################################################
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

            # print(f'Exact Pauli Transfer Matrix for 2qb {two_qb_operatorname[idx_ops]} operator')
            # pretty_print_matrix(two_qb_PTM[idx_ops, :, :])

        ##########################################
        ### Compute inverse noise of operators ###
        ##########################################
        # $(\mathcal{N}^{(l)})^{-1} = \mathcal{{O}}^{(l), exact} (\mathcal{\hat{O}}^{(l)})^{-1}$

        invNoise_2qb = np.zeros((no_of_operators_2qb, 4**nqubits, 4**nqubits))

        for idx_ops in range(0, no_of_operators_2qb):
            invNoise_2qb[idx_ops, :, :] = np.matrix(two_qb_PTM[idx_ops, :, :]) @ (
                np.matrix(np.linalg.inv(two_qb_hat[idx_ops, :, :]))
            )

        for idx_ops in range(0, no_of_operators_2qb):
            print(f"Inverse noise for 2qb {two_qb_operatorname[idx_ops]} operator")
            pretty_print_matrix(invNoise_2qb[idx_ops, :, :])

            print(
                f"Print diagonal elements of {two_qb_operatorname[idx_ops]} invNoise_2qb:"
            )
            for ii in range(0, np.shape(invNoise_2qb[idx_ops, :, :])[0]):
                print("%.4f" % (np.around(invNoise_2qb[idx_ops, ii, ii], 4)))

    ############################################################
    ### Decompose inverse noises in term of Basis Operations ###
    ############################################################

    ################################
    ##                            ##
    ##    ##              ##      ##
    ##  ####      ####    ##      ##
    ##    ##     ##  ##   #####   ##
    ##    ##     ##  ##   ##  ##  ##
    ##    ##      #####   ##  ##  ##
    ##  #####        ##   #####   ##
    ##               ###          ##
    ##                            ##
    ################################

    nqubits = 1

    if type_of_gates["1 qb gate"] >= 1:
        nqubits = 1

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

        ##################################################################
        ### Get indicative sampling cost contribution from operator(s) ###
        ##################################################################

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

    #####################################################################
    ### Get indicative sampling cost contribution from initial states ###
    #####################################################################

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

    ###################################################################
    ### Get indicative sampling cost contribution from measurements ###
    ###################################################################

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
    print("     CQ_1qb =", CQ_1qb)
    print("   Crho_1qb =", Crho_1qb)

    if type_of_gates["1 qb gate"] >= 1:
        ########################################
        ### Compute indicative sampling cost ###
        ########################################

        print("     CO_1qb =", CO_1qb[0])

    if type_of_gates["2 qb gate"] >= 1:
        #################################
        ##                             ##
        ##    ####             ##      ##
        ##  ##   ##    ####    ##      ##
        ##       ##   ##  ##   #####   ##
        ##      ##    ##  ##   ##  ##  ##
        ##    ##       #####   ##  ##  ##
        ##  #######       ##   #####   ##
        ##                ###          ##
        ##                             ##
        #################################

        nqubits = 2

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

        ##################################################################
        ### Get indicative sampling cost contribution from operator(s) ###
        ##################################################################

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

        #####################################################################
        ### Get indicative sampling cost contribution from initial states ###
        #####################################################################

        Qhat_row_2qb = gjk_2qb * np.linalg.inv(np.kron(T, T))
        Qhat_col_2qb = np.transpose(Qhat_row_2qb)

        rhohat_col_2qb = np.kron(T, T)

        idealrho_Pauligates = []
        for ii in range(0, 4):
            for jj in range(0, 4):
                idealrho_Pauligates.append(
                    np.kron(Pauligates_1qubit[ii], Pauligates_1qubit[jj])
                )

        ideal_state_2qb = np.matrix([[1], [0], [0], [0]])
        ideal_state_rho_2qb = ideal_state_2qb * np.transpose(ideal_state_2qb)

        ideal_rho_PTM_2qb = np.zeros((4**nqubits, 1))
        for ii in range(0, 16):
            ideal_rho_PTM_2qb[ii, 0] = np.trace(
                idealrho_Pauligates[ii] * ideal_state_rho_2qb
            )

        qrhovector_2qb, _, _, _ = np.linalg.lstsq(
            rhohat_col_2qb, ideal_rho_PTM_2qb, rcond=None
        )
        Crho_2qb = np.sum(np.abs(qrhovector_2qb))
        qrhoprob_2qb = np.abs(qrhovector_2qb) / Crho_2qb
        CDF_rho_2qb = np.cumsum(qrhoprob_2qb)

        ###################################################################
        ### Get indicative sampling cost contribution from measurements ###
        ###################################################################

        idealQ_Pauligates = []
        for ii in range(0, 4):
            for jj in range(0, 4):
                idealQ_Pauligates.append(
                    np.kron(Pauligates_1qubit[ii], Pauligates_1qubit[jj])
                )

        ideal_state_2qb_Q = np.kron(Pauligates_1qubit[3], Pauligates_1qubit[3])

        ideal_Q_PTM_row_2qb = np.zeros((1, 4**nqubits))
        for ii in range(0, 4**nqubits):
            ideal_Q_PTM_row_2qb[0, ii] = (1 / (2**nqubits)) * np.trace(
                idealQ_Pauligates[ii] * ideal_state_2qb_Q
            )

        ideal_Q_PTM_col_2qb = np.transpose(ideal_Q_PTM_row_2qb)

        qQvector_2qb, _, _, _ = np.linalg.lstsq(
            Qhat_col_2qb, ideal_Q_PTM_col_2qb, rcond=None
        )
        CQ_2qb = np.sum(np.abs(qQvector_2qb))
        qQprob_2qb = np.abs(qQvector_2qb) / CQ_2qb
        CDF_Q_2qb = np.cumsum(qQprob_2qb)

        ########################################
        ### Compute indicative sampling cost ###
        ########################################

        print("     CO_2qb =", CO_2qb[0])

    ##############################################
    ### Compute total indicative sampling cost ###
    ##############################################

    if type_of_gates["1 qb gate"] >= 1 and type_of_gates["2 qb gate"] == 0:
        Csample_total = (Crho_1qb * CQ_1qb) ** circuit.nqubits * np.prod(CO_1qb)

    elif type_of_gates["2 qb gate"] >= 1 and type_of_gates["1 qb gate"] == 0:
        Csample_total = (Crho_1qb * CQ_1qb) ** circuit.nqubits * np.prod(CO_2qb)

    elif type_of_gates["2 qb gate"] >= 1 and type_of_gates["1 qb gate"] >= 1:
        Csample_total = (
            (Crho_1qb * CQ_1qb) ** circuit.nqubits * np.prod(CO_2qb) * np.prod(CO_1qb)
        )

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

    print(
        f"########################### MONTE CARLO SAMPLING USING {nshots_MC} SHOTS ###########################"
    )

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

    def measurements_top_and_bottom_registers(
        qc, index_of_Q, top_register, bottom_register
    ):
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

    if type_of_gates["1 qb gate"] >= 1:
        # 1 qubit Cumulative Distribution Function
        CDF_O_1qb_matrix = np.zeros((13, 2))
        CDF_O_1qb_matrix[:, 0] = np.arange(0, 13, 1)
        CDF_O_1qb_matrix[:, 1] = np.arange(0, 13, 1)
        CDF_O_1qb_matrix = np.hstack((CDF_O_1qb_matrix, CDF_O_1qb, qOvector_1qb))
        # pretty_print_matrix(CDF_O_1qb_matrix)

    if type_of_gates["2 qb gate"] >= 1:
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

    if type_of_gates["1 qb gate"] >= 1 and type_of_gates["2 qb gate"] == 0:
        MC_file_path = f"MC_results/MC_results_{nshots_GST}.txt"
        with open(MC_file_path, "w") as file:
            print(
                "Run, idx_1qb_rho, idx_1qb_BO, idx_1qb_Q, sgn_1qb_rho, sgn_1qb_BO, sgn_1qb_Q, sgn_total, measurement_outcome",
                file=file,
            )

    elif type_of_gates["2 qb gate"] >= 1 and type_of_gates["1 qb gate"] == 0:
        MC_file_path = f"MC_results/MC_results_{nshots_GST}.txt"
        with open(MC_file_path, "w") as file:
            print(
                "Run, idx_1qb_rho, idx_2qb_BO, idx_1qb_Q, sgn_1qb_rho, sgn_2qb_BO, sgn_1qb_Q, sgn_total, measurement_outcome",
                file=file,
            )

    elif type_of_gates["2 qb gate"] >= 1 and type_of_gates["1 qb gate"] >= 1:
        MC_file_path = f"MC_results/MC_results_{nshots_GST}.txt"
        with open(MC_file_path, "w") as file:
            print(
                "Run, idx_1qb_rho, idx_1qb_BO, idx_2qb_BO, idx_1qb_Q, sgn_1qb_rho, sgn_1qb_BO, sgn_2qb_BO, sgn_1qb_Q, sgn_total, measurement_outcome",
                file=file,
            )

    MC_results = []
    all_data = []
    QEM_matrix = np.zeros((2**circuit.nqubits, 2))

    for idx_run in range(1, nshots_MC + 1):
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

        # if type_of_gates['1 qb gate'] >= 1 and type_of_gates['2 qb gate'] == 0:
        #     index_of_1qb_rho = np.where(CDF_rho_1qb >= np.random.rand())[0][0]
        #     sgn_of_1qb_rho = int(np.sign(qrhovector_1qb[index_of_1qb_rho]))

        # elif type_of_gates['2 qb gate'] >= 1 and type_of_gates['1 qb gate'] == 0:
        #     index_of_2qb_rho = np.where(CDF_rho_2qb >= np.random.rand())[0][0]
        #     sgn_of_2qb_rho = int(np.sign(qrhovector_2qb[index_of_2qb_rho]))

        # elif type_of_gates['2 qb gate'] >= 1 and type_of_gates['1 qb gate'] >= 1:
        #     index_of_2qb_rho = np.where(CDF_rho_2qb >= np.random.rand())[0][0]
        #     sgn_of_2qb_rho = int(np.sign(qrhovector_2qb[index_of_2qb_rho]))

        ## ADD BASIS OPERATIONS
        # Do the 1qb and 2qb basis operations respectively.
        for data in circuit.raw["queue"]:
            # print(data)
            num_qubit_gate = len(data["init_args"])
            name_qubit_gate = data["name"]
            class_qubit_gate = data["_class"]
            ctrl_qb = data["_control_qubits"]
            targ_qb = data["_target_qubits"]

            ## SINGLE QUBIT OPERATOR + BASIS OPERATION
            if num_qubit_gate == 1:
                if class_qubit_gate != "M":
                    # print(f'Sample {num_qubit_gate}-qubit gate CDF')
                    MC_circ.add(getattr(gates, class_qubit_gate)(targ_qb[0]))

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
                MC_circ.add(getattr(gates, class_qubit_gate)(ctrl_qb[0], targ_qb[0]))

                ## Add 2qb basis operation
                top_register = ctrl_qb[0]  # 0
                bottom_register = targ_qb[0]  # 1

                index_of_2qb_BO = np.where(
                    CDF_O_2qb_matrix[:, 3 + count_2qb_gate] >= np.random.rand()
                )[0][
                    0
                ]  # Start from 3rd column (0th indexing)
                index_of_2qb_BO_vec.append(index_of_2qb_BO)

                sgn_of_2qb_BO = int(np.sign(qOvector_2qb[index_of_2qb_BO, idx_ops]))
                sgn_of_2qb_BO_vec.append(sgn_of_2qb_BO)

                index_of_2qb_BO_top = int(CDF_O_2qb_matrix[index_of_2qb_BO, 1])
                index_of_2qb_BO_bottom = int(CDF_O_2qb_matrix[index_of_2qb_BO, 2])

                if index_of_2qb_BO < 169:
                    # ===============#=========================#
                    # 16 SCENARIOS  #  13^2 BASIS OPERATIONS  #
                    # =========================================#
                    if (
                        index_of_2qb_BO_top < 10 and index_of_2qb_BO_bottom < 10
                    ):  ### SINGLE QUBIT BASIS OP ON TOP AND BOTTOM SEPARATELY
                        MC_circ.add(
                            gates.Unitary(
                                BasisOps_169[index_of_2qb_BO_top],
                                top_register,
                                trainable=False,
                                name="%s" % (gatenames[index_of_2qb_BO_top]),
                            )
                        )
                        MC_circ.add(
                            gates.Unitary(
                                BasisOps_169[index_of_2qb_BO_bottom],
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
                                BasisOps_169[index_of_2qb_BO_top],
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
                                BasisOps_169[index_of_2qb_BO_top],
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
                                BasisOps_169[index_of_2qb_BO_top],
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
                                BasisOps_169[index_of_2qb_BO_bottom],
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
                                BasisOps_169[index_of_2qb_BO_bottom],
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
                                BasisOps_169[index_of_2qb_BO_bottom],
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

                elif index_of_2qb_BO >= 169:
                    MC_circ.add(
                        gates.Unitary(
                            BasisOps_241[index_of_2qb_BO - 169],
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

        # if type_of_gates['2 qb gate'] >= 1 and type_of_gates['1 qb gate'] == 0:
        #     index_of_2qb_Q =  np.where(CDF_Q_2qb >= np.random.rand())[0][0]
        #     sgn_of_2qb_Q = int(np.sign(qQvector_2qb[index_of_2qb_Q]))

        #     top_register = ctrl_qb[0] # 0
        #     bottom_register = targ_qb[0] # 1
        #     MC_circ = measurements_top_and_bottom_registers(MC_circ, index_of_2qb_Q, top_register, bottom_register)

        # elif type_of_gates['1 qb gate'] >= 1 and type_of_gates['2 qb gate'] == 0:
        #     for _ in range(0,circuit.nqubits):
        #         index_of_1qb_Q =  np.where(CDF_Q_1qb >= np.random.rand())[0][0]
        #         sgn_of_1qb_Q = int(np.sign(qQvector_1qb[index_of_1qb_Q]))
        #         register = targ_qb[0]
        #         MC_circ = measurements_single_register(MC_circ, index_of_1qb_Q, register)

        # elif type_of_gates['2 qb gate'] >= 1 and type_of_gates['1 qb gate'] >= 1:
        #     index_of_2qb_Q =  np.where(CDF_Q_2qb >= np.random.rand())[0][0]
        #     sgn_of_2qb_Q = int(np.sign(qQvector_2qb[index_of_2qb_Q]))

        ## COMPUTE TOTAL SGN
        if type_of_gates["2 qb gate"] >= 1 and type_of_gates["1 qb gate"] >= 1:
            total_sgns = [
                sgn_of_1qb_rho_vec,
                sgn_of_1qb_BO_vec,
                sgn_of_2qb_BO_vec,
                sgn_of_1qb_Q_vec,
            ]
        elif type_of_gates["1 qb gate"] >= 1 and type_of_gates["2 qb gate"] == 0:
            total_sgns = [sgn_of_1qb_rho_vec, sgn_of_1qb_BO_vec, sgn_of_1qb_Q_vec]
        elif type_of_gates["2 qb gate"] >= 1 and type_of_gates["1 qb gate"] == 0:
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

        # print(MC_circ.draw())

        ## PERFORM 1 SHOT
        result = MC_circ.execute(nshots=1)
        counts = dict(result.frequencies(binary=True))
        # print(idx_run, counts)

        MC_output = int(list(counts)[0], 2)
        MC_output_binary = format(MC_output, "02b")
        # MC_run = [[idx_run], [index_of_rho], index_of_BO_vec, [index_of_Q], MC_output_binary]

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
        # with open(MC_file_path, 'a') as file:
        # print(idx_run, index_of_2qb_rho, *index_of_1qb_BO_vec, *index_of_2qb_BO_vec, index_of_2qb_Q, sgn_of_2qb_rho, *sgn_of_1qb_BO_vec, *sgn_of_2qb_BO_vec, sgn_of_2qb_Q, sgn_tot, MC_output_binary, file=file)

        if type_of_gates["1 qb gate"] >= 1 and type_of_gates["2 qb gate"] == 0:
            # print(idx_run, *index_of_1qb_rho_vec, *index_of_1qb_BO_vec, *index_of_1qb_Q_vec, *sgn_of_1qb_rho_vec, *sgn_of_1qb_BO_vec, *sgn_of_1qb_Q_vec, sgn_tot, MC_output_binary)
            MC_file_path = f"MC_results/MC_results_{nshots_GST}.txt"
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
            MC_file_path = f"MC_results/MC_results_{nshots_GST}.txt"
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
            MC_file_path = f"MC_results/MC_results_{nshots_GST}.txt"
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

        if np.remainder(idx_run, 1000) == 0:
            QEM_eff_counts = np.zeros((2**circuit.nqubits, 1))
            QEM_eff_prob = np.zeros((2**circuit.nqubits, 1))
            QEM_eff_counts[:, 0] = QEM_matrix[:, 0] - QEM_matrix[:, 1]

            QEM_eff_prob = QEM_eff_counts / np.sum(QEM_eff_counts[:, 0])
            print(f"prob_QEM with {idx_run} MC shots\n", QEM_eff_prob)

    QEM_eff_counts = np.zeros((2**circuit.nqubits, 1))
    QEM_eff_prob = np.zeros((2**circuit.nqubits, 1))
    QEM_eff_counts[:, 0] = QEM_matrix[:, 0] - QEM_matrix[:, 1]

    QEM_eff_prob = QEM_eff_counts / np.sum(QEM_eff_counts[:, 0])

    print(f"Final prob_QEM with {nshots_MC} Monte Carlo samples:\n", QEM_eff_prob)
    if (QEM_eff_prob < 0).any():
        print("Insufficient counts, negative values still exist in prob_QEM.")

    prob_QEM = {}
    for ii in range(0, 2**circuit.nqubits):
        key = bin(ii)[2:].zfill(circuit.nqubits)
        prob_QEM[key] = QEM_eff_prob[ii, 0]

    toc_PEC = time.time() - tic_PEC
    print("Total time: %.4f seconds" % (toc_PEC))

    return prob_QEM, Csample_total / np.sqrt(nshots_MC)
