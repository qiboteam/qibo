"""Error Mitigation Methods."""

import numpy as np
from scipy.optimize import curve_fit

from qibo import gates
from qibo.config import raise_error


def get_gammas(c, solve=True):
    """Standalone function to compute the ZNE coefficients given the noise levels.

    Args:
        c (numpy.ndarray): Array containing the different noise levels, note that in the CNOT insertion paradigm this corresponds to the number of CNOT pairs to be inserted. The canonical ZNE noise levels are obtained as 2*c + 1.
        solve (bool): If ``True`` computes the coeffients by solving the linear system. Otherwise, use the analytical solution valid for the CNOT insertion method.

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


def get_noisy_circuit(circuit, cj, insertion_gate="CNOT"):
    """Standalone function to generate the noisy circuit with the inverse gate pairs insertions.

    Args:
        circuit (qibo.models.circuit.Circuit): Input circuit to modify.
        cj (int): Number of insertion gate pairs to add.
        insertion_gate (str): Which gate to use for the insertion. Default value: 'CNOT', use 'RX' for the ``RX(pi/2)`` gate instead.

    Returns:
        qibo.models.circuit.Circuit: The circuit with the inserted CNOT pairs.
    """
    if insertion_gate not in ("CNOT", "RX"):  # pragma: no cover
        raise_error(
            ValueError,
            "Invalid insertion gate specification. Please select one between 'CNOT' and 'RX'.",
        )
    if insertion_gate == "CNOT" and circuit.nqubits < 2:  # pragma: no cover
        raise_error(
            ValueError,
            "Provide a circuit with at least 2 qubits when using the 'CNOT' insertion gate. Alternatively, try with the 'RX' insertion gate instead.",
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
                for i in range(cj):
                    noisy_circuit.add(gates.CNOT(control, target))
                    noisy_circuit.add(gates.CNOT(control, target))
            elif gate.init_kwargs["theta"] == theta:
                qubit = gate.qubits[0]
                for i in range(cj):
                    noisy_circuit.add(gates.RX(qubit, theta=theta))
                    noisy_circuit.add(gates.RX(qubit, theta=-theta))
    return noisy_circuit


def ZNE(
    circuit,
    observable,
    noise_levels,
    backend=None,
    noise_model=None,
    nshots=10000,
    solve_for_gammas=False,
    insertion_gate="CNOT",
):
    """Runs the Zero Noise Extrapolation method for error mitigation.

    The different noise levels are realized by the insertion of pairs of either ``CNOT`` or ``RX(pi/2)`` gates that resolve to the identiy in the noise-free case.

    Args:
        circuit (qibo.models.circuit.Circuit): Input circuit.
        observable (numpy.ndarray): Observable to measure.
        noise_levels (numpy.ndarray): Sequence of noise levels.
        backend (qibo.backends.abstract.Backend): Calculation engine.
        noise_model (qibo.noise.NoiseModel): Noise model applied to simulate noisy computation.
        nshots (int): Number of shots.
        solve_for_gammas (bool): If ``true``, explicitely solve the equations to obtain the gamma coefficients.
        insertion_gate (str): Which gate to use for the insertion. Default value: 'CNOT', use 'RX' for the ``RX(pi/2)`` gate instead.

    Returns:
        numpy.ndarray: Estimate of the expected value of ``observable`` in the noise free condition.
    """

    if backend == None:  # pragma: no cover
        from qibo.backends import GlobalBackend

        backend = GlobalBackend()
    expected_val = []
    for cj in noise_levels:
        noisy_circuit = get_noisy_circuit(circuit, cj, insertion_gate=insertion_gate)
        if noise_model != None and backend.name != "qibolab":
            noisy_circuit = noise_model.apply(noisy_circuit)
        circuit_result = backend.execute_circuit(noisy_circuit, nshots=nshots)
        expected_val.append(circuit_result.expectation_from_samples(observable))
    gamma = get_gammas(noise_levels, solve=solve_for_gammas)
    return (gamma * expected_val).sum()


def sample_training_circuit(
    circuit,
    replacement_gates=None,
    sigma=0.5,
):
    """Samples a training circuit for CDR by susbtituting some of the non-Clifford gates.

    Args:
        circuit (qibo.models.circuit.Circuit): Circuit to sample from, decomposed in ``RX(pi/2)``, ``X``, ``CNOT`` and ``RZ`` gates.
        replacement_gates (list): Candidates for the substitution of the non-Clifford gates. The list should be composed by tuples of the form (``gates.XYZ``, ``kwargs``). For example, phase gates are used by default: ``list((RZ, {'theta':0}), (RZ, {'theta':pi/2}), (RZ, {'theta':pi}), (RZ, {'theta':3*pi/2}))``.
        sigma (float): Standard devation of the gaussian used for sampling.

    Returns:
        qibo.models.circuit.Circuit: The sampled circuit.
    """
    if replacement_gates is None:
        replacement_gates = [(gates.RZ, {"theta": n * np.pi / 2}) for n in range(4)]
    # Find all the non-Clifford RZ gates
    gates_to_replace = []
    for i, gate in enumerate(circuit.queue):
        if isinstance(gate, gates.RZ):
            if gate.init_kwargs["theta"] % (np.pi / 2) != 0.0:
                gates_to_replace.append((i, gate))

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
    if len(gates_to_replace) == 0:
        raise_error(ValueError, "No non-Clifford RZ gate found, no circuit sampled.")
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
    backend,
    noise_model,
    nshots=10000,
    model=lambda x, a, b: a * x + b,
    n_training_samples=100,
    full_output=False,
):
    """Runs the CDR error mitigation method.

    Args:
        circuit (qibo.models.circuit.Circuit): Input circuit decomposed in the primitive gates: ``X``, ``CNOT``, ``RX(pi/2)``, ``RZ(theta)``.
        observable (numpy.ndarray): Observable to measure.
        backend (qibo.backends.abstract.Backend): Calculation engine.
        noise_model (qibo.noise.NoiseModel): Noise model used for simulating noisy computation.
        nshots (int): Number of shots.
        model : Model used for fitting. This should be a callable function object ``f(x, *params)`` taking as input the predictor variable and the parameters. By default a simple linear model ``f(x,a,b) := a*x + b`` is used.
        n_training_samples (int): Number of training circuits to sample.
        full_output (bool): If True, this function returns additional information: `val`, `optimal_params`, `train_val`.

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
        if noise_model != None and backend.name != "qibolab":
            c = noise_model.apply(c)
        circuit_result = backend.execute_circuit(c, nshots=nshots)
        val = circuit_result.expectation_from_samples(observable)
        train_val["noisy"].append(val)
    # Fit the model
    optimal_params = curve_fit(model, train_val["noisy"], train_val["noise-free"])[0]
    # Run the input circuit
    if noise_model != None and backend.name != "qibolab":
        noisy_circuit = noise_model.apply(circuit)
    circuit_result = backend.execute_circuit(noisy_circuit, nshots=nshots)
    val = circuit_result.expectation_from_samples(observable)
    mit_val = model(val, *optimal_params)
    # Return data
    if full_output == True:
        return mit_val, val, optimal_params, train_val
    else:
        return mit_val


def vnCDR(
    circuit,
    observable,
    backend,
    noise_levels,
    noise_model,
    nshots=10000,
    model=lambda x, *params: (x * np.array(params).reshape(-1, 1)).sum(0),
    n_training_samples=100,
    insertion_gate="CNOT",
    full_output=False,
):
    """Runs the vnCDR error mitigation method.

    Args:
        circuit (qibo.models.circuit.Circuit): Input circuit decomposed in the primitive gates: ``X``, ``CNOT``, ``RX(pi/2)``, ``RZ(theta)``.
        observable (numpy.ndarray): Observable to measure.
        backend (qibo.backends.abstract.Backend): Calculation engine.
        noise_levels (numpy.ndarray): Sequence of noise levels.
        noise_model (qibo.noise.NoiseModel): Noise model used for simulating noisy computation.
        nshots (int): Number of shots.
        model : Model used for fitting. This should be a callable function object ``f(x, *params)`` taking as input the predictor variable and the parameters. By default a simple linear model ``f(x,a) := a*x`` is used, with ``a`` beeing the diagonal matrix containing the parameters.
        n_training_samples (int): Number of training circuits to sample.
        insertion_gate (str): Which gate to use for the insertion. Default value: 'CNOT', use 'RX' for the ``RX(pi/2)`` gate instead.
        full_output (bool): If True, this function returns additional information: `val`, `optimal_params`, `train_val`.

    Returns:
        mit_val (float): Mitigated expectation value of `observable`.
        val (list): Expectation value of `observable` with increased noise levels.
        optimal_params (list): Optimal values for `params`.
        train_val (dict): Contains the noise-free and noisy expectation values obtained with the training circuits.
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
            if noise_model != None and backend.name != "qibolab":
                noisy_c = noise_model.apply(c)
            circuit_result = backend.execute_circuit(noisy_c, nshots=nshots)
            val = circuit_result.expectation_from_samples(observable)
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
        if noise_model != None and backend.name != "qibolab":
            noisy_c = noise_model.apply(circuit)
        circuit_result = backend.execute_circuit(noisy_c, nshots=nshots)
        val.append(circuit_result.expectation_from_samples(observable))
    mit_val = model(np.array(val).reshape(-1, 1), *optimal_params[0])[0]
    # Return data
    if full_output == True:
        return mit_val, val, optimal_params, train_val
    else:
        return mit_val
