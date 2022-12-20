import numpy as np

from qibo import gates, models


def noise_model(circuit, params):
    """Creates a noisy sample from the circuit given as argument.

    The function applies a :class:`qibo.gates.ThermalRelaxationChannel` after each step of the circuit
    and, after each gate, a :class:`qibo.gates.DepolarizingChannel`, whose parameter depends on whether the
    gate applies on one or two qubits. In the end on the samples are applied bitflips errors using
    the method `qibo.states.CircuitResult.apply_bitflips()`.


    Args:
        circuit (qibo.models.Circuit): Circuit on which noise will be applied. Since in the end are
        applied bitflips, measurement gates are required.
        params (dictionary): object which contains the parameters of the channels organized as follow
        params = {"t1" : (t1, t2,..., tn),
          "t2" : (t1, t2,..., tn),
          "gate time" : (time1, time2),
          "excited population": 0,
          "depolarizing error" : (lambda1, lambda2),
          "bitflips error" : ([p1, p2,..., pm], [p1, p2,..., pm])
         }
        Where n is the number of qubits, and m the number of measurement gates.
        The first four parameters are used by the thermal relaxation error. The first two  elements are the
        tuple containing the T_1 and T_2 parameters; the third one is a tuple which contain the gate times,
        for single and two qubit gates; then we have the excited population parameter.
        The fifth parameter is a tuple containing the depolaraziong errors for single and 2 qubit gate.
        The last parameter is a tuple containg the two arrays for bitflips probability errors: the first one implements 0->1 errors, the other one 1->0.

    Returns:
        circuit (qibo.models.Circuit)


    """
    t1 = params["t1"]
    t2 = params["t2"]
    time1 = params["gate_time"][0]
    time2 = params["gate_time"][1]
    excited_population = params["excited_population"]
    depolarizing_error_1 = params["depolarizing_error"][0]
    depolarizing_error_2 = params["depolarizing_error"][1]
    bitflips_01 = params["bitflips_error"][0]
    bitflips_10 = params["bitflips_error"][1]
    idle_qubits = params["idle_qubits"]

    noisy_circuit = models.Circuit(circuit.nqubits, density_matrix=True)

    time_steps = max(circuit.queue.moment_index)
    current_time = np.zeros(circuit.nqubits)
    for t in range(time_steps):
        for qubit in range(circuit.nqubits):
            if circuit.queue.moments[t][qubit] == None:
                pass
            elif isinstance(circuit.queue.moments[t][qubit], gates.measurements.M):
                for key in list(circuit.measurement_tuples):
                    if len(circuit.measurement_tuples[key]) > 1:
                        q1 = circuit.measurement_tuples[key][0]
                        q2 = circuit.measurement_tuples[key][1]
                        if current_time[q1] != current_time[q2] and idle_qubits == True:
                            q_min = q1
                            q_max = q2
                            if current_time[q1] > current_time[q2]:
                                q_min = q2
                                q_max = q1
                            time_difference = current_time[q_max] - current_time[q_min]
                            noisy_circuit.add(
                                gates.ThermalRelaxationChannel(
                                    q_min,
                                    t1[q_min],
                                    t2[q_min],
                                    time_difference,
                                    excited_population,
                                )
                            )
                            current_time[q_min] += time_difference
                q = circuit.queue.moments[t][qubit].qubits
                if len(circuit.queue.moments[t][qubit].qubits) == 1:
                    q = q[0]
                    noisy_circuit.add(gates.M(q, p0=bitflips_01[q], p1=bitflips_10[q]))
                else:
                    p0q = []
                    p1q = []
                    for j in q:
                        p0q.append(bitflips_01[j])
                        p1q.append(bitflips_10[j])
                    noisy_circuit.add(gates.M(*q, p0=p0q, p1=p1q))
                    circuit.queue.moments[t][
                        max(circuit.queue.moments[t][qubit].qubits)
                    ] = None
            elif len(circuit.queue.moments[t][qubit].qubits) == 1:
                noisy_circuit.add(circuit.queue.moments[t][qubit])
                noisy_circuit.add(
                    gates.DepolarizingChannel(
                        circuit.queue.moments[t][qubit].qubits, depolarizing_error_1
                    )
                )
                noisy_circuit.add(
                    gates.ThermalRelaxationChannel(
                        qubit,
                        t1[qubit],
                        t2[qubit],
                        time1,
                        excited_population,
                    )
                )
                current_time[qubit] += time1

            else:
                q1 = circuit.queue.moments[t][qubit].qubits[0]
                q2 = circuit.queue.moments[t][qubit].qubits[1]
                if current_time[q1] != current_time[q2] and idle_qubits == True:
                    q_min = q1
                    q_max = q2
                    if current_time[q1] > current_time[q2]:
                        q_min = q2
                        q_max = q1
                    time_difference = current_time[q_max] - current_time[q_min]
                    noisy_circuit.add(
                        gates.ThermalRelaxationChannel(
                            q_min,
                            t1[q_min],
                            t2[q_min],
                            time_difference,
                            excited_population,
                        )
                    )
                    current_time[q_min] += time_difference

                noisy_circuit.add(circuit.queue.moments[t][qubit])
                noisy_circuit.add(
                    gates.DepolarizingChannel(
                        tuple(set(circuit.queue.moments[t][qubit].qubits)),
                        depolarizing_error_2,
                    )
                )
                noisy_circuit.add(
                    gates.ThermalRelaxationChannel(
                        q1, t1[q1], t2[q1], time2, excited_population
                    )
                )
                noisy_circuit.add(
                    gates.ThermalRelaxationChannel(
                        q2, t1[q2], t2[q2], time2, excited_population
                    )
                )
                current_time[circuit.queue.moments[t][qubit].qubits[0]] += time2
                current_time[circuit.queue.moments[t][qubit].qubits[1]] += time2
                circuit.queue.moments[t][
                    max(circuit.queue.moments[t][qubit].qubits)
                ] = None

    measurements = []
    for m in circuit.measurements:
        q = m.qubits
        if len(q) == 1:
            q = q[0]
            measurements.append(gates.M(q, p0=bitflips_01[q], p1=bitflips_10[q]))
        else:
            p0q = []
            p1q = []
            for j in q:
                p0q.append(bitflips_01[j])
                p1q.append(bitflips_10[j])
            measurements.append(gates.M(*q, p0=p0q, p1=p1q))
    noisy_circuit.measurements = measurements

    return noisy_circuit


def hellinger_distance(p, q):
    """Hellinger distance between two discrete distributions.

    Args:
        p (collections.Counter): First frequencies.
        q (collections.Counter): Second frequencies.

    Returns:
        Hellinger distance between p and q.
    """
    nqubits = len(list(p.keys())[0])
    sum = 0
    for k in range(2**nqubits):
        index = "{:b}".format(k).zfill(nqubits)
        p_i = p[index]
        q_i = q[index]
        sum += (np.sqrt(p_i) - np.sqrt(q_i)) ** 2
    hellinger = np.sqrt(sum) / np.sqrt(2)
    return hellinger


def loss(parameters, *args):

    circuit = args[0]
    nshots = args[1]
    target_freq = args[2]
    idle_qubits = args[3]
    backend = args[4]
    qubits = circuit.nqubits
    parameters = np.array(parameters)
    # if any(parameters<0):
    #     return np.inf
    # elif parameters[2*qubits+2]>4/3 or parameters[2*qubits+3]>15/16:
    #     return np.inf
    # elif any(parameters[2*qubits+4:4*qubits+4]>1):
    #     return np.inf
    # elif any(2*parameters[0:qubits]-parameters[qubits:2*qubits] <0):
    #     return np.inf

    params = {
        "t1": tuple(parameters[0:qubits]),
        "t2": tuple(parameters[qubits : 2 * qubits]),
        "gate_time": tuple(parameters[2 * qubits : 2 * qubits + 2]),
        "excited_population": 0,
        "depolarizing_error": tuple(parameters[2 * qubits + 2 : 2 * qubits + 4]),
        "bitflips_error": (
            parameters[2 * qubits + 4 : 3 * qubits + 4],
            parameters[3 * qubits + 4 : 4 * qubits + 4],
        ),
        "idle_qubits": idle_qubits,
    }
    print(params)
    noisy_circuit = noise_model(circuit, params)
    freq = backend.execute_circuit(circuit=noisy_circuit, nshots=nshots).frequencies()
    norm = sum(freq.values())
    for k in freq:
        freq[k] /= norm
    print(hellinger_distance(target_freq, freq))
    return hellinger_distance(target_freq, freq)


class NoiseModel:
    def __init__(self):
        self.noisy_circuit = {}
        self.params = {}
        self.hellinger = {}
        self.hellinger0 = {}
        self.extra = {}

    def add(self, params):
        self.params = params

    def apply(self, circuit):
        self.noisy_circuit = noise_model(circuit, self.params)

    def fit(
        self,
        target_result,
        initial_params,
        method="trust-constr",
        jac=None,
        hess=None,
        hessp=None,
        bounds=True,
        constraints=True,
        tol=None,
        callback=None,
        options=None,
        compile=False,
        processes=None,
        backend=None,
    ):
        from qibo import optimizers

        if backend == None:
            from qibo.backends import GlobalBackend

            backend = GlobalBackend()

        circuit = target_result.circuit
        nshots = target_result.nshots
        target_freq = target_result.frequencies()
        idle_qubits = initial_params["idle_qubits"]
        norm = sum(target_freq.values())
        for k in target_freq:
            target_freq[k] /= norm
        qubits = target_result.nqubits
        if bounds == True:
            from scipy.optimize import Bounds

            qubits = target_result.nqubits
            lb = np.zeros(4 * qubits + 4)
            ub = [np.inf] * (2 * qubits + 2) + [4 / 3, 15 / 16] + [1] * 2 * qubits
            bounds = Bounds(lb, ub, keep_feasible=True)
        if constraints == True:
            from scipy.optimize import LinearConstraint

            qubits = target_result.nqubits
            cons = np.eye(4 * qubits + 4)
            for j in range(qubits):  # t1 t2
                cons[j, j] = 2
                cons[j, qubits + j] = 1
            lb = np.zeros(4 * qubits + 4)
            ub = [np.inf] * (2 * qubits + 2) + [4 / 3, 15 / 16] + [1] * 2 * qubits
            constraints = LinearConstraint(cons, lb, ub, keep_feasible=True)

        if tol == None:
            tol = 10 / np.sqrt(nshots)

        initial_params = (
            list(
                initial_params["t1"]
                + initial_params["t2"]
                + initial_params["gate_time"]
                + initial_params["depolarizing_error"]
            )
            + initial_params["bitflips_error"][0]
            + initial_params["bitflips_error"][1]
        )

        args = (circuit, nshots, target_freq, idle_qubits, backend)
        self.hellinger0 = loss(initial_params, *args)
        print(self.hellinger0)
        result, parameters, extra = optimizers.optimize(
            loss,
            initial_params,
            args=args,
            method=method,
            jac=jac,
            hess=hess,
            hessp=hessp,
            bounds=bounds,
            constraints=constraints,
            tol=tol,
            callback=callback,
            options=options,
            compile=compile,
            processes=processes,
            backend=backend,
        )
        params = {
            "t1": tuple(parameters[0:qubits]),
            "t2": tuple(parameters[qubits : 2 * qubits]),
            "gate_time": tuple(parameters[2 * qubits : 2 * qubits + 2]),
            "excited_population": 0,
            "depolarizing_error": tuple(parameters[2 * qubits + 2 : 2 * qubits + 4]),
            "bitflips_error": (
                parameters[2 * qubits + 4 : 3 * qubits + 4],
                parameters[3 * qubits + 4 : 4 * qubits + 4],
            ),
        }
        self.hellinger = result
        self.params = params
        self.extra = extra
