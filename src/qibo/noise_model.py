# -*- coding: utf-8 -*-
import numpy as np

from qibo import gates, models
from qibo.gates import DepolarizingChannel, ThermalRelaxationChannel
from qibo.noise import DepolarizingError, NoiseModel, ThermalRelaxationError


def noise_model(circuit, params):
    """Creates a noisy circuit from the circuit given as argument.

    The function applies a :class:`qibo.gates.ThermalRelaxationChannel` after each step of the circuit
    and, after each gate, a :class:`qibo.gates.DepolarizingChannel`, whose parameter depends on whether the
    gate applies on one or two qubits. In the end on the samples are applied bitflips errors using
    the :class:`qibo.gates.PauliNoiseChannel`.


    Args:
        circuit (qibo.models.Circuit): Circuit on which noise will be applied. Since in the end are
        applied bitflips, measurement gates are required.
        params (dictionary): object which contains the parameters of the channels organized as follow
        params = {"t1" : (t1, t2,..., tn),
          "t2" : (t1, t2,..., tn),
          "gate time" : (time1, time2),
          "excited population": 0,
          "depolarizing error" : (lambda1, lambda2),
          "bitflips error" : (p1, p2,..., pm)
         }
        Where n is the number of qubits, and m the number of measurement gates.
        The first four parameters are used by the thermal relaxation error. The first two  elements are the
        tuple containing the T_1 and T_2 parameters; the third one is a tuple which contain the gate times,
        for single and two qubit gates; then we have the excited population parameter.
        The fifth parameter is a tuple containing the depolaraziong errors for single and 2 qubit gate.
        The last parameter is a m-long tuple of probabilities for bitflips error.

    Returns:
        Circuit (qibo.models.Circuit) padded with noise channels.


    """
    t1 = params["t1"]
    t2 = params["t2"]
    time1 = params["gate time"][0]
    time2 = params["gate time"][1]
    excited_population = params["excited population"]
    depolarizing_error_1 = params["depolarizing error"][0]
    depolarizing_error_2 = params["depolarizing error"][1]
    bitflips = params["bitflips error"]

    noisy_circuit = models.Circuit(circuit.nqubits, density_matrix=True)

    time_steps = max(circuit.queue.moment_index)
    current_time = np.zeros(circuit.nqubits)

    for t in range(time_steps):
        for qubit in range(circuit.nqubits):
            if circuit.queue.moments[t][qubit] == None:
                pass

            elif len(circuit.queue.moments[t][qubit].qubits) == 1:
                noisy_circuit.add(circuit.queue.moments[t][qubit])
                # noisy_circuit.add(gates.PauliNoiseChannel(qubit, 0.1, 0.0, 0.2))
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
                if current_time[q1] != current_time[q2]:
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
                # for q in circuit.queue.moments[t][qubit].qubits:
                #    noisy_circuit.add(gates.PauliNoiseChannel(q, 0.1, 0.0, 0.2))
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

    for key in list(circuit.measurement_tuples):
        if len(circuit.measurement_tuples[key]) > 1:
            q1 = circuit.measurement_tuples[key][0]
            q2 = circuit.measurement_tuples[key][1]
            if current_time[q1] != current_time[q2]:
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

    for q in circuit.measurement_gate.qubits:
        noisy_circuit.add(gates.PauliNoiseChannel(q, px=bitflips[q]))

    noisy_circuit.measurement_tuples = dict(circuit.measurement_tuples)
    noisy_circuit.measurement_gate = circuit.measurement_gate

    return noisy_circuit
