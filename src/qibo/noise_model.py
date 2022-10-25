# -*- coding: utf-8 -*-
from qibo import models,gates
from qibo.noise import NoiseModel, ThermalRelaxationError, DepolarizingError

def noise_model(Circuit,params):
    """Creates a noisy circuit from the circuit given as argument.
    
    The function applies a :class:`qibo.gates.ThermalRelaxationChannel` after each step of the circuit
    and, after each gate, a :class:`qibo.gates.DepolarizingChannel`, whose parameter depends on whether the
    gate applies on one or two qubits. In the end on the samples are applied bitflips errors using
    the :class:`qibo.gates.PauliNoiseChannel`.
    
    
    Args:
        Circuit (qibo.models.Circuit): Circuit on which noise will be applied. Since in the end are
        applied bitflips, measurement gates are required.
        Params (tuple): object which contains the parameters of the channels organized as follow
        ``((((t1,t2),...,(t1,t2)),(time_1,time_2),excited_population),(lam_1,lam_2),(px_1,...,px_m))``.
        The first element is a tuple which contains the parameters of the :class:`qibo.gates.ThermalRelaxationChannel`. The second element
        is a tuple which contains the parameter of the :class:`qibo.gates.DepolarizingChannel` for 1-qubit gate and 2-qubit gate, respectively.
        The third element is a dictionary "error map" for the bit flips.
        nshots (int): number of final samples.
        
    Returns:
        Circuit (qibo.models.Circuit) padded with noise channels.
    
    
    """
    Noisy_Circuit = Circuit.__class__(**Circuit.init_kwargs)
    
    time_steps = max(Circuit.queue.moment_index)
    time = time_steps*[0,]
    current_time = Circuit.nqubits*[0,]
    t1 = params[0][1][0]
    t2 = params[0][1][1]

    for t in range(time_steps):
        for qubit in range(Circuit.nqubits):
            if Circuit.queue.moments[t][qubit] == None :
                 pass
                    
            elif len(Circuit.queue.moments[t][qubit].qubits) == 1 :
                Noisy_Circuit.add(Circuit.queue.moments[t][qubit])
                Noisy_Circuit.add(gates.DepolarizingChannel(Circuit.queue.moments[t][qubit].qubits,params[1][0]))
                Noisy_Circuit.add(gates.ThermalRelaxationChannel(qubit, params[0][0][qubit][0], params[0][0][qubit][1], t1, params[0][2]))
                current_time[qubit] += t1
                
            else:
                q1 = Circuit.queue.moments[t][qubit].qubits[0]
                q2 = Circuit.queue.moments[t][qubit].qubits[1]
                if current_time[q1] != current_time[q2]:
                    q_min = q1
                    q_max = q2
                    if current_time[q1] > current_time[q2]:
                        q_min = q2
                        q_max = q1
                    time = current_time[q_max] - current_time[q_min]
                    Noisy_Circuit.add(gates.ThermalRelaxationChannel(q_min, params[0][0][q_min][0], params[0][0][q_min][1], time, params[0][2]))
                    current_time[q_min] += time
                    
                Noisy_Circuit.add(Circuit.queue.moments[t][qubit])
                Noisy_Circuit.add(gates.DepolarizingChannel(tuple(set(Circuit.queue.moments[t][qubit].qubits)),params[1][1]))
                Noisy_Circuit.add(gates.ThermalRelaxationChannel(q1, params[0][0][q1][0], params[0][0][q1][1], t2, params[0][2]))
                Noisy_Circuit.add(gates.ThermalRelaxationChannel(q2, params[0][0][q2][0], params[0][0][q2][1], t2, params[0][2]))
                current_time[Circuit.queue.moments[t][qubit].qubits[0]] += t2
                current_time[Circuit.queue.moments[t][qubit].qubits[1]] += t2
                Circuit.queue.moments[t][max(Circuit.queue.moments[t][qubit].qubits)] = None
    
    for key in list(Circuit.measurement_tuples):
        if len(Circuit.measurement_tuples[key]) > 1 :
            q1 = Circuit.measurement_tuples[key][0]
            q2 = Circuit.measurement_tuples[key][1]
            if current_time[q1] != current_time[q2]:
                q_min = q1
                q_max = q2
                if current_time[q1] > current_time[q2]:
                    q_min = q2
                    q_max = q1
                time = current_time[q_max] - current_time[q_min]
                Noisy_Circuit.add(gates.ThermalRelaxationChannel(q_min, params[0][0][q_min][0], params[0][0][q_min][1], time, params[0][2]))
                current_time[q_min] += time
    
    for q in Circuit.measurement_gate.qubits:
        Noisy_Circuit.add(gates.PauliNoiseChannel(q, px = params[2][q]))
        
    Noisy_Circuit.measurement_tuples = dict(Circuit.measurement_tuples)
    Noisy_Circuit.measurement_gate = Circuit.measurement_gate
                        
    return Noisy_Circuit
    
