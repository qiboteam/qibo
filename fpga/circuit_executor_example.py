import copy
import numpy as np
from qibo.gates import I, RX, RY
from qibo.config import raise_error
from qibo.models import Circuit
from pulse_abstraction import *
from static_config import num_qubits, sampling_rate, lo_frequency, qubit_static_parameters, ADC_sampling_rate

def pulse_sequence_from_circuit(circuit: Circuit, qubit_config: dict):
    """Interim workaround to turn circuit object into pulses
    Currently only XY rotation implemented
    
    Args:
        circuit: Circuit object
    
    Returns:
        Pulse sequence representation of the circuit object
    """
    queue = circuit.queue
    qubit_time = np.zeros(num_qubits)
    sequence = []
    for gate in queue:
        q = gate.target_qubits
        if isinstance(gate, I):
            continue
        elif isinstance(gate, RX) or isinstance(gate, RY):
            gate_name = gate.name
            qubit = q[0]
            qubit_data = qubit_config[qubit]
            start_time = qubit_time[qubit]
            angle = gate.parameters
            if angle == 0:
                continue

            time_mod = abs(angle / np.pi)
            phase_mod = 0 if angle > 0 else -180
            pulses = copy.deepcopy(qubit_data["gates"][gate_name])
            pulses, end_time = _prepare_gate_pulses(pulses, start_time, time_mod, phase_mod)
            qubit_time[qubit] = end_time
            sequence += pulses
        else:
            raise_error(NotImplementedError)

    ps = PulseSequence(sequence)
    return ps
    

def _prepare_gate_pulses(gp, start, time_mod=1, phase_mod=0):
    for p in gp:
        duration = p.duration * time_mod
        p.start = start
        p.phase += phase_mod
        p.duration = duration
        start += duration
    return gp, start

def _probability_extraction(data: np.ndarray, refer_0: np.ndarray, refer_1: np.ndarray):
    move = copy.copy(refer_0)
    refer_0 = refer_0 - move
    refer_1 = refer_1 - move
    data = data - move
    # Rotate the data so that vector 0-1 is overlapping with Ox
    angle = copy.copy(np.arccos(refer_1[0]/np.sqrt(refer_1[0]**2 + refer_1[1]**2))*np.sign(refer_1[1]))
    new_data = np.array([data[0]*np.cos(angle) + data[1]*np.sin(angle),
                         -data[0]*np.sin(angle) + data[1]*np.cos(angle)])
    # Rotate refer_1 to get state 1 reference
    new_refer_1 = np.array([refer_1[0]*np.cos(angle) + refer_1[1]*np.sin(angle),
                            -refer_1[0]*np.sin(angle) + refer_1[1]*np.cos(angle)])
    # Condition for data outside bound
    if new_data[0] < 0:
        new_data[0] = 0
    elif new_data[0] > new_refer_1[0]:
        new_data[0] = new_refer_1[0]
    return new_data[0]/new_refer_1[0]

ADC_time_array = np.arange(0, sample_size / ADC_sampling_rate, 1 / ADC_sampling_rate)

def _parse_result(raw_data, qubit_config):
    static_data = qubit_static_parameters[qubit_config["id"]]
    ro_channel = static_data["channel"][2]
    # For now readout is done with mixers
    IF_frequency = static_data["resonantor_frequency"] - lo_frequency # downconversion
    it = np.sum(raw_data[ro_channel[0]] * np.cos(2 * np.pi * IF_frequency * ADC_time_array))
    qt = np.sum(raw_data[ro_channel[1]] * np.cos(2 * np.pi * IF_frequency * ADC_time_array))

    return it, qt

def parse_result(raw_data, qubit_config):
    i, q = _parse_result(raw_data, qubit_config)
    data = np.array([i, q])
    ref_zero = np.array(qubit_config["iq_state"]["0"])
    ref_one = np.array(qubit_config["iq_state"]["1"])
    return _probability_extraction(data, ref_zero, ref_one)


if __name__ == "__main__":
    from scheduler import TaskScheudler
    from randomized_benchmarking_example import randomized_benchmark
    import matplotlib.pyplot as plt

    qb = 0
    shots = 1000
    gates = 10

    circuits = randomized_benchmark(qb, gates)
    ts = TaskScheudler()
    if not ts.config_ready():
        #ts.poll_config()
        pass
    qubit_config = ts.fetch_config()
    qubit_data = qubit_config[qb]

    tasks = [ts.execute_pulse_sequence(pulse_sequence_from_circuit(c, qubit_config), shots) for c in circuits]
    results = [parse_result(f.result(), qubit_data) for f in tasks]

    sweep = range(1, gates + 1)
    plt.plot(sweep, results)
    plt.show()
