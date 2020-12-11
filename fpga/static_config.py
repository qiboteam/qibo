#TODO: maybe move to qibo.config

num_qubits = 2
sampling_rate = 2.3e9
n_channels = 4
sample_size = 32000
readout_pulse_generation = "IQ"
LO_frequency = 4.51e9
qubit_static_parameters = [
    {
        "id": 1,
        "channel": [2, None, [0, 1]],
        "frequency_range": [2.6e9, 2.61e9],
        "resonator_frequency": 4.5241e9,
        "neighbours": [2]
    }, {
        "id": 2,
        "channel": [3, None, [0, 1]],
        "frequency_range": [3.14e9, 3.15e9],
        "resonator_frequency": 4.5241e9,
        "neighbours": [1]
    }
]
