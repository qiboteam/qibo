#TODO: maybe move to qibo.config

#from pulse_abstraction import BasicPulse, Rectangular

num_qubits = 2
sampling_rate = 2.3e9
n_channels = 4
sample_size = 32000
readout_pulse_type = "IQ"
readout_pulse_duration = 5e-6
readout_pulse_amplitude = 0.75
lo_frequency = 4.51e9
readout_nyquist_zone = 4
ADC_sampling_rate = 2e9
qubit_static_parameters = [
    {
        "id": 0,
        "channel": [2, None, [0, 1]], # XY control, Z line, readout
        "frequency_range": [2.6e9, 2.61e9],
        "resonator_frequency": 4.5241e9,
        "neighbours": [2]
    }, {
        "id": 1,
        "channel": [3, None, [0, 1]],
        "frequency_range": [3.14e9, 3.15e9],
        "resonator_frequency": 4.5241e9,
        "neighbours": [1]
    }
]
dac_mode_for_nyquist = ["NRZ", "MIX", "MIX", "NRZ"] # fifth onwards not calibrated yet
