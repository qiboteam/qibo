"""Hardware static parameters."""

num_qubits = 2
sampling_rate = 2.3e9
nchannels = 4
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

pulse_file = 'C:/fpga_python/fpga/tmp/wave_ch1.csv'

from qibo.hardware.pulses import BasicPulse, Rectangular
calibration_placeholder = [{
    "id": 0,
    "qubit_frequency": 3.0473825e9,
    "qubit_amplitude": 0.75 / 2,
    "T1": 5.89e-6,
    "T2": 1.27e-6,
    "T2_Spinecho": 3.5e-6,
    "pi-pulse": 24.78e-9,
    "drive_channel": 3,
    "readout_channel": (0, 1),
    "iq_state": {
        "0": [0.016901687416102748, -0.006633150376482062],
        "1": [0.009458352995780546, -0.008570922209494462]
    },
    "gates": {
        "rx": [BasicPulse(3, 0, 24.78e-9, 0.375, 3.0473825e9 - sampling_rate, 0, Rectangular())],
        "ry": [BasicPulse(3, 0, 24.78e-9, 0.375, 3.0473825e9 - sampling_rate, 90, Rectangular())],
    }
}]
