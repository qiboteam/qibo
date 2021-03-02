import numpy as np
from qibo.hardware import experiment, pulses
from qibo.hardware.calibration import fitting, tasks

default_averaging = experiment.static.default_averaging

class Qubit:
    def __init__(self):
        self.id = None
        self.qubit_amplitude = None
        self.qubit_frequency = None
        self.drive_channel = None
        self.T1 = None
        self.T2_Spinecho = None
        self.readout_channel = ()
        self.iq_state = {}
        self.pulses = {}

    def load_from_staic_config(self, static_config):
        self.id = static_config["id"]
        self.qubit_amplitude = static_config["amplitude"]
        self.drive_channel = static_config["channel"][0]

def _parse_result(raw_data, static_config):
    final = experiment.static.sample_size / experiment.static.ADC_sampling_rate
    step = 1 / experiment.static.ADC_sampling_rate
    ADC_time_array = np.arange(0, final, step)

    ro_channel = static_config["channel"][2]
    # For now readout is done with mixers
    IF_frequency = static_config["resonator_frequency"] - experiment.static.lo_frequency # downconversion

    cos = np.cos(2 * np.pi * IF_frequency * ADC_time_array)
    it = np.sum(raw_data[ro_channel[0]] * cos)
    qt = np.sum(raw_data[ro_channel[1]] * cos)
    ampl = np.sqrt(it**2 + qt**2)
    phase = np.arctan2(qt, it) * 180 / np.pi

    return [it, qt, ampl, phase]

def _execute_pulse_sequences(scheduler, pulse_sequences, static_config):
    steps = len(pulse_sequences)
    res = np.zeros((4, steps))
    for i in range(steps):
        data = scheduler.execute_pulse_sequence(pulse_sequences[i], static_config).result()
        it, qt, ampl, phase = _parse_result(data, static_config)
        res[0, i] = it
        res[1, i] = qt
        res[2, i] = ampl
        res[3, i] = phase

    return res

def partial_qubit_calibration(static_config: dict, qubit: Qubit, scheduler):
    """ Calibrate a qubit's frequency and pi-pulse only
    
    """
    log = {}

    # First determine the qubit frequency via pulse spectroscopy
    freq_start, freq_end = static_config["frequency_range"]
    channel = qubit.drive_channel
    amplitude = qubit.qubit_amplitude
    freq_sweep, seq = tasks.PulseSpectroscopy(freq_start, freq_end, amplitude, channel)
    res = _execute_pulse_sequences(scheduler, seq, static_config)
    ampl_array = res[2]
    log["pulse"] = {
        "freq_sweep": freq_sweep.tolist(),
        "result": res.tolist()
    }
    freq = fitting.fit_pulse(ampl_array, freq_sweep)
    qubit.qubit_frequency = freq

    # Next, do Rabi oscillation to determine pi-pulse time
    time_sweep, seq = tasks.RabiTime(0, 600e-9, 3e-9, freq, amplitude, channel)
    res = _execute_pulse_sequences(scheduler, seq, static_config)
    log["rabi"] = {
        "time_sweep": time_sweep.tolist(),
        "result": res.tolist()
    }
    pi_pulse = fitting.fit_rabi(ampl_array, time_sweep)
    idx_zero = np.argmax(ampl_array)
    idx_one = np.argmin(ampl_array)
    qubit.iq_state = {
        "0": [res[0, idx_zero], res[1, idx_zero]],
        "1": [res[0, idx_one], res[1, idx_one]]
    }
    freq_nyquist = freq - experiment.static.sampling_rate
    qubit["rx"] = [pulses.BasicPulse(channel, 0, pi_pulse, amplitude, freq_nyquist, 0, pulses.Rectangular())]
    qubit["ry"] = [pulses.BasicPulse(channel, 0, pi_pulse, amplitude, freq_nyquist, 90, pulses.Rectangular())]

    return qubit, log
