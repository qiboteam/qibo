import numpy as np
from qibo import K
from qibo.hardware import pulses
from qibo.hardware.calibration import fitting, tasks

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
    final = K.experiment.static.ADC_length / K.experiment.static.ADC_sampling_rate
    step = 1 / K.experiment.static.ADC_sampling_rate
    ADC_time_array = np.arange(0, final, step)
    ADC_time_array = ADC_time_array[50:]

    ro_channel = static_config["channel"][2]
    # For now readout is done with mixers
    IF_frequency = static_config["resonator_frequency"] - K.experiment.static.lo_frequency # downconversion

    cos = np.cos(2 * np.pi * IF_frequency * ADC_time_array)
    it = np.sum(raw_data[ro_channel[0]] * cos)
    qt = np.sum(raw_data[ro_channel[1]] * cos)
    ampl = np.sqrt(it**2 + qt**2)
    phase = np.arctan2(qt, it) * 180 / np.pi

    return [it, qt, ampl, phase]

def _parse_batch_result(raw_data, static_config):
    steps = len(raw_data)
    res = np.zeros((4, steps))
    for i in range(steps):
        it, qt, ampl, phase = _parse_result(raw_data[i], static_config)
        res[0, i] = it
        res[1, i] = qt
        res[2, i] = ampl
        res[3, i] = phase

    return res

def _execute_pulse_sequences(scheduler, pulse_sequences, static_config):
    steps = len(pulse_sequences)
    res = np.zeros((4, steps))
    for i in range(steps):
        data = scheduler.execute_pulse_sequence(pulse_sequences[i], K.experiment.static.default_averaging).result()
        it, qt, ampl, phase = _parse_result(data, static_config)
        res[0, i] = it
        res[1, i] = qt
        res[2, i] = ampl
        res[3, i] = phase

    return res

def partial_qubit_calibration(static_config: dict, qubit: Qubit):
    """ Calibrate a qubit's frequency and pi-pulse only
    
    """
    log = {}

    # First determine the qubit frequency via pulse spectroscopy
    freq_start, freq_end = static_config["frequency_range"]
    channel = qubit.drive_channel
    amplitude = qubit.qubit_amplitude
    freq_sweep, seq = tasks.PulseSpectroscopy(freq_start, freq_end, amplitude, channel)
    res = K.scheduler.execute_batch_sequence(seq, K.experiment.static.default_averaging).result()
    res = _parse_batch_result(res, static_config)
    ampl_array = res[2]
    log["pulse"] = {
        "freq_sweep": freq_sweep.tolist(),
        "result": res.tolist()
    }
    freq = fitting.fit_pulse(ampl_array, freq_sweep)
    log["freq"] = freq
    qubit.qubit_frequency = freq

    # Next, do Rabi oscillation to determine pi-pulse time
    time_sweep, seq = tasks.RabiTime(0, 600e-9, 3e-9, freq, amplitude, channel)
    res = K.scheduler.execute_batch_sequence(seq, K.experiment.static.default_averaging).result()
    res = _parse_batch_result(res, static_config)
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
    freq_nyquist = freq - K.experiment.static.sampling_rate
    qubit.pulses["rx"] = [pulses.BasicPulse(channel, 0, pi_pulse, amplitude, freq_nyquist, 0, pulses.Rectangular())]
    qubit.pulses["ry"] = [pulses.BasicPulse(channel, 0, pi_pulse, amplitude, freq_nyquist, 90, pulses.Rectangular())]
    log["pi-pulse"] = pi_pulse

    return qubit, log

if __name__ == "__main__":
    import json
    static_config = K.experiment.static.qubit_static_parameters[0]
    qb = Qubit()
    qb.load_from_staic_config(static_config)
    qb, log = partial_qubit_calibration(static_config, qb)
    with open("./calib_test.json", "w+") as w:
        w.write(json.dumps(log))
