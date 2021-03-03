import copy
import bisect
import numpy as np
from qibo.abstractions import circuit
from qibo.config import raise_error
from qibo.hardware import pulses, experiment

def square(t, start, duration, amplitude, freq, I_phase, Q_phase):
    x = amplitude * (1 * (start < t) & 1 * (start+duration > t))
    I_phase = I_phase * np.pi / 180
    Q_phase = Q_phase * np.pi / 180
    i = x * np.cos(2 * np.pi * freq * t + I_phase)
    q = - x * np.sin(2 * np.pi * freq * t + Q_phase)
    return i, q

def TTL(t, start, duration, amplitude):
    x = amplitude * (1 * (start < t) & 1 * (start + duration > t))
    return x


class PulseSequence:
    """Describes a sequence of pulses for the FPGA to unpack and convert into arrays

    Current FPGA binary has variable sampling rate but fixed sample size.
    Due to software limitations we need to prepare all 16 DAC channel arrays.
    @see BasicPulse, MultifrequencyPulse and FilePulse for more information about supported pulses.

    Args:
        pulses: Array of Pulse objects
    """
    def __init__(self, pulses, duration=None):
        self.pulses = pulses
        self.nchannels = experiment.static.nchannels
        self.sample_size = experiment.static.sample_size
        self.sampling_rate = experiment.static.sampling_rate
        self.file_dir = experiment.static.pulse_file

        if duration is None:
            self.duration = self.sample_size / self.sampling_rate
        else:
            self.duration = duration
            self.sample_size = int(duration * self.sampling_rate)
        end = experiment.static.readout_start_time + experiment.static.readout_pulse_duration + 1e-6
        self.time = np.linspace(end - self.duration, end, num=self.sample_size)

    def compile(self):
        """Compiles pulse sequence into waveform arrays

        FPGA binary is currently unable to parse pulse sequences, so this is a temporary workaround to prepare the arrays

        Returns:
            Numpy.ndarray holding waveforms for each channel. Has shape (nchannels, sample_size).
        """
        waveform = np.zeros((self.nchannels, self.sample_size))
        for pulse in self.pulses:
            #if pulse.serial[0] == "P":
            if isinstance(pulse, pulses.BasicPulse):
                waveform = self._compile_basic(waveform, pulse)
            #elif pulse.serial[0] == "M":
            elif isinstance(pulse, pulses.MultifrequencyPulse):
                waveform = self._compile_multi(waveform, pulse)
            #elif pulse.serial[0] == "F":
            elif isinstance(pulse, pulses.FilePulse):
                waveform = self._compile_file(waveform, pulse)
            else:
                raise_error(TypeError, "Invalid pulse type {}.".format(pulse))
        # Hardcoded bypass for now
        i_wfm, q_wfm, adc_ttl, ro_ttl, qb_ttl = self._generate_readout_pulses(self.time)
        waveform[0] = i_wfm
        waveform[1] = q_wfm
        waveform[4] = adc_ttl
        waveform[5] = ro_ttl
        waveform[6] = qb_ttl
        return waveform

    def _compile_basic(self, waveform, pulse):
        i_start = bisect.bisect(self.time, pulse.start)
        #i_start = int((pulse.start / self.duration) * self.sample_size)
        i_duration = int((pulse.duration / self.duration) * self.sample_size)
        envelope = pulse.shape.envelope(self.time, pulse.start, pulse.duration, pulse.amplitude)
        waveform[pulse.channel, i_start:i_start + i_duration] += envelope * np.sin(2 * np.pi * pulse.frequency * self.time[i_start:i_start + i_duration] + pulse.phase)
        return waveform

    def _compile_multi(self, waveform, pulse):
        for m in pulse.members:
            if m.serial[0] == "P":
                waveform += self._compile_basic(waveform, m)
            elif m.serial[0] == "F":
                waveform += self._compile_file(waveform, m)
        return waveform

    def _compile_file(self, waveform, pulse):
        i_start = int((pulse.start / self.duration) * self.sample_size)
        arr = np.genfromtxt(self.file_dir, delimiter=',')[:-1]
        waveform[pulse.channel, i_start:i_start + len(arr)] = arr
        return waveform

    def serialize(self):
        """Returns the serialized pulse sequence."""
        return ", ".join([pulse.serial() for pulse in self.pulses])

    @staticmethod
    def _generate_readout_pulses(time_array):
        # TODO: Fix for multiplexed readout
        i_wfm, q_wfm = square(time_array, experiment.static.readout_start_time, experiment.static.readout_pulse_duration,
                              experiment.static.readout_pulse_amplitude, experiment.static.readout_IF_frequency, experiment.static.readout_phase[0],
                              experiment.static.readout_phase[1])
        # ADC TTL
        start = experiment.static.readout_start_time + experiment.static.ADC_delay
        adc_ttl = TTL(time_array, start, 10e-9, 1)

        # RO SW TTL
        start = experiment.static.readout_start_time + experiment.static.RO_SW_delay
        ro_ttl = TTL(time_array, start, experiment.static.readout_pulse_duration, 1)

        # QB SW TTL
        start = experiment.static.readout_start_time + experiment.static.QB_SW_delay
        qb_ttl = TTL(time_array, start, experiment.static.readout_pulse_duration, 1)

        return i_wfm, q_wfm, adc_ttl, ro_ttl, qb_ttl

class Circuit(circuit.AbstractCircuit):

    def __init__(self, nqubits, scheduler=None):
        super().__init__(nqubits)
        self.scheduler = scheduler

    def _add_layer(self):
        raise_error(NotImplementedError)

    def fuse(self):
        raise_error(NotImplementedError)

    @staticmethod
    def _probability_extraction(data, refer_0, refer_1):
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

    def execute(self, nshots):
        if self.scheduler is None:
            raise_error(RuntimeError, "Cannot execute circuit on hardware if "
                                      "scheduler is not provided.")

        qubit_times = np.zeros(self.nqubits)
        # Get calibration data
        self.qubit_config = self.scheduler.fetch_config()
        # compile pulse sequence
        pulse_sequence = [pulse for gate in self.queue
            for pulse in gate.pulse_sequence(self.qubit_config, qubit_times)]
        pulse_sequence = PulseSequence(pulse_sequence)
        # execute using the scheduler
        self._final_state = self.scheduler.execute_pulse_sequence(pulse_sequence, nshots)
        return self._final_state

    def __call__(self, nshots):
        return self.execute(nshots)

    @property
    def final_state(self):
        if self._final_state is None:
            raise_error(RuntimeError)
        return self._final_state

    def parse_result(self, qubit):
        final = experiment.static.sample_size / experiment.static.ADC_sampling_rate
        step = 1 / experiment.static.ADC_sampling_rate
        ADC_time_array = np.arange(0, final, step)
        ADC_time_array = ADC_time_array[50:]

        static_data = experiment.static.qubit_static_parameters[self.qubit_config[qubit]["id"]]
        ro_channel = static_data["channel"][2]
        # For now readout is done with mixers
        IF_frequency = static_data["resonator_frequency"] - experiment.static.lo_frequency # downconversion

        raw_data = self.final_state.result()
        cos = np.cos(2 * np.pi * IF_frequency * ADC_time_array)
        it = np.sum(raw_data[ro_channel[0]] * cos)
        qt = np.sum(raw_data[ro_channel[1]] * cos)
        data = np.array([it, qt])
        ref_zero = np.array(self.qubit_config[qubit]["iq_state"]["0"])
        ref_one = np.array(self.qubit_config[qubit]["iq_state"]["1"])
        return self._probability_extraction(data, ref_zero, ref_one)
