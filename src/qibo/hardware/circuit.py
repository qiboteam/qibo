import numpy as np
from qibo.hardware import pulses
from qibo.config import raise_error, HW_PARAMS


class PulseSequence:
    """Describes a sequence of pulses for the FPGA to unpack and convert into arrays

    Current FPGA binary has variable sampling rate but fixed sample size.
    Due to software limitations we need to prepare all 16 DAC channel arrays.
    @see BasicPulse, MultifrequencyPulse and FilePulse for more information about supported pulses.

    Args:
        pulses: Array of Pulse objects
    """
    def __init__(self, pulses):
        self.pulses = pulses
        self.nchannels = HW_PARAMS.nchannels
        self.sample_size = HW_PARAMS.sample_size
        self.sampling_rate = HW_PARAMS.sampling_rate
        self.file_dir = HW_PARAMS.pulse_file

        self.duration = self.sample_size / self.sampling_rate
        self.time = np.linspace(0, self.duration, num=self.sample_size)

    def compile(self) -> np.ndarray:
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
        return waveform

    def _compile_basic(self, waveform, pulse):
        i_start = int((pulse.start / self.duration) * self.sample_size)
        i_duration = int((pulse.duration / self.duration) * self.sample_size)
        envelope = pulse.shape.envelope(self.time, pulse.start, pulse.duration, pulse.amplitude)
        waveform[pulse.channel, i_start:i_start + i_duration] = envelope * np.sin(
            2 * np.pi * pulse.frequency * self.time[:i_duration] + pulse.phase)
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
