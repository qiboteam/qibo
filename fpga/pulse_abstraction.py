"""Contains the pulse abstraction and pulse shaping for the FPGA
"""

import numpy as np
from qibo.config import raise_error
from static_config import sample_size, sampling_rate, n_channels

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
        self.n_channels = n_channels
        self.sample_size = sample_size
        self.sampling_rate = sampling_rate

        self.duration = self.sample_size / self.sampling_rate
        self.time = np.linspace(0, self.duration, num=self.sample_size)

    def compile(self) -> np.ndarray:
        """Compiles pulse sequence into waveform arrays

        FPGA binary is currently unable to parse pulse sequences, so this is a temporary workaround to prepare the arrays

        Returns:
            Numpy.ndarray holding waveforms for each channel. Has shape (n_channels, sample_size).
        """
        waveform = np.zeros((self.n_channels, self.sample_size))
        for pulse in self.pulses:
            #if pulse.serial[0] == "P":
            if isinstance(pulse, BasicPulse):
                waveform = self._compile_basic(waveform, pulse)
            #elif pulse.serial[0] == "M":
            elif isinstance(pulse, MultifrequencyPulse):
                waveform = self._compile_multi(waveform, pulse)
            #elif pulse.serial[0] == "F":
            elif isinstance(pulse, FilePulse):
                waveform = self._compile_file(waveform, pulse)
            else:
                raise Exception("Invalid pulse type \"{}\"".format(pulse))
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
        arr = np.genfromtxt('C:/fpga_python/fpga/tmp/wave_ch1.csv', delimiter=',')[:-1]
        waveform[pulse.channel, i_start:i_start + len(arr)] = arr
        return waveform

    def serialize(self) -> str:
        """Returns the serialized pulse sequence
        """
        return ", ".join([p.serial() for p in self.pulses])

class Pulse:
    """Describes a pulse to be added onto the channel waveform
    """
    def __init__(self):
        self.channel = None

    def serial(self):
        """Returns the serialized pulse
        """
        raise_error(NotImplementedError)

    def __repr__(self):
        return self.serial()

class BasicPulse(Pulse):
    """Describes a single pulse to be added to waveform array.

    Args:
        channel (int): FPGA channel to play pulse on.
        start (float): Start time of pulse in seconds.
        duration (float): Pulse duration in seconds.
        amplitude (float): Pulse amplitude in volts.
        frequency (float): Pulse frequency in Hz.
        shape: (PulseShape): Pulse shape, @see Rectangular, Gaussian, DRAG for more information.
    """
    def __init__(self, channel, start, duration, amplitude, frequency, phase, shape):
        self.channel = channel
        self.start = start
        self.duration = duration
        self.amplitude = amplitude
        self.frequency = frequency
        self.phase = phase
        self.shape = shape  # PulseShape objects

    def serial(self):
        return "P({}, {}, {}, {}, {}, {}, {})".format(self.channel, self.start, self.duration,
                                                      self.amplitude, self.frequency, self.phase, self.shape)

class MultifrequencyPulse(Pulse):
    """Describes multiple pulses to be added to waveform array.

    Used when multiple pulses are overlapping to avoid overwrite
    """
    def __init__(self, members):
        self.members = members

    def serial(self):
        return "M({})".format(", ".join([m.serial() for m in self.members]))

class FilePulse(Pulse):
    """Commands the FPGA to load a file as a waveform array in the specified channel
    """
    def __init__(self, channel, start, filename):
        self.channel = channel
        self.start = start
        self.filename = filename

    def serial(self):
        return "F({}, {}, {})".format(self.channel, self.start, self.filename)

class PulseShape:
    """Describes the pulse shape to be used
    """
    def __init__(self):
        self.name = ""

    def envelope(self, time, start, duration, amplitude):
        raise_error(NotImplementedError)

    def __repr__(self):
        return "({})".format(self.name)

class Rectangular(PulseShape):
    """Rectangular/square pulse shape    
    """
    def __init__(self):
        self.name = "rectangular"

    def envelope(self, time, start, duration, amplitude):
        """Constant amplitude envelope
        """
        return amplitude

class Gaussian(PulseShape):
    """Gaussian pulse shape
    """
    def __init__(self, sigma):
        self.name = "gaussian"
        self.sigma = sigma

    def envelope(self, time, start, duration, amplitude):
        """Gaussian envelope centered with respect to the pulse:
        A\exp^{-\frac{1}{2}\frac{(t-\mu)^2}{\sigma^2}}
        """
        mu = start + duration / 2
        return amplitude * np.exp(-0.5 * (time - mu) ** 2 / self.sigma ** 2)

    def __repr__(self):
        return "({}, {})".format(self.name, self.sigma)

class Drag(PulseShape):
    """Derivative Removal by Adiabatic Gate (DRAG) pulse shape
    """
    def __init__(self, sigma, beta):
        self.name = "drag"
        self.sigma = sigma
        self.beta = beta

    def envelope(self, time, start, duration, amplitude):
        """DRAG envelope centered with respect to the pulse:
        G + i\beta(-\frac{t-\mu}{\sigma^2})G
        where Gaussian G = A\exp^{-\frac{1}{2}\frac{(t-\mu)^2}{\sigma^2}}
        """
        mu = start + duration / 2
        gaussian = amplitude * np.exp(-0.5 * (time - mu) ** 2 / self.sigma ** 2)
        return gaussian + 1j * self.beta * (-(time - mu) / self.sigma ** 2) * gaussian

    def __repr__(self):
        return "({}, {}, {})".format(self.name, self.sigma, self.beta)
