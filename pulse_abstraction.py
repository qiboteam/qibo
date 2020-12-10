import numpy as np


class PulseSequence:
    def __init__(self, pulses, n_channels, sample_size, sampling_rate):
        self.pulses = pulses
        self.n_channels = n_channels
        self.sample_size = sample_size
        self.sampling_rate = sampling_rate

        self.duration = self.sample_size / self.sampling_rate
        self.time = np.linspace(0, self.duration, num=self.sample_size)

    def compile(self):
        waveform = np.zeros((self.n_channels, self.sample_size))
        for pulse in self.pulses:
            if pulse.serial[0] == "P":
                waveform = self._compile_basic(waveform, pulse)
            elif pulse.serial[0] == "M":
                waveform = self._compile_multi(waveform, pulse)
            elif pulse.serial[0] == "F":
                waveform = self._compile_file(waveform, pulse)
            else:
                raise Exception("Invalid pulse type \"{}\"".format(pulse.serial[0]))
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

    def serialize(self):
        return ", ".join([p.serial for p in self.pulses])


class Pulse:
    def __init__(self, serial):
        self.serial = serial


class BasicPulse(Pulse):
    def __init__(self, channel, start, duration, amplitude, frequency, phase, shape):
        self.channel = channel
        self.start = start
        self.duration = duration
        self.amplitude = amplitude
        self.frequency = frequency
        self.phase = phase
        self.shape = shape  # PulseShape objects
        super().__init__(
            "P({}, {}, {}, {}, {}, {}, {})".format(channel, start, duration, amplitude, frequency, phase, shape.serial))


class MultifrequencyPulse(Pulse):
    def __init__(self, members):
        self.members = members
        super().__init__("M({})".format(", ".join([m.serial for m in members])))


class FilePulse(Pulse):
    def __init__(self, channel, start, filename):
        self.channel = channel
        self.start = start
        self.filename = filename
        super().__init__("F({}, {}, {})".format(channel, start, filename))


class PulseShape:
    def __init__(self, name, *args):
        self.parameters = []
        self.serial = "(" + name
        for p in args:
            self.parameters.append(p)
            self.serial += ", {}".format(p)
        self.serial += ")"

    def envelope(self, time, start, duration, amplitude):
        return amplitude * time


class Rectangular(PulseShape):
    def __init__(self):
        super().__init__("retangular")


class Gaussian(PulseShape):
    def __init__(self, sigma):
        super().__init__("gaussian", sigma)
        self.sigma = sigma

    def envelope(self, time, start, duration, amplitude):
        mu = start + duration / 2
        return amplitude * np.exp(-0.5 * (time - mu) ** 2 / self.sigma ** 2)


class Drag(PulseShape):
    def __init__(self, sigma, beta):
        super().__init__("drag", sigma, beta)
        self.sigma = sigma
        self.beta = beta

    def envelope(self, time, start, duration, amplitude):
        mu = start + duration / 2
        gaussian = amplitude * np.exp(-0.5 * (time - mu) ** 2 / self.sigma ** 2)
        return gaussian + 1j * self.beta * ((-(time - mu)) / self.sigma ** 2) * gaussian
