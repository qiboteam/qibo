"""Contains the pulse abstraction and pulse shaping for the FPGA."""
from abc import ABC, abstractmethod
from qibo.config import raise_error


class Pulse(ABC):
    """Describes a pulse to be added onto the channel waveform."""
    def __init__(self):
        self.channel = None

    @abstractmethod
    def serial(self):
        """Returns the serialized pulse."""
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
    """Gaussian pulse shape"""
    import numpy as np

    def __init__(self, sigma):
        self.name = "gaussian"
        self.sigma = sigma

    def envelope(self, time, start, duration, amplitude):
        """Gaussian envelope centered with respect to the pulse:
        A\exp^{-\frac{1}{2}\frac{(t-\mu)^2}{\sigma^2}}
        """
        mu = start + duration / 2
        return amplitude * self.np.exp(-0.5 * (time - mu) ** 2 / self.sigma ** 2)

    def __repr__(self):
        return "({}, {})".format(self.name, self.sigma)


class Drag(PulseShape):
    """Derivative Removal by Adiabatic Gate (DRAG) pulse shape"""
    import numpy as np

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
        gaussian = amplitude * self.np.exp(-0.5 * (time - mu) ** 2 / self.sigma ** 2)
        return gaussian + 1j * self.beta * (-(time - mu) / self.sigma ** 2) * gaussian

    def __repr__(self):
        return "({}, {}, {})".format(self.name, self.sigma, self.beta)


class SWIPHT(PulseShape):
    """Speeding up Wave forms by Inducing Phase to Harmful Transitions pulse shape"""

    def __init__(self, g):
        self.name = "SWIPHT"
        self.g = g

    def envelope(self, time, start, duration, amplitude):
        import numpy as np

        ki_qq = self.g * np.pi
        t_g = 5.87 / (2 * abs(ki_qq))
        t = time
        gamma = 138.9 * (t / t_g)**4 *(1 - t / t_g)**4 + np.pi / 4
        gamma_1st = 4 * 138.9 * (t / t_g)**3 * (1 - t / t_g)**3 * (1 / t_g - 2 * t / t_g**2)
        gamma_2nd = 4*138.9*(t / t_g)**2 * (1 - t / t_g)**2 * (14*(t / t_g**2)**2 - 14*(t / t_g**3) + 3 / t_g**2)
        omega = gamma_2nd / np.sqrt(ki_qq**2 - gamma_1st**2) - 2*np.sqrt(ki_qq**2 - gamma_1st**2) * 1 / np.tan(2 * gamma)
        omega = omega / max(omega)

        return omega

    def __repr__(self):
        return "({}, {})".format(self.name, self.g)
