import numpy as np
from typing import List
from qibo.hardware.circuit import PulseSequence
from qibo.hardware import experiment
from qibo.hardware.pulses import BasicPulse, Rectangular

def PulseSpectroscopy(frequency_start: float, frequency_stop: float, qubit_amplitude: float, channel: int) -> List[PulseSequence]:
    """ Pulse spectroscopy task to determine qubit frequency
    In this task, the qubit is driven for a long time at different frequencies
    We can determine the resonant frequency from the frequency that provides the largest shift from the ground state
    Each pulse sequence corresponds to a 40 microsecond pulse of frequency between frequency start and end

    Args:
        frequency_start (float): Starting frequency of the sweep
        frequency_end (float): Ending frequency of the sweep
        qubit_amplitude (float): Amplitude of the driving signal
        channel (int): Qubit drive channel

    Returns:
        List of PulseSequence with 40 microsecond pulses sweeping across the specified frequency range
    """
    frequency_sweep = np.linspace(frequency_start, frequency_stop, 200)
    seq = []
    pulse_length = 40e-6
    duration = 50e-6
    pulse_start = experiment.static.readout_pulse_duration - pulse_length

    for freq in frequency_sweep:
        pulse_freq = freq - experiment.static.sampling_rate
        p = BasicPulse(channel, pulse_start, pulse_length, qubit_amplitude, pulse_freq, 0, Rectangular())
        ps = PulseSequence([p], duration)
        seq.append(ps)

    return frequency_sweep, seq

def RabiTime(time_start: float, time_stop: float, time_step: float, qubit_frequency: float, qubit_amplitude: float, channel: int) -> List[PulseSequence]:
    """ Rabi oscillation task to determine qubit pi-pulse duration
    In this task, the qubit is driven with sweeping pulse duration
    We determine the pi-pulse by fitting the results to a sine curve and obtaining the Rabi frequency
    Each pulse sequence corresponds to a pulse with duration between start and end

    Args:
        time_start (float): Starting pulse duration
        time_end (float): Last pulse duration
        time_step (float): Pulse sweep step time
        qubit_frequency (float): Driving frequency of the qubit
        qubit_amplitude (float): Amplitude of the driving signal
        channel (int): Qubit drive channel
    
    Returns:
        List of PulseSequence with pulses of duration spanning time_start and time_stop
    """

    duration = max(10e-6, time_stop + 7e-6)
    seq = []
    pulse_sweep = np.arange(time_start, time_stop, time_step)
    freq = qubit_frequency - experiment.static.sampling_rate

    for pulse_length in pulse_sweep:
        pulse_start = experiment.static.readout_pulse_duration - pulse_length

        p = BasicPulse(channel, pulse_start, pulse_length, qubit_amplitude, freq, 0, Rectangular())
        ps = PulseSequence([p], duration)
        seq.append(ps)

    return pulse_sweep, seq

def Ramsey(tau_start, tau_stop, tau_step, qubit_frequency, qubit_amplitude, pi_pulse, channel, phase_shift=0):
    """ Ramsey interferometry to determine qubit resonant frequency or T2* time
    In this task, the qubit is driven by two pi/2 pulses seperated by a delay tau which is spanned by tau_start and tau_stop
    We can fit the results to a decaying sine wave - the decay represents the T2* time and we can obtain the frequency of the oscillation as f
    If we drive the qubit at frequency omega, the resonant frequency would be omega +- f

    Args:
        tau_start (float): Starting delay duration
        tau_end (float): Last delay duration
        tau_step (float): Delay step time
        qubit_frequency (float): Driving frequency of the qubit
        qubit_amplitude (float): Amplitude of the driving signal
        pi-pulse (float): Pi-pulse duration of the qubit
        channel (int): Qubit drive channel
        phase_shift (int): Phase shift of the second pulse to provide some detuning for omega
    
    Returns:
        List of PulseSequence with two pi-half pulses seperated by delay tau spanned by tau_start and tau_stop
    
    """
    duration = max(10e-6, tau_stop + 7e-6)
    seq = []
    tau_sweep = np.arange(tau_start, tau_stop, tau_step)
    freq = qubit_frequency - experiment.static.sampling_rate
    pi_half = pi_pulse / 2
    p2_start = experiment.static.readout_pulse_duration - pi_half
    p2 = BasicPulse(channel, p2_start, pi_half, 0.75 / 2, freq, phase_shift, Rectangular())

    for tau in tau_sweep:
        p1_start = p2_start - tau - pi_half
        p1 = BasicPulse(channel, p1_start, pi_half, 0.75 / 2, freq, 0, Rectangular())
        
        ps = PulseSequence([p1, p2], duration)
        seq.append(ps)

    return tau_sweep, seq

def Spinecho(qubit_frequency, qubit_amplitude, pi_pulse, channel, tau_start=0, tau_stop=2e-5, tau_step=1e-7):
    """ Spinecho experiment to determine T2 time
    In this task, the qubit is driven with a pi/2 pulse, followed by a pi-pulse and another pi/2 pulse seperated by a total delay tau
    We determine the T2 time from fitting the exponential decay from the ground state to the mixed state

    Args:
        qubit_frequency (float): Driving frequency of the qubit
        qubit_amplitude (float): Amplitude of the driving signal
        pi-pulse (float): Pi-pulse duration of the qubit
        channel (int): Qubit drive channel
        tau_start (float): Starting delay duration
        tau_end (float): Last delay duration
        tau_step (float): Delay step time
    
    Returns:
        List of PulseSequence with two pi-half pulses seperated by delay tau spanned by tau_start and tau_stop
    
    """
    duration = max(3e-5, tau_stop + 7e-6)
    seq = []
    tau_sweep = np.arange(tau_start, tau_stop, tau_step)
    freq = qubit_frequency - experiment.static.sampling_rate
    pi_half = pi_pulse / 2
    p3_start = experiment.static.readout_pulse_duration - pi_half
    p3 = BasicPulse(channel, p3_start, pi_half, 0.75 / 2, freq, 0, Rectangular())

    for tau in tau_sweep:
        tau_half = tau / 2
        p2_start = p3_start - tau_half - pi_pulse
        p2 = BasicPulse(channel, p2_start, pi_pulse, 0.75 / 2, freq, 0, Rectangular())
        p1_start = p2_start - tau_half - pi_pulse
        p1 = BasicPulse(channel, p1_start, pi_half, 0.75 / 2, freq, 0, Rectangular())

        ps = PulseSequence([p1, p2, p3], duration)
        seq.append(ps)

    return tau_sweep, seq

def T1(qubit_frequency, qubit_amplitude, pi_pulse, channel, tau_start=-2e-6, tau_stop=2e-5, tau_step=1e-7):
    """ T1 experiment to determine T1 time
    In this task, the qubit is driven with a pi-pulse, followed by a delay tau and then readout
    We add a few extra steps (defined by negative tau) where the qubit is not driven to determine the background level (ground state)
    We determine the T1 time by fitting the exponential decay of the qubit from the excited state to the ground state

    Args:
        qubit_frequency (float): Driving frequency of the qubit
        qubit_amplitude (float): Amplitude of the driving signal
        pi-pulse (float): Pi-pulse duration of the qubit
        channel (int): Qubit drive channel
        tau_start (float): Starting delay duration
        tau_end (float): Last delay duration
        tau_step (float): Delay step time
    
    Returns:
        List of PulseSequence
    
    """
    duration = max(30e-6, tau_stop + 7e-6)
    seq = []
    tau_sweep = np.arange(tau_start, tau_stop, tau_step)
    freq = qubit_frequency - experiment.static.sampling_rate

    for tau in tau_sweep:
        if tau >= 0:
            start = experiment.static.readout_pulse_duration - tau
            p = BasicPulse(channel, start, pi_pulse, 0.75 / 2, freq, 0, Rectangular())
            ps = PulseSequence([p], duration)
        else:
            ps = PulseSequence([], duration)
        seq.append(ps)

    return tau_sweep, seq
