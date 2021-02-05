import copy
import math
from abc import ABC, abstractmethd
from qibo.config import raise_error


class HardwareGate(ABC):

    def __init__(self, *q):
        self.target_qubits = tuple(q)

    @abstractmethod
    def pulse_sequence(self, qubit_config, qubit_times):
        raise_error(NotImplementedError)


class I(HardwareGate):

    def pulse_sequence(self, qubit_config, qubit_times):
        return []


class _Rn_(HardwareGate):

    def pulse_sequence(self, qubit_config, qubit_times):
        if self.theta == 0:
            return []

        q = self.target_qubits[0]
        time_mod = abs(self.theta / math.pi)
        phase_mod = 0 if angle > 0 else -180

        pulses = copy.deepcopy(qubit_config[q]["gates"][self.name])
        for p in pulses:
            duration = p.duration * time_mod
            p.start = qubit_times[q]
            p.phase += phase_mod
            p.duration = duration
            qubit_times[q] += duration

        return pulses


class RX(_Rn_, gates.RX):

    def __init__(self, q, theta):
        self.target_qubits = (q,)
        self.theta = theta


class RY(_Rn_, gates.RY):

    def __init__(self, q, theta):
        self.target_qubits = (q,)
        self.theta = theta
