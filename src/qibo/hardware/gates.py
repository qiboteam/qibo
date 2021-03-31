import copy
import math
from abc import ABC, abstractmethod
from qibo.abstractions import gates
from qibo.abstractions.abstract_gates import Gate
from qibo.config import raise_error


class HardwareGate(ABC, Gate):

    @abstractmethod
    def pulse_sequence(self, qubit_config, qubit_times):
        raise_error(NotImplementedError)

    @property
    def unitary(self):
        """Returns the unitary representation of the gate as a numpy array.

        Not required for hardware execution but required to construct the
        example circuits.
        """
        from qibo.core import cgates
        backend_gate = getattr(cgates, self.__class__.__name__)
        backend_gate = backend_gate(*self.init_args, **self.init_kwargs)
        return backend_gate.unitary


class I(HardwareGate, gates.I):

    def __init__(self, *q):
        gates.I.__init__(self, *q)

    def pulse_sequence(self, qubit_config, qubit_times):
        return []


class _Rn_(HardwareGate, gates._Rn_):

    def __init__(self, q, theta):
        gates._Rn_.__init__(self, q, theta)

    def pulse_sequence(self, qubit_config, qubit_times):
        if self.parameters == 0:
            return []

        q = self.target_qubits[0]
        time_mod = abs(self.parameters / math.pi)
        phase_mod = 0 if self.parameters > 0 else -180

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
        gates.RX.__init__(self, q, theta)


class RY(_Rn_, gates.RY):

    def __init__(self, q, theta):
        gates.RY.__init__(self, q, theta)
