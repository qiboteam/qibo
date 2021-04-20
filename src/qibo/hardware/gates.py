import copy
import math
from abc import ABC, abstractmethod
from qibo.abstractions import gates
from qibo.abstractions.abstract_gates import Gate
from qibo.config import raise_error


class HardwareGate(ABC, Gate):

    @abstractmethod
    def pulse_sequence(self, qubit_config, qubit_times, qubit_phases):
        raise_error(NotImplementedError)

    @property
    def unitary(self):
        """Returns the unitary representation of the gate as a numpy array.

        Not required for hardware execution but required to construct the
        example circuits.
        """
        from qibo import gates
        backend_gate = getattr(gates, self.__class__.__name__)
        backend_gate = backend_gate(*self.init_args, **self.init_kwargs)
        return backend_gate.unitary


class I(HardwareGate, gates.I):

    def __init__(self, *q):
        gates.I.__init__(self, *q)

    def pulse_sequence(self, qubit_config, qubit_times, qubit_phases):
        return []


class _Rn_(HardwareGate, gates._Rn_):

    def __init__(self, q, theta):
        gates._Rn_.__init__(self, q, theta)

    def pulse_sequence(self, qubit_config, qubit_times, qubit_phases):
        if self.parameters == 0:
            return []

        q = self.target_qubits[0]
        time_mod = abs(self.parameters / math.pi)
        phase_mod = 0 if self.parameters > 0 else -180
        phase_mod += qubit_phases[q]

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

class M(HardwareGate, gates.M):
    def __init__(self, *q):
        gates.M.__init__(self, *q)

    def pulse_sequence(self, qubit_config, qubit_times):
        pulses = []
        for q in self.target_qubits:
            pulses += qubit_config[q]["gates"][self.name]
        
        return pulses
class H(HardwareGate, gates.H):
    def __init__(self, q):
        gates.H.__init__(self, q)

    def pulse_sequence(self, qubit_config, qubit_times, qubit_phases):
        q = self.target_qubits[0]
        pulses = []
        composite = [RY(q, 90), RX(q, 180)]
        for g in composite:
            pulses += g.pulse_sequence(qubit_config, qubit_times, qubit_phases)

        return pulses

class CNOT(HardwareGate, gates.CNOT):
    def __init__(self, q0, q1):
        gates.CNOT.__init__(self, q0, q1)
    
    def pulse_sequence(self, qubit_config, qubit_times, qubit_phases):
        q = self.target_qubits[0]
        control = self.control_qubits[0]
        pulses = copy.deepcopy(qubit_config[q]["gates"][self.name + "_{}".format(control)])

        for p in pulses:
            duration = p.duration
            p.start = qubit_times[q]
            p.phase = qubit_phases[q]
            p.duration = duration
            qubit_times[q] += duration

        return pulses
