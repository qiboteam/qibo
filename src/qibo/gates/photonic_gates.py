from abc import abstractmethod
import numpy as np

from .abstract import ParametrizedGate
from .gates import H


class PhotonicGate:
    def __init__(self, wires: tuple[int, ...]):
        import perceval as pcvl
        self._wires = wires
        self._pcvl = pcvl

    @property
    def wires(self):
        return self._wires

    @property
    @abstractmethod
    def photonic_component(self):
        """return a Perceval photonic component from the gate internal data"""


class PS(ParametrizedGate, PhotonicGate):
    def __init__(self, wire: int, phi, trainable: bool = True):
        ParametrizedGate.__init__(self, trainable=trainable)
        self.nparams = 1
        self.parameter_names = [phi]
        self.parameters = phi,

        PhotonicGate.__init__(self, (wire,))
        self.target_qubits = (wire,)
        self.init_args = [wire, phi]
        self.name = "Phase shifter"
        self.draw_label = "PS"
        self.init_kwargs = {"phi": phi, "trainable": trainable}

    @property
    def photonic_component(self):
        phi = self.parameters[0]
        if isinstance(phi, str):
            return self._pcvl.PS(self._pcvl.Parameter(phi))
        return self._pcvl.PS(phi)


class BS(ParametrizedGate, PhotonicGate):
    def __init__(self, q0, q1, theta=np.pi / 2, trainable: bool = True):
        ParametrizedGate.__init__(self, trainable=True)
        self.nparams = 1
        self.parameter_names = [theta]
        self.parameters = theta,

        PhotonicGate.__init__(self, (q0, q1))
        self.name = "Beam splitter"
        self.draw_label = "X"
        self.target_qubits = (q0, q1)
        self.init_args = [q0, q1, theta]
        self.init_kwargs = {"theta": theta, "trainable": trainable}

    @property
    def wires(self):
        return self.target_qubits

    @property
    def photonic_component(self):
        theta = self.parameters[0]
        if isinstance(theta, str):
            return self._pcvl.BS(self._pcvl.Parameter(theta))
        return self._pcvl.BS(theta)


PHOTONIC_GATE_TYPES = (PhotonicGate, H)
