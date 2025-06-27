from .abstract import ParametrizedGate
from .gates import H
import perceval as pcvl
import numpy as np


class PhotonicGate:
    def __init__(self, wires: tuple[int, ...]):
        self._pcvl: pcvl.AComponent
        self._wires = wires

    @property
    def wires(self):
        return self._wires

    @property
    def photonic_component(self):
        return self._pcvl


class PS(ParametrizedGate, PhotonicGate):
    def __init__(self, wire: int, phi, trainable: bool = True):
        ParametrizedGate.__init__(self, trainable=trainable)
        self.nparams = 1
        self.parameter_names = [phi]
        self.parameters = phi,
        if isinstance(phi, str):
            self._pcvl = pcvl.PS(pcvl.Parameter(phi))
        else:
            self._pcvl = pcvl.PS(phi)

        PhotonicGate.__init__(self, (wire,))
        self.target_qubits = (wire,)
        self.init_args = [wire, phi]
        self.name = "Phase shifter"
        self.draw_label = "PS"
        self.init_kwargs = {"phi": phi, "trainable": trainable}


class BS(ParametrizedGate, PhotonicGate):
    def __init__(self, q0, q1, theta=np.pi / 2, trainable: bool = True):
        ParametrizedGate.__init__(self, trainable=True)
        self.nparams = 1
        self.parameter_names = [theta]
        self.parameters = theta,
        if isinstance(theta, str):
            self._pcvl = pcvl.BS(pcvl.Parameter(theta))
        else:
            self._pcvl = pcvl.BS(theta)
        PhotonicGate.__init__(self, (q0, q1))
        self.name = "Beam splitter"
        self.draw_label = "X"
        self.target_qubits = (q0, q1)
        self.init_args = [q0, q1, theta]

    @property
    def wires(self):
        return self.target_qubits


PHOTONIC_GATE_TYPES = (PhotonicGate, H)
