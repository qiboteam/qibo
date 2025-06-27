from qibo.backends.numpy import NumpyBackend, NumpyMatrices
from perceval import Experiment, Processor, BasicState, probs_to_samples, BS
from perceval import __version__ as pcvl_version
from qibo import __version__, Circuit
from qibo.measurements import MeasurementResult
from .modality import Modality
import numpy as np

from qibo.gates import H, PhotonicGate


class LOQCMatrices:
    def __init__(self, dtype):
        self.dtype = dtype

    def _cast(self, x, dtype):
        if isinstance(x, list):
            return np.array(x, dtype=dtype)
        return x.astype(dtype)
    
    def BS(self, theta):
        cos = np.cos(theta / 2.0) + 0j
        isin = -1j * np.sin(theta / 2.0)
        return self._cast([[cos, isin], [isin, cos]], dtype=self.dtype)
    
    def PS(self, phi):
        return self._cast([self.np.exp(1j * phi)], dtype=self.dtype)

    def H(self):
        npmat = NumpyMatrices(self.dtype)
        return npmat.H


class SlosBackend(NumpyBackend):

    def __init__(self):
        super().__init__()
        self.name = "slos"
        self.matrices = LOQCMatrices(self.dtype)
        self.tensor_types = np.ndarray
        self.versions = {"qibo": __version__, "perceval": pcvl_version}
        self.modality = Modality.PHOTONIC_CV

    def _convert_gate(self, gate):
        if isinstance(gate, PhotonicGate):
            return gate._pcvl
        if isinstance(gate, H):
            return BS.H()

    def _convert_circuit(self, circuit: Circuit):
        exp = Experiment(circuit.nqubits)
        for gate in circuit.queue:
            component = self._convert_gate(gate)
            exp.add(gate.wires, component)
        return exp

    def execute_circuit(self, circuit: Circuit, initial_state=None, nshots=None):
        experiment = self._convert_circuit(circuit)
        if initial_state is not None:
            experiment.with_input(BasicState(initial_state))
        p = Processor("SLOS", experiment)
        samples = probs_to_samples(p.probs()["results"], count=nshots)
        
        m = MeasurementResult(experiment.m)
        m.register_samples([list(k) for k in samples])
        return m


