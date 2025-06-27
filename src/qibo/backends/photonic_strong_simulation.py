from .abstract import Backend
from qibo.backends.numpy import NumpyBackend
from perceval import Experiment, Processor, BasicState, probs_to_samples
from perceval import __version__ as pcvl_version
import numpy as np
from qibo import __version__
from qibo.measurements import MeasurementResult
from .modality import Modality


class LOQCMatrices:
    def __init__(self, dtype):
        import numpy as np

        self.dtype = dtype
        self.np = np

    def _cast(self, x, dtype):
        if isinstance(x, list):
            return self.np.array(x, dtype=dtype)
        return x.astype(dtype)
    
    def BS(self, theta):
        cos = self.np.cos(theta / 2.0) + 0j
        isin = -1j * self.np.sin(theta / 2.0)
        return self._cast([[cos, isin], [isin, cos]], dtype=self.dtype)
    
    def PS(self, phi):
        return self._cast([self.np.exp(1j * phi)], dtype=self.dtype)


class SlosBackend(NumpyBackend):

    def __init__(self):
        super().__init__()
        self.name = "slos"
        self.matrices = LOQCMatrices(self.dtype)
        self.tensor_types = np.ndarray
        self.versions = {"qibo": __version__, "perceval": pcvl_version}
        self.modality = Modality.PHOTONIC_CV

    def cast(self, x, dtype=None, copy=False):
        if dtype is None:
            dtype = self.dtype
        if isinstance(x, self.tensor_types):
            return x.astype(dtype, copy=copy)
        elif self.is_sparse(x):
            return x.astype(dtype, copy=copy)
        return np.asarray(x, dtype=dtype, copy=copy if copy else None)

    def execute_circuit(self, circuit: Experiment, initial_state=None, nshots=None):
        experiment = circuit
        if initial_state is not None:
            experiment.with_input(BasicState(initial_state))
        p = Processor("SLOS", experiment)
        samples = probs_to_samples(p.probs()["results"], count=nshots)
        
        M=MeasurementResult(experiment.m)
        M.register_samples([list(k) for k in samples])

        return M
    
    def matrix(self, gate):
        """Convert a gate to its matrix representation in the computational basis."""
        name = gate.__class__.__name__
        _matrix = getattr(self.matrices, name)
        if callable(_matrix):
            _matrix = _matrix(len(gate.target_qubits))
        return self.cast(_matrix, dtype=_matrix.dtype)

    def matrix_parametrized(self, gate):
        """Convert a parametrized gate to its matrix representation in the computational basis."""
        name = gate.__class__.__name__
        _matrix = getattr(self.matrices, name)
        _matrix = _matrix(*gate.parameters)
        return self.cast(_matrix, dtype=_matrix.dtype)


