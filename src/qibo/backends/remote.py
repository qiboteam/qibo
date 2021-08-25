from qibo.backends.numpy import NumpyBackend
from qibo.config import raise_error


class QilimanjaroBackend(NumpyBackend):  # pragma: no cover
    # remote backend is not tested until `qilimanjaroq` is available

    description = "Uses remote devices controlled by Qilimanjaro."

    def __init__(self):
        super().__init__()
        self.name = "qilimanjaroq"
        import qilimanjaroq  # pylint: disable=E0401
        self.remote_module = qilimanjaroq
        from qibo import gates
        self.remote_gates = gates
        self.remote_circuit = qilimanjaroq.circuit.RemoteCircuit
