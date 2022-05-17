from qibo.engines.numpy import NumpyEngine
from qibo.engines.tensorflow import TensorflowEngine


engine = NumpyEngine()

# TODO: Implement engine setter, similar to ``qibo.set_backend()``
# This global engine will be used as default by ``circuit.execute()