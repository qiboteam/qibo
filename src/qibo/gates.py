from qibo.config import BACKEND_NAME
if BACKEND_NAME == "tensorflow":
    from qibo.tensorflow.cgates import *
else: # pragma: no cover
    raise NotImplementedError("Only Tensorflow backend is implemented.")
