from qibo.config import BACKEND_NAME, raise_error
if BACKEND_NAME == "tensorflow":
    from qibo.tensorflow.callbacks import *
else: # pragma: no cover
    raise_error(NotImplementedError, "Only Tensorflow backend is implemented.")
