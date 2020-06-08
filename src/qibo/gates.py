from qibo.config import BACKEND_NAME
if BACKEND_NAME == "tensorflow":
    import tensorflow as tf
    from qibo.tensorflow.cgates import *
else:
    raise NotImplementedError("Only Tensorflow backend is implemented.")
