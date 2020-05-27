from qibo.config import BACKEND_NAME
if BACKEND_NAME == "tensorflow":
    import tensorflow as tf
    if tf.config.list_physical_devices("GPU"):
        # If GPU is available use Tensorflow gates
        from qibo.tensorflow.gates import *
    else:
        # For CPU use custom operator gates
        from qibo.tensorflow.cgates import *

else:
    raise NotImplementedError("Only Tensorflow backend is implemented.")
