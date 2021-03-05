"""Use ops in python."""
import tensorflow as tf
from tensorflow.python.framework import load_library # pylint: disable=no-name-in-module
from tensorflow.python.platform import resource_loader # pylint: disable=no-name-in-module
from qibo.config import get_threads


if tf.config.list_physical_devices("GPU"): # pragma: no cover
    # case not covered by GitHub workflows because it requires GPU
    library_path = '_qibo_tf_custom_operators_cuda.so'
else:
    library_path = '_qibo_tf_custom_operators.so'

custom_module = load_library.load_op_library(
    resource_loader.get_path_to_datafile(library_path))

# initial_state operator
initial_state = custom_module.initial_state

# transpose_state operator (for multi-GPU)
transpose_state = custom_module.transpose_state

# swap state pieces operator (for multi-GPU)
swap_pieces = custom_module.swap_pieces

# measurement frequencies operator
measure_frequencies = custom_module.measure_frequencies

# apply_gate operator
def apply_gate(state, gate, qubits, nqubits, target, omp_num_threads=get_threads()):
    """Applies arbitrary one-qubit gate to a state vector.

    Modifies ``state`` in-place.
    Gates can be controlled to multiple qubits.

    Args:
        state (tf.Tensor): State vector of shape ``(2 ** nqubits,)``.
        gate (tf.Tensor): Gate matrix of shape ``(2, 2)``.
        qubits (tf.Tensor): Tensor that contains control and target qubits in
            sorted order. See :meth:`qibo.tensorflow.cgates.TensorflowGate.qubits_tensor`
            for more details.
        nqubits (int): Total number of qubits in the state vector.
        target (int): Qubit ID that the gate will act on.
            Must be smaller than ``nqubits``.

    Return:
        state (tf.Tensor): State vector of shape ``(2 ** nqubits,)`` after
            ``gate`` is applied.
    """
    return custom_module.apply_gate(state, gate, qubits, nqubits, target, omp_num_threads)


apply_two_qubit_gate = custom_module.apply_two_qubit_gate

# gate specific operators
apply_x = custom_module.apply_x

apply_y = custom_module.apply_y

apply_z = custom_module.apply_z

apply_z_pow = custom_module.apply_z_pow

apply_fsim = custom_module.apply_fsim

apply_swap = custom_module.apply_swap

def collapse_state(state, qubits, result, nqubits, normalize=True, omp_num_threads=get_threads()):
    return custom_module.collapse_state(state, qubits, result, nqubits, normalize, omp_num_threads)
