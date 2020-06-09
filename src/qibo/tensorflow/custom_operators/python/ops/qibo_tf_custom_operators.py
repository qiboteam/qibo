"""Use ops in python."""
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import load_library
from tensorflow.python.platform import resource_loader

custom_module = load_library.load_op_library(
    resource_loader.get_path_to_datafile('_qibo_tf_custom_operators.so'))

# initial_state operator
initial_state = custom_module.initial_state

# transpose_state operator (for multi-GPU)
transpose_state = custom_module.transpose_state
def transpose_state(state, nqubits, qubit_order, new_state=None):
    if new_state is None:
        new_state = tf.zeros_like(state)
    new_state = custom_module.transpose_state(state, new_state, nqubits, qubit_order)
    del(state)
    return new_state


def sort_controls(controls):
    """Checks if ``controls`` variable has valid type."""
    if not (isinstance(controls, list) or isinstance(controls, tuple)):
        raise TypeError("Control qubits must be a list or tuple but {} "
                        "was given.".format(type(controls)))
    return sorted(controls)[::-1]

# apply_gate operator
def apply_gate(state, gate, nqubits, target, controls=[]):
    """Applies arbitrary one-qubit gate to a state vector.

    Modifies ``state`` in-place.
    Gates can be controlled to multiple qubits.

    Args:
        state (tf.Tensor): State vector of shape ``(2 ** nqubits,)``.
        gate (tf.Tensor): Gate matrix of shape ``(2, 2)``.
        nqubits (int): Total number of qubits in the state vector.
        target (int): Qubit ID that the gate will act on.
            Must be smaller than ``nqubits``.
        controls (list): List with qubit IDs that the gate will be controlled on.
            All qubit IDs must be smaller than ``nqubits``.

    Return:
        state (tf.Tensor): State vector of shape ``(2 ** nqubits,)`` after
            ``gate`` is applied.
    """
    controls = sort_controls(controls)
    return custom_module.apply_gate(state, gate, controls, controls, nqubits, target)

def apply_twoqubit_gate(state, gate, nqubits, targets, controls=[]):
    """Applies arbitrary two-qubit gate to a state vector."""
    controls = sort_controls(controls)
    t1, t2 = targets
    return custom_module.apply_two_qubit_gate(state, gate, controls, controls, nqubits,
                                              t1, t2)

# gate specific operators
def apply_x(state, nqubits, target, controls=[]):
    controls = sort_controls(controls)
    return custom_module.apply_x(state, controls, controls, nqubits, target)

def apply_y(state, nqubits, target, controls=[]):
    controls = sort_controls(controls)
    return custom_module.apply_y(state, controls, controls, nqubits, target)

def apply_z(state, nqubits, target, controls=[]):
    controls = sort_controls(controls)
    return custom_module.apply_z(state, controls, controls, nqubits, target)

def apply_zpow(state, theta, nqubits, target, controls=[]):
    controls = sort_controls(controls)
    return custom_module.apply_z_pow(state, theta, controls, controls, nqubits, target)

def apply_fsim(state, gate, nqubits, targets, controls=[]):
    controls = sort_controls(controls)
    t1, t2 = targets
    return custom_module.apply_fsim(state, gate, controls, controls, nqubits, t1, t2)

def apply_swap(state, nqubits, targets, controls=[]):
    controls = sort_controls(controls)
    t1, t2 = targets
    return custom_module.apply_swap(state, controls, controls, nqubits, t1, t2)
