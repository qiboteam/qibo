"""Use ops in python."""
from tensorflow.python.framework import load_library
from tensorflow.python.platform import resource_loader

custom_module = load_library.load_op_library(
    resource_loader.get_path_to_datafile('_qibo_tf_custom_operators.so'))

# initial_state operator
initial_state = custom_module.initial_state

def sort_controls(controls):
    """Checks if ``controls`` variable has valid type."""
    if not (isinstance(controls, list) or isinstance(controls, tuple)):
        raise TypeError("Control qubits must be a list or tuple but {} "
                        "was given.".format(type(controls)))
    return sorted(controls)[::-1]

# apply_gate operator
def apply_gate(state, gate, nqubits, target, controls=[]):
    """Applies one-qubit gates to a state vector.

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
    return custom_module.apply_gate(state, gate, controls, nqubits, target)

# gate specific operators
def apply_x(state, nqubits, target, controls=[]):
    controls = sort_controls(controls)
    return custom_module.apply_x(state, controls, nqubits, target)

def apply_y(state, nqubits, target, controls=[]):
    controls = sort_controls(controls)
    return custom_module.apply_y(state, controls, nqubits, target)

def apply_z(state, nqubits, target, controls=[]):
    controls = sort_controls(controls)
    return custom_module.apply_z(state, controls, nqubits, target)

def apply_zpow(state, theta, nqubits, target, controls=[]):
    controls = sort_controls(controls)
    return custom_module.apply_z_pow(state, theta, controls, nqubits, target)

def apply_swap(state, nqubits, target1, target2, controls=[]):
    controls = sort_controls(controls)
    return custom_module.apply_swap(state, controls, nqubits, target1, target2)
