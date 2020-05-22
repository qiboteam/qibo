"""Use ops in python."""
from tensorflow.python.framework import load_library
from tensorflow.python.platform import resource_loader

custom_module = load_library.load_op_library(
    resource_loader.get_path_to_datafile('_qibo_tf_custom_operators.so'))

# initial_state operator
initial_state = custom_module.initial_state

# apply_gate operator
def apply_gate(state, gate, nqubits, target, control=[]):
    if not (isinstance(control, list) or isinstance(control, tuple)):
        raise TypeError("Control qubits must be a list or tuple but {} "
                        "was given.".format(type(control)))

    return custom_module.apply_gate(state, gate, nqubits, target, control)
