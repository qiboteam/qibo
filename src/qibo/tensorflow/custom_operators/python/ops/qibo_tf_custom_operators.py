"""Use ops in python."""
from tensorflow.python.framework import load_library
from tensorflow.python.platform import resource_loader

custom_module = load_library.load_op_library(
    resource_loader.get_path_to_datafile('_qibo_tf_custom_operators.so'))

# initial_state operator
initial_state = custom_module.initial_state

# apply_gate operator
apply_gate = custom_module.apply_gate
