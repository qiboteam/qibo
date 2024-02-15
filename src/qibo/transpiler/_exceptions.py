"""Custom exceptions raised in transpiler routines."""


class BlockingError(Exception):
    """Raise when an error occurs in the blocking procedure"""


class ConnectivityError(Exception):
    """Raise for an error in the connectivity"""


class DecompositionError(Exception):
    """A decomposition error is raised when, during transpiling,
    gates are not correctly decomposed in native gates"""


class PlacementError(Exception):
    """Raise for an error in the initial qubit placement"""


class TranspilerPipelineError(Exception):
    """Raise when an error occurs in the transpiler pipeline"""
