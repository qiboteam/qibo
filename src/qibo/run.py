# -*- coding: utf-8 -*-
# @authors: S. Carrazza and A. Garcia


def run(shots=1024, backend=None):
    """
    Prepares model and executes for a given backend.
    Args:
        backend (qibo.backends): one of the available backends (see :ref:`Backends`).
        shots (int): number of measurement shots.
    Returns:
        dict: the result object
    """
    if backend is None:
        from qibo.backends import tensorflow as backend
    return backend.execute(shots)
