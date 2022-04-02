'''
this file is to collect various functions used in error mitigation
'''

from qibo import models
import numpy as np

def circuit_folding(circ, scaling_factor):
    ''' folds a quantum circuit
    '''
    n = circ.nqubits
    circ_inv = circ.invert()
    c = models.Circuit(n)
    repeat_count = int((scaling_factor-1)/2)
    for i in range(repeat_count):
        c += (circ + circ_inv)
    c += circ
    return c


def depolarizing(rho,f):
    '''
    Args:
        rho: a function in L(C^D), set of linear map, represented as a matrix.
        f: f is a parameter, smaller f means less noise, it is between 0 and 1
    Returns:
        D: rho maps to f*rho + (1-f)*trace(rho)*I/d
    '''

    d = rho.shape[0]
    return f*rho + (1-f)*np.trace(rho)*np.eye(d)/d


