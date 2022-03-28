'''
this file is to collect various functions used in error mitigation
'''

from qibo import models

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