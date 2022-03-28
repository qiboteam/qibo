'''
this file is to collect various functions used in error mitigation
'''

from qibo import models

def circuit_folding(circ, repeat_count):
    ''' folds a quantum circuit
    '''
    n = circ.nqubits
    circ_inv = circ.invert()
    c = models.Circuit(n)
    for i in range(repeat_count):
        c += (circ + circ_inv)
    c += circ
    return c