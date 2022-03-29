'''
this file is to collect various functions used in error mitigation
'''

from qibo import models

def circuit_folding(circ, scaling_factor):
    '''
    performs circuit folding and outputs a circuit that is k=scaling_factor
    times as long as the original circuit
    '''
    n = circ.nqubits
    repeat_count = int((scaling_factor - 1) / 2)
    circ_inv = circ.invert()
    c = models.Circuit(n)
    for i in range(repeat_count):
        c += (circ + circ_inv)
    c += circ
    return c