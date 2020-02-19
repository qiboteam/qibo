#!/usr/bin/env python
# @authors: S. Carrazza and A. Garcia
from qibo.models import Circuit
from qibo.gates import H, CNOT, S, T, Iden, RX, RY, RZ, MX, MZ, MY, Flatten
from qibo.run import run
import numpy as np


def test():
    N = 2
    c = Circuit(N)
    state = np.array([1, 1, 1, 1])
    state = state/np.linalg.norm(state)
    c.add(Flatten(state))

    """
    c.add(H(0))
    c.add(H(1))
    c.add(H(2))
    c.add(CNOT(N - 1, 0))
    c.add(S(1))
    c.add(T(2))
    c.add(Iden(3))
    c.add(RX(1, 0.2))
    c.add(RY(3, 0.1))
    c.add(RZ(4, 0.6))

    c.add(MX(0))
    c.add(MZ(1))
    c.add(MY(2))
    """
    results = run(shots=1024)
    print(results)
    print(str(results['wave_func']))

    #assert results.get('virtual_machine') == True
    #assert results.get('wave_func') is not None
    #assert type(results.get('measure')) == dict


if __name__ == '__main__':
    test()
