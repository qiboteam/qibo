```python
import numpy as np

from qap import qubo_qap, qubo_qap_penalty, qubo_qap_feasibility, qubo_qap_energy, hamiltonian_qap
```

# QAP 

The quadratic assignment problem (QAP) is an important combinatorial optimization problems that was first introduced by Koopmans and Beckmann. The objective of the problem is to assign a set of facilities to a set of locations in such a way as to minimize the total assignment cost. The assignment cost for a pair of facilities is a function of the flow between the facilities and the distance between the locations of the facilities.


```python
def load_qap(filename):
    """Load qap problem from a file
    
    The file format is compatible with the one used in QAPLIB
    
    """

    with open(filename, 'r') as fh:
        n = int(fh.readline())

        numbers = [float(n) for n in fh.read().split()]

        data = np.asarray(numbers).reshape(2, n, n)
        f = data[1]
        d = data[0]
        
    i = range(len(f))
    f[i, i] = 0
    d[i, i] = 0

    return f, d
```

## Load qap problem from a file


```python
F, D = load_qap('tiny04a.dat')
print(f'The QAP instance is:')
print(F)
print(D)
```

    The QAP instance is:
    [[0.         0.29541331 0.68442855 0.19882279]
     [0.29541331 0.         0.61649225 0.16210679]
     [0.68442855 0.61649225 0.         0.73052088]
     [0.19882279 0.16210679 0.73052088 0.        ]]
    [[0.         0.77969778 0.43045022 0.43294055]
     [0.77969778 0.         0.1920096  0.58829618]
     [0.43045022 0.1920096  0.         0.47901122]
     [0.43294055 0.58829618 0.47901122 0.        ]]


## Calculate the penalty


```python
penalty = qubo_qap_penalty((F, D))
print(f'The penalty is {penalty}')
```

    The penalty is 2.2783420340595995


## Formulate the QUBO


```python
linear, quadratic, offset = qubo_qap((F, D), penalty=penalty)
print(f'linear: {linear}')
print()
print(f'quadratic: {quadratic}')
print()
print(f'offset: {offset}\n')
```

    linear: {0: -4.556684, 1: -4.556684, 2: -4.556684, 3: -4.556684, 4: -4.556684, 5: -4.556684, 6: -4.556684, 7: -4.556684, 8: -4.556684, 9: -4.556684, 10: -4.556684, 11: -4.556684, 12: -4.556684, 13: -4.556684, 14: -4.556684, 15: -4.556684}
    
    quadratic: {(1, 0): 2.278342, (2, 0): 2.278342, (3, 0): 2.278342, (4, 0): 2.278342, (5, 0): 0.2303331, (6, 0): 0.12716073, (7, 0): 0.1278964, (8, 0): 2.278342, (9, 0): 0.5336474, (10, 0): 0.2946124, (11, 0): 0.29631686, (12, 0): 2.278342, (13, 0): 0.15502168, (14, 0): 0.085583314, (15, 0): 0.08607845, (2, 1): 2.278342, (3, 1): 2.278342, (4, 1): 0.2303331, (5, 1): 2.278342, (6, 1): 0.05672219, (7, 1): 0.17379051, (8, 1): 0.5336474, (9, 1): 2.278342, (10, 1): 0.13141686, (11, 1): 0.4026467, (12, 1): 0.15502168, (13, 1): 2.278342, (14, 1): 0.038175885, (15, 1): 0.11696669, (3, 2): 2.278342, (4, 2): 0.12716073, (5, 2): 0.05672219, (6, 2): 2.278342, (7, 2): 0.14150628, (8, 2): 0.2946124, (9, 2): 0.13141686, (10, 2): 2.278342, (11, 2): 0.32784894, (12, 2): 0.085583314, (13, 2): 0.038175885, (14, 2): 2.278342, (15, 2): 0.09523835, (4, 3): 0.1278964, (5, 3): 0.17379051, (6, 3): 0.14150628, (7, 3): 2.278342, (8, 3): 0.29631686, (9, 3): 0.4026467, (10, 3): 0.32784894, (11, 3): 2.278342, (12, 3): 0.08607845, (13, 3): 0.11696669, (14, 3): 0.09523835, (15, 3): 2.278342, (5, 4): 2.278342, (6, 4): 2.278342, (7, 4): 2.278342, (8, 4): 2.278342, (9, 4): 0.48067763, (10, 4): 0.2653692, (11, 4): 0.2669045, (12, 4): 2.278342, (13, 4): 0.1263943, (14, 4): 0.069778904, (15, 4): 0.0701826, (6, 5): 2.278342, (7, 5): 2.278342, (8, 5): 0.48067763, (9, 5): 2.278342, (10, 5): 0.11837243, (11, 5): 0.36268005, (12, 5): 0.1263943, (13, 5): 2.278342, (14, 5): 0.03112606, (15, 5): 0.095366806, (7, 6): 2.278342, (8, 6): 0.2653692, (9, 6): 0.11837243, (10, 6): 2.278342, (11, 6): 0.2953067, (12, 6): 0.069778904, (13, 6): 0.03112606, (14, 6): 2.278342, (15, 6): 0.07765097, (8, 7): 0.2669045, (9, 7): 0.36268005, (10, 7): 0.2953067, (11, 7): 2.278342, (12, 7): 0.0701826, (13, 7): 0.095366806, (14, 7): 0.07765097, (15, 7): 2.278342, (9, 8): 2.278342, (10, 8): 2.278342, (11, 8): 2.278342, (12, 8): 2.278342, (13, 8): 0.5695855, (14, 8): 0.31445286, (15, 8): 0.3162721, (10, 9): 2.278342, (11, 9): 2.278342, (12, 9): 0.5695855, (13, 9): 2.278342, (14, 9): 0.14026703, (15, 9): 0.42976263, (11, 10): 2.278342, (12, 10): 0.31445286, (13, 10): 0.14026703, (14, 10): 2.278342, (15, 10): 0.3499277, (12, 11): 0.3162721, (13, 11): 0.42976263, (14, 11): 0.3499277, (15, 11): 2.278342, (13, 12): 2.278342, (14, 12): 2.278342, (15, 12): 2.278342, (14, 13): 2.278342, (15, 13): 2.278342, (15, 14): 2.278342}
    
    offset: 18.226736272476796
    


## Generate a random solution and check its feasibility


```python
rng = np.random.default_rng(seed=1234)
random_solution = {i: rng.integers(2) for i in range(F.size)}
print(f'The random solution is {random_solution}\n')
```

    The random solution is {0: 1, 1: 1, 2: 1, 3: 0, 4: 0, 5: 1, 6: 0, 7: 0, 8: 0, 9: 0, 10: 1, 11: 0, 12: 1, 13: 0, 14: 1, 15: 0}
    



```python
feasibility = qubo_qap_feasibility((F, D), random_solution)
print(f'The feasibility of the random solution is {feasibility}\n')
```

    The feasibility of the random solution is False
    


## Generate a feasible solution and check its feasibility


```python
feasible_solution = np.zeros(F.shape)
sequence = np.arange(F.shape[0])
np.random.shuffle(sequence)
for i in range(F.shape[0]):
    feasible_solution[i, sequence[i]] = 1
feasible_solution = {k:v for k, v in enumerate(feasible_solution.flatten())}
print(f'The feasible solution is {feasible_solution}\n')
```

    The feasible solution is {0: 0.0, 1: 0.0, 2: 1.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 1.0, 8: 0.0, 9: 1.0, 10: 0.0, 11: 0.0, 12: 1.0, 13: 0.0, 14: 0.0, 15: 0.0}
    



```python
feasibility = qubo_qap_feasibility((F, D), feasible_solution)
print(f'The feasibility of the feasible solution is {feasibility}\n')
```

    The feasibility of the feasible solution is True
    


## Calculate the energy of the feasible solution


```python
energy = qubo_qap_energy((F,D), feasible_solution)
print(f'The energy of the feasible solution is {energy}')
```

    The energy of the feasible solution is 2.7219091992575177


# Hamiltonian


```python
ham = hamiltonian_qap((F, D), dense=False)
```

    [Qibo 0.1.6|INFO|2022-05-31 14:47:26]: Using qibojit backend on /GPU:0

