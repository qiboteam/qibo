# Quantum Networks

## The Quantum Network Model

The quantum network model is a mathematical framework that allows us to uniquely describe quantum information processing that involves multiple points in time and space.
Each distinguished point in time and space is treated as a linear system $\mathcal{H}_i$.
A quantum network involving $n$ points in time and space is a Hermitian operator $\mathcal{N}$ that acts on the tensor product of the linear systems $\mathcal{H}_0 \otimes \mathcal{H}_1 \otimes \cdots \otimes \mathcal{H}_{n-1}$.
Each system $\mathcal{H}_i$ is either an input or an output of the network.

A physically implementable quantum network is described by a semi-positive definite operator $\mathcal{N}$ that satisfies the causal constraints.

A simple example is a quantum channel $\Gamma: \mathcal{H}_0 \to \mathcal{H}_1$, where $\mathcal{H}_0$ is the input system and $\mathcal{H}_1$ is the output system.
The quantum channel is a linear map, such that it maps any input quantum state to an output quantum state, which is a sufficient and necessary condition for the map to be physical.
A Hermitian operator $J^\Gamma$ acting on $\mathcal{H}_0\otimes \mathcal{H}_1$ is associated with a quantum channel $\Gamma$, if $J^\Gamma$ satisfies the following conditions:
$$
J^\Gamma \geq 0, \quad \text{and} \quad \text{Tr}_{\mathcal{H}_1} J^\Gamma = \mathbb{I}_{\mathcal{H}_0}.
$$
The first condition is called *complete positivity*, and the second condition is called *trace-preserving*.
In particular, the second condition ensures that the information of the input system is only accessible through the output system.

In particular, a quantum state $\rho$ may be also considered as a quantum network, where the input system is the trivial system $\mathbb{C}$, and the output system is the quantum system $\mathcal{H}$.
The constraints on the quantum channels are then equivalent to the constraints on the quantum states:
$$
\rho \geq 0, \quad \text{and} \quad \text{Tr} \rho = \mathbb{I}_\mathbb{C} = 1.
$$

> For more details, see G. Chiribella *et al.*, *Theoretical framework for quantum networks*,
> [Physical Review A 80.2 (2009): 022339](https://journals.aps.org/pra/abstract/10.1103/PhysRevA.80.022339).

## Quantum Network in `qibo`

The manipulation of quantum networks in `qibo` is done through the `QuantumNetwork` class.

```python
from qibo.quantum_info.quantum_networks import QuantumNetwork
```

A quantum state is a quantum network with a single input system and a single output system, where the input system is the trivial 1-dimensional system.
We need to specify the dimensions of each system in the `partition` argument.

```python
from qibo.quantum_info import random_density_matrix, random_unitary

state = random_density_matrix(2)
state_choi = QuantumNetwork(state, (1,2))
print(f'A quantum state is a quantum netowrk of the form {state_choi}')
```

```
>>> A quantum state is a quantum netowrk of the form J[1 -> 2]
```

A general quantum channel can be created in a similar way.

```python
from qibo.gates import DepolarizingChannel

test_ch = DepolarizingChannel(0,0.5)
N = len(test_ch.target_qubits)
partition = (2**N, 2**N)

depolar_choi = QuantumNetwork(test_ch.to_choi(), partition)
print(f'A quantum channel is a quantum netowrk of the form {depolar_choi}')
```

```
>>> A quantum channel is a quantum netowrk of the form J[2 -> 2]
```

One may apply a quantum channel to a quantum state, or compose two quantum channels, using the `@` operator.

```python
new_state = depolar_choi @ depolar_choi @ state_choi
```

## Example

For 3-dimensional systems, an unital channel may not be a mixed unitary channel.

> Example 4.3 in (Watrous, John. The theory of quantum information. Cambridge university press, 2018.)

```python
A1 = np.array([
    [0,0,0],
    [0,0,1/np.sqrt(2)],
    [0,-1/np.sqrt(2),0],
])
A2 = np.array([
    [0,0,1/np.sqrt(2)],
    [0,0,0],
    [-1/np.sqrt(2),0,0],
])
A3 = np.array([
    [0,1/np.sqrt(2),0],
    [-1/np.sqrt(2),0,0],
    [0,0,0],
])

Choi1 = QuantumNetwork(A1, (3,3), pure=True) * 3
Choi2 = QuantumNetwork(A2, (3,3), pure=True)*3
Choi3 = QuantumNetwork(A3, (3,3), pure=True)*3
```

The three channels are pure but not unital. Which means they are not unitary.

```python
print(f"Choi1 is unital: {Choi1.unital()}")
print(f"Choi2 is unital: {Choi2.unital()}")
print(f"Choi3 is unital: {Choi3.unital()}")
```

```
>>> Choi1 is unital: False
Choi2 is unital: False
Choi3 is unital: False
```

However, the mixture of the three operators are unital.
As the matrices are orthogonal, they are the extreme points of the convex set of the unital channels.
Therefore, this mixed channel is not a mixed unitary channel.

```python
Choi = Choi1/3 + Choi2/3 + Choi3/3
print(f"The mixed channel is unital: {Choi.unital()}")
```

