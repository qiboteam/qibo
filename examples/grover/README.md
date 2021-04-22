# A General Grover Model

The examples presented here provide information to run a general Grover model accesible as
`from qibo.models import Grover`. This model allows to construct a general
circuit to make use of generalized Grover models to search for states on an
unstructured database. References can be checked at 

- For Grover's original search algorithm: [arXiv:quant-ph/9605043](https://arxiv.org/abs/quant-ph/9605043)
- For the iterative version with unknown solutions:[arXiv:quant-ph/9605034](https://arxiv.org/abs/quant-ph/9605034)
- For the Grover algorithm with any superposition:[arXiv:quant-ph/9712011](https://arxiv.org/abs/quant-ph/9712011)

The arguments of the model are

- oracle (`qibo.core.circuit.Circuit`): quantum circuit that flips
    the sign using a Grover ancilla initialized with -X-H-. Grover ancilla
    expected to be the last qubit of oracle circuit.
- superposition_circuit (`qibo.core.circuit.Circuit`): quantum circuit that
    takes an initial state to a superposition. Expected to use the first
    set of qubits to store the relevant superposition.
- initial_state_circuit (`qibo.core.circuit.Circuit`): quantum circuit
    that initializes the state. If empty defaults to |000..00>
- superposition_qubits (int): number of qubits that store the relevant superposition.
    Leave empty if superposition does not use ancillas.
- superposition_size (int): how many states are in a superposition.
    Leave empty if its an equal superposition of quantum states.
- number_solutions (int): number of expected solutions. Needed for normal Grover.
    Leave empty for iterative version.
- target_amplitude (float): absolute value of the amplitude of the target state. Only for
    advanced use and known systems.
- check (function): function that returns True if the solution has been
    found. Required of iterative approach.
    First argument should be the bitstring to check.
- check_args (tuple): arguments needed for the check function.
    The found bitstring not included.
- iterative (bool): force the use of the iterative Grover

We provide three different examples to run Grover with:

### Example 1: standard Hadamard superposition and simple oracle. 

In this first example we show how to use a standard Grover search where the
search space is an equally weighted superposition of quantum states and the oracle
is simply defined through a layer of Hadamard gates. This example makes no use of 
ancilla qubits, so the command lines simplify greatly. 
1. First we create a superposition circuit. It is done by just initializing a 5-qubit circuit,
and adding Hadamard gates
```text
superposition = Circuit(5) 
superposition.add([gates.H(i) for i in range(5)])
```

2. The next step is creating the oracle. In this case, we look for the states where all the 
qubits are in the `1` state
```text
oracle = Circuit(5 + 1)
oracle.add(gates.X(5).controlled_by(*range(5)))
```

3. Now we create the Grover model.
```text
grover = Grover(oracle, superposition_circuit=superposition, number_solutions=1)
solution, iterations = grover()
```
   In this case there are no ancilla qubits, and it is straightforward to see that there is only
one possible solution, so we do not have to use any further information as the input for the circuit.
   
### Example 2: standard Hadamard superposition and oracle with ancillas

This second example is more complicated since we create an oracle function that makes use of ancilla qubits.
We want to create a Grover model such that it searches all the basis states with `num_1` qubits in the
`1` state. The search space is all the possibilities with `qubits` qubits. Those parameters
are to be defined in the script. Functions `one_sum, sum_circuit, oracle` create the corresponding circuits for the oracle.
The superposition circuit is the standard Hadamard one and is therefore not specified. However, note that since 
there are ancilla qubits in the oracle, we need to give the information of the size of the search space through the
argument `superposition_qubits`. We also provide a `check` function counting the number of `1` in a bit string to be used 
in the iterative approach.

In the non-iterative standard case we must write
```text
grover = Grover(oracle, superposition_qubits=qubits, number_solutions=int(binom(qubits, num_1)))
solution, iterations = grover()
```

For the iterative case, the corresponding code is 
```text
grover = Grover(oracle, superposition_qubits=qubits, check=check, check_args=(num_1,))
solution, iterations = grover()
```


### Example 3: Ancillas for superposition and oracle, setting size of search space

In this third example we create a Grover model with two components:
- A superposition circuit creating all the elements in the computational basis with `num_1` qubits in the `1` state.
- An oracle checking that the first `num_1` qubits are in the `1` state and does not care about any other qubit. This
oracle is written in such a way that it needs ancilla qubits. This is not necessary in QIBO, it is done in this way for
  illustrating the model.

The joint action of both elements looks for the |1...10...0> element. The size of the search space is not 2^n in this case, 
but smaller. 

1. First, the superposition circuit is created
```text
superposition = superposition_circuit(qubits, num_1)
```
This circuit has `qubits` superposition qubits and some ancillas.

2. Then the oracle is created
```text
oracle = oracle(qubits, num_1)
or_circuit = Circuit(oracle.nqubits)
or_circuit.add(oracle.on_qubits(*(list(range(qubits)) + [oracle.nqubits - 1] + list(range(qubits, oracle.nqubits - 1)))))
```
The `oracle` object has `qubits` qubits for the superposition, an ancilla qubit detecting whether the conditions are 
fulfilled or not, and some other auxiliary ancillas. In order to relabel the qubits so that the important ancilla is
at the bottom of the circuit, we must add two more lines and use a feature provided by QIBO.

Again, calling the Grover model is enough to execute the circuit. The binomial function allows to obtain the exact size
of the search space.
```text
grover = Grover(or_circuit, superposition_circuit=superposition, superposition_qubits=qubits, number_solutions=1,
                superposition_size=int(binomial(qubits, num_1)))
```
