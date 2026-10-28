# Quantum Phase Estimation

Code at: [https://github.com/qiboteam/qibo/tree/main/examples/qpe](https://github.com/qiboteam/qibo/tree/main/examples/qpe).

Quantum Phase Estimation (QPE) is a building block of many quantum algorithms, such as
Shor's factorization and quantum simulation. Given a unitary `U` and one of its eigenstates
`|ψ>`, such that

```
U|ψ> = e^{2πiφ}|ψ>,
```

QPE estimates the phase `φ`, which belongs to the interval `[0, 1)`, with `t` bits of precision.
In its canonical form, the algorithm uses `t` counting qubits, initially in `|0>`, and as many
target qubits as the ones `U` acts on, prepared in `|ψ>`:

1. A Hadamard gate is applied to each counting qubit.
2. For each counting qubit `j = 0, ..., t - 1`, the unitary `U` is applied `2^{t - 1 - j}` times
   to the target qubits, controlled by the counting qubit `j`. Since `|ψ>` is an eigenstate, the
   controlled `U` kicks back a phase to the counting qubit, which is left in the state
   `(|0> + e^{2πiφ 2^{t - 1 - j}}|1>) / √2`. This phase depends on the binary digits of `φ`
   from the `(j + 1)`-th on.
3. The inverse Quantum Fourier Transform is applied to the counting qubits, which turns
   these phases into the digits of `φ`.
4. The counting qubits are measured. The measured bitstring `b_1 b_2 ... b_t` approximates the
   phase as the binary fraction `φ ≈ 0.b_1 b_2 ... b_t`, which is exact if `φ` is an integer
   multiple of `2^{-t}`.

For more details, check

- A. Yu. Kitaev, *Quantum measurements and the Abelian Stabilizer Problem*: [arXiv:quant-ph/9511026](https://arxiv.org/abs/quant-ph/9511026)
- R. Cleve, A. Ekert, C. Macchiavello and M. Mosca, *Quantum algorithms revisited*: [arXiv:quant-ph/9708016](https://arxiv.org/abs/quant-ph/9708016)
- M. A. Nielsen and I. L. Chuang, *Quantum Computation and Quantum Information*, Sec. 5.2.

## QPE in Qibo

The circuit is accessible as `from qibo.models import QPE`. Its arguments are

- unitary (`qibo.models.circuit.Circuit`): circuit that implements `U`. It cannot contain
  measurements or any other gate that cannot be controlled (see
  `qibo.models.circuit.Circuit.controlled_by`).
- nqubits_counting (int): number of counting qubits `t`.
- measure (bool): if `True`, the counting qubits are measured at the end of the circuit.
  Defaults to `True`.

The returned circuit has `t + n` qubits, where `n` is the number of qubits of `unitary`. The first
`t` qubits are the counting qubits, with the qubit `0` being the most significant one, and the
last `n` qubits are the target qubits, in the same order as the qubits of `unitary`.

The circuit does not prepare the eigenstate `|ψ>`. This has to be done by adding the circuit
that prepares it to a larger circuit before the QPE circuit, as shown in the examples below.

The measured bitstrings can be converted into phases with
`from qibo.quantum_info import bitstring_to_phase`.

We provide three examples of how to use QPE.

### Example 1: a phase that is exactly representable with the counting qubits

In this first example, we estimate the phase of the single-qubit gate

```
U = diag(1, e^{2πiφ}),  with φ = 5 / 16,
```

whose eigenstate `|1>` has phase `φ`. Since `5 / 16 = 0.0101` in binary, four counting qubits
are enough to obtain it exactly.

```python
import math

from qibo import Circuit, gates
from qibo.models import QPE
from qibo.quantum_info import bitstring_to_phase

phase = 5 / 16
unitary = Circuit(1)
unitary.add(gates.U1(0, theta=2 * math.pi * phase))

nqubits_counting = 4
qpe = QPE(unitary, nqubits_counting)

# prepare the eigenstate |1> in the target qubit, which is the last qubit
circuit = Circuit(qpe.nqubits)
circuit.add(gates.X(nqubits_counting))
circuit.add(qpe)

frequencies = circuit(nshots=100).frequencies()
print(frequencies)
print(bitstring_to_phase(frequencies.most_common(1)[0][0]))
```

```
Counter({'0101': 100})
0.3125
```

All the shots return the bitstring `0101`, which is the binary representation of `φ`.

### Example 2: a phase that is not exactly representable with the counting qubits

In general, `φ` is not an integer multiple of `2^{-t}`. In this case, the measured bitstring is
random and its distribution is peaked around `φ`. The most probable bitstring is the best
`t`-bit estimate of `φ`, with a probability of at least `4 / π^2 ≈ 0.405` (see Sec. 5.2.1 of
Nielsen and Chuang), and the error of this estimate is at most `2^{-(t + 1)}`.

Here we use `φ = 1 / 3`, which has an infinite binary expansion, and we compute the exact
probabilities of the counting qubits with `measure=False` instead of sampling them.

```python
import math

from qibo import Circuit, gates
from qibo.models import QPE

phase = 1 / 3
unitary = Circuit(1)
unitary.add(gates.U1(0, theta=2 * math.pi * phase))

print("t  estimate  error     probability")
for nqubits_counting in range(3, 9):
    qpe = QPE(unitary, nqubits_counting, measure=False)
    circuit = Circuit(qpe.nqubits)
    circuit.add(gates.X(nqubits_counting))
    circuit.add(qpe)

    probabilities = circuit().probabilities(qubits=list(range(nqubits_counting)))
    best = int(probabilities.argmax())
    estimate = best / 2**nqubits_counting
    print(
        f"{nqubits_counting}  {estimate:.6f}  {abs(estimate - phase):.2e}  "
        f"{float(probabilities[best]):.3f}"
    )
```

```
t  estimate  error     probability
3  0.375000  4.17e-02  0.688
4  0.312500  2.08e-02  0.685
5  0.343750  1.04e-02  0.684
6  0.328125  5.21e-03  0.684
7  0.335938  2.60e-03  0.684
8  0.332031  1.30e-03  0.684
```

Each additional counting qubit halves the error of the estimate, while the probability of
obtaining it stays above the `4 / π^2` bound. The cost is that the unitary is applied `2^t - 1`
times in total, so the depth of the circuit doubles as well.

### Example 3: energies of a Hamiltonian

QPE can be used to estimate the energies of a Hamiltonian `H`. For that, we use as unitary the
time evolution `U = e^{-iτH}`, whose eigenvalue for an eigenstate with energy `E` is `e^{2πiφ}`,
with `φ = -τE / (2π)` modulo `1`. After the estimation, the energy is recovered with

```
E = -2πφ / τ,
```

where phases above `1 / 2` are shifted to `φ - 1`, which gives positive energies. This assumes
that `τ|E| < π` for all the energies, otherwise different energies would have the same phase.
The energy resolution is `2π / (τ 2^t)`.

The full script is in [main.py](main.py), and it can be run with

```
python main.py --nqubits-counting 8 --nshots 1000 --tau 1.0
```

We consider the transverse-field Ising Hamiltonian of two qubits,
`hamiltonians.TFIM(2, h=0.5)`, and we implement `U` with a `gates.Unitary`.

```python
from qibo import Circuit, gates, hamiltonians
from qibo.models import QPE

tau = 1.0
nqubits_counting = 8

hamiltonian = hamiltonians.TFIM(2, h=0.5)
energies = hamiltonian.eigenvalues()
eigenvectors = hamiltonian.eigenvectors()

unitary = Circuit(2)
unitary.add(gates.Unitary(hamiltonian.exp(tau), 0, 1))
qpe = QPE(unitary, nqubits_counting)
```

The eigenvectors of the Hamiltonian are the columns of a unitary matrix, which maps the state
`|00>` into the ground state and `|11>` into the state with the highest energy. Hence, we can
use it to prepare the eigenstates in the target qubits.

```python
import math

from qibo.quantum_info import bitstring_to_phase

target_qubits = (nqubits_counting, nqubits_counting + 1)

# ground state
circuit = Circuit(qpe.nqubits)
circuit.add(gates.Unitary(eigenvectors, *target_qubits))
circuit.add(qpe)

frequencies = circuit(nshots=1000).frequencies()
phase = bitstring_to_phase(frequencies.most_common(1)[0][0])
print(
    f"Estimated energy: {-2 * math.pi * phase / tau:.4f}, exact energy: {energies[0]:.4f}"
)
```

```
Estimated energy: -1.4235, exact energy: -1.4142
```

The error of the estimate is smaller than the energy resolution, which is `0.0245` for 8
counting qubits and `τ = 1`.

If the target qubits are prepared in a superposition of eigenstates, the distribution of the
counting qubits has a peak for each eigenstate in the superposition. For instance, we can use
the superposition of the ground state and the state with the highest energy, which are
associated with the states `|00>` and `|11>`, respectively.

```python
circuit = Circuit(qpe.nqubits)
circuit.add(gates.H(target_qubits[0]))
circuit.add(gates.CNOT(*target_qubits))
circuit.add(gates.Unitary(eigenvectors, *target_qubits))
circuit.add(qpe)

frequencies = circuit(nshots=1000).frequencies()
for bitstring, counts in frequencies.most_common(2):
    phase = bitstring_to_phase(bitstring)
    energy = -2 * math.pi * (phase if phase < 0.5 else phase - 1) / tau
    print(f"Phase {phase} in {counts} shots, energy {energy:.4f}")
```

```
Phase 0.7734375 in 312 shots, energy 1.4235
Phase 0.2265625 in 297 shots, energy -1.4235
```

The two highest peaks are the estimates of the lowest and the highest energies of the
Hamiltonian, `-√2` and `√2`. The one with the phase above `1 / 2` is shifted as explained
above. Since the phases are not exactly representable with 8 counting qubits, the remaining
shots are spread over the bitstrings next to the peaks. The number of shots in each peak, and
their order, change between runs because of the sampling.
