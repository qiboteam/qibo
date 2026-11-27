# Noise-aware transpilation to a chip with star connectivity

Code at: [https://github.com/qiboteam/qibo/tree/main/examples/noise_aware_transpilation](https://github.com/qiboteam/qibo/tree/main/examples/noise_aware_transpilation).

This tutorial transpiles a Hamming-weight encoder to a 5-qubit chip with star connectivity and
the native gates `CZ` and `U3`. The placement of the qubits and the routing depend on the noise
of the chip, which is read from the calibration data that `qibocal` exports. It has four steps:

1. build the circuit with `qibo.models.encodings.hamming_weight_encoder`,
2. get the calibration data from `qibocal` and build a `HardwareModel`,
3. transpile with the noise-aware pipeline of `qibo.transpiler`, with and without the hardware
   model and with and without `sampling_only`, and compare the results,
4. return the final, transpiled circuit.

The folder has two files: `main.py`, which runs all the steps, and `calibration.json`, which is
the input of the second step. Run the script from this folder, *i.e.* `python main.py`.

#### Step 1: the Hamming-weight encoder

The encoder of weight `k = 1` on `n = 3` qubits loads `3 choose 1 = 3` amplitudes in the
bitstrings `100`, `010` and `001`. The real and the imaginary parts of the three amplitudes are
random numbers in `[-1, 1)`, generated with `backend.random_sample` and a fixed seed, so the
example is reproducible. The encoder infers the type of the data from its data type. The data is
complex, so the circuit has `RZ` gates, and a controlled-`RZ` gate, that load the phases. With
real data, such as the real parts alone, there are no `RZ` gates. The circuit ends with a
measurement of the three qubits.

```python
from qibo import gates
from qibo.backends import get_backend
from qibo.models.encodings import hamming_weight_encoder

backend = get_backend()

nqubits, weight = 3, 1
samples = 2 * backend.random_sample((2, 3), seed=10, dtype=backend.float64) - 1
data = backend.cast(samples[0] + 1j * samples[1], dtype=backend.complex128)
circuit = hamming_weight_encoder(nqubits, weight, data=data, backend=backend)
circuit.add(gates.M(*range(nqubits)))
circuit.draw()
```

```
0: ───RBS─RZ─RBS─RZ─RZ─M─
1: ───|──────RBS─RZ─o──M─
2: ─X─RBS─RZ───────────M─
```

The circuit has gates, such as `RBS`, that are not native, and it is not written for the
connectivity of the chip, so it has to be transpiled.

#### Step 2: calibration data from `qibocal`

`qibocal` writes a `calibration.json` file with the coherence times, the randomized
benchmarking fidelities, the readout fidelities and the fidelities of the `CZ` gate of each
coupling. The chip of this tutorial has qubits `0` to `4`, with qubit `2` at the center of the
star, so the couplings are `0-2`, `1-2`, `2-3` and `2-4`.

The file is produced by running the calibration protocols of `qibocal` on the chip. For
how to write the runcards and run the protocols, see the
[`qibocal` documentation](https://qibo.science/qibocal/stable/).

> **Note:** the `calibration.json` of this folder follows the format of `qibocal`, and it was
> validated with its `Calibration` model, but its values are **illustrative**, they were not
> measured on a chip. The `dummy` platform of `qibolab` has the same star of five qubits, but its
> instrument returns random data, so it cannot be used to get meaningful values.

The file is loaded with `json` and converted to a `qibo.transpiler.hardware.HardwareModel`. The
native gates are `CZ`, `U3` and the measurement `M`, and the durations, in ns, are used to split
the error of each gate into the thermal relaxation of its qubits, from `t1` and `t2`, and a
depolarizing error. The connectivity is the set of couplings of the file, so it is the star
around qubit `2`.

```python
import json
from pathlib import Path

from qibo.transpiler import HardwareModel, NativeGates

calibration = json.loads(Path("calibration.json").read_text())
native_gates = NativeGates.CZ | NativeGates.U3 | NativeGates.M
hardware = HardwareModel.from_calibration(
    calibration, durations={"1q": 40.0, "2q": 60.0}, native_gates=native_gates
)
print(sorted(hardware.connectivity.edges))
```

```
[(0, 2), (2, 1), (2, 3), (2, 4)]
```

The errors that the model takes from the file are the following ones, in the order of the
qubits and of the couplings of the chip.

<table class="docutils">
<thead>
<tr>
<th>Qubit</th>
<th><code>t1</code> (µs)</th>
<th><code>t2</code> (µs)</th>
<th>Readout error</th>
</tr>
</thead>
<tbody>
<tr><td>0</td><td>32</td><td>24</td><td>0.035</td></tr>
<tr><td>1</td><td>21</td><td>15</td><td>0.060</td></tr>
<tr><td>2</td><td>45</td><td>31</td><td>0.025</td></tr>
<tr><td>3</td><td>38</td><td>29</td><td>0.030</td></tr>
<tr><td>4</td><td>27</td><td>18</td><td>0.050</td></tr>
</tbody>
</table>

<table class="docutils">
<thead>
<tr><th>Coupling</th><th><code>CZ</code> error</th></tr>
</thead>
<tbody>
<tr><td>0-2</td><td>0.015</td></tr>
<tr><td>1-2</td><td>0.038</td></tr>
<tr><td>2-3</td><td>0.009</td></tr>
<tr><td>2-4</td><td>0.028</td></tr>
</tbody>
</table>

#### Step 3: transpilation, with and without noise awareness

`qibo.transpiler.transpile` builds the pipeline of the chosen `optimization_level` and applies
it to the circuit. We use level `2` and compare four transpilations:

- without the hardware model (noise-unaware) and with it (noise-aware), through the argument
  `hardware`. At levels `1` and `2`, `hardware` makes the pipeline use its errors to search
  the initial layout, where a layout without SWAP gates is not kept just for being free of
  SWAP gates, so that the one with the least noise is chosen, to route with the `"noise"`
  heuristic of the `"sabre"` router, which adds the SWAP gates on the least noisy couplings,
  to choose other physical qubits for the routed circuit with
  `qibo.transpiler.placer.NoiseAwarePostLayout`, and to optimize the single-qubit gates. At
  level `0` it has no effect.
- without and with `sampling_only=True`, which also enables the rewrites that keep the
  distribution of the measured qubits but can change the state before the measurements. One of
  them removes the diagonal gates, such as `RZ`, that are placed right before the measurements.
  It only has an effect on circuits with measurements.

Each result is checked with `assert_transpiling`, which raises an error if the transpiled
circuit does not respect the connectivity and the native gates, or if it is not equivalent to
the original circuit. With `sampling_only` it is not equivalent, so that check is skipped, and
the probabilities of the measured qubits of the transpiled circuit are compared with the ones
of the original circuit instead. The estimated fidelity is the product of the fidelities of
the `U3` and `CZ` gates, which are `1 - error` for the qubit or the coupling of each gate. It
does not include the readout error nor the decoherence of the idle qubits, and each `U3` gate
is charged one single-qubit error. It only depends on the number of gates and on the qubits
where they act. Each wire of a transpiled circuit is the hardware qubit of
`transpiled_circuit.wire_names`.

```python
from qibo.transpiler import transpile
from qibo.transpiler.asserts import assert_transpiling

configurations = (
    ("noise-unaware", None, False),
    ("noise-unaware, sampling_only", None, True),
    ("noise-aware", hardware, False),
    ("noise-aware, sampling_only", hardware, True),
)
probabilities = backend.execute_circuit(circuit, nshots=1).probabilities(
    qubits=tuple(range(nqubits))
)
transpiled_circuits, summary = {}, []
for label, model, sampling_only in configurations:
    transpiled_circuit, layout = transpile(
        circuit,
        connectivity=hardware.connectivity,
        native_gates=native_gates,
        optimization_level=2,
        seed=42,
        sampling_only=sampling_only,
        hardware=model,
        backend=backend,
    )

    assert_transpiling(
        original_circuit=circuit,
        transpiled_circuit=transpiled_circuit,
        connectivity=hardware.connectivity,
        final_layout=layout,
        native_gates=native_gates,
        check_circuit_equivalence=not sampling_only,
    )
    transpiled_probabilities = backend.execute_circuit(
        transpiled_circuit, nshots=1
    ).probabilities(qubits=tuple(transpiled_circuit.measurements[0].qubits))
    backend.assert_allclose(transpiled_probabilities, probabilities, atol=1e-8)

    fidelity = 1.0
    for gate in transpiled_circuit.queue:
        if isinstance(gate, (gates.CZ, gates.U3)):
            qubits = tuple(
                transpiled_circuit.wire_names[qubit] for qubit in gate.qubits
            )
            fidelity *= 1 - (hardware.error(qubits) or 0.0)

    print(f"\n{label}")
    # hardware qubits on which the three qubits of the circuit are placed
    print(layout.initial_layout[:nqubits])
    transpiled_circuit.draw()
    print(f"estimated fidelity {fidelity:.4f}")

    transpiled_circuits[label] = transpiled_circuit
    summary.append(
        (
            label,
            transpiled_circuit.gate_types[gates.CZ],
            transpiled_circuit.gate_types[gates.U3],
            fidelity,
        )
    )
```

The four results are presented in the order of the loop. In each of them, the first line is the
list of the hardware qubits on which the three qubits of the circuit are placed.

**Noise-unaware.** The placement does not know the errors of the chip, and it uses the
couplings `0-2`, four times, and `1-2`, twice. The latter has the highest `CZ` error.

```
[2, 0, 1]
Global phase = exp(i * 3π/2)
2: ─U3─Z─U3─Z─U3─o─U3─o─U3─Z─U3─Z─U3─M─
0: ────|────|─U3─Z─U3─Z─U3─o────o────M─
1: ─U3─o─U3─o─U3─────────────────────M─
3: ────────────────────────────────────
4: ────────────────────────────────────
estimated fidelity 0.8543
```

**Noise-unaware, with `sampling_only`.** The `RZ` gates before the measurements are removed,
and the circuit has two `CZ` gates and two `U3` gates less. The placement is the same.

```
[2, 0, 1]
Global phase = exp(i * π/2)
2: ─U3─Z─U3─Z─U3─o─U3─o─U3─M─
0: ────|────|─U3─Z─U3─Z─U3─M─
1: ─U3─o─U3─o─U3───────────M─
3: ──────────────────────────
4: ──────────────────────────
estimated fidelity 0.8824
```

**Noise-aware.** The placement is chosen with the errors of the chip. The circuit uses the
couplings `2-3`, four times, and `0-2`, twice, and does not use the coupling `1-2`.

```
[2, 3, 0]
Global phase = exp(i * 3π/2)
2: ─U3─Z─U3─Z─U3─o─U3─o─U3─Z─U3─Z─U3─M─
3: ────|────|─U3─Z─U3─Z─U3─o────o────M─
0: ─U3─o─U3─o─U3─────────────────────M─
1: ────────────────────────────────────
4: ────────────────────────────────────
estimated fidelity 0.9208
```

**Noise-aware, with `sampling_only`.** Both effects add up. The circuit uses the couplings
`2-3` and `0-2`, twice each, which are the two with the lowest error.

```
[2, 0, 3]
Global phase = exp(i * π/2)
2: ─U3─Z─U3─Z─U3─o─U3─o─U3─M─
0: ────|────|─U3─Z─U3─Z─U3─M─
3: ─U3─o─U3─o─U3───────────M─
1: ──────────────────────────
4: ──────────────────────────
estimated fidelity 0.9396
```

#### Step 4: the final circuit and the summary

The final circuit is the noise-aware one with `sampling_only`, which is the last one of the
loop.

```python
final_circuit = transpiled_circuits["noise-aware, sampling_only"]

print(f"\n{'transpilation':<30} {'CZ':>3} {'U3':>3} {'estimated fidelity':>19}")
for label, ncz, nu3, fidelity in summary:
    print(f"{label:<30} {ncz:>3} {nu3:>3} {fidelity:>19.4f}")
```

<table class="docutils">
<thead>
<tr>
<th>Transpilation</th>
<th><code>CZ</code></th>
<th><code>U3</code></th>
<th>Estimated fidelity</th>
</tr>
</thead>
<tbody>
<tr><td>noise-unaware</td><td>6</td><td>13</td><td>0.8543</td></tr>
<tr><td>noise-unaware, sampling_only</td><td>4</td><td>11</td><td>0.8824</td></tr>
<tr><td>noise-aware</td><td>6</td><td>13</td><td>0.9208</td></tr>
<tr><td>noise-aware, sampling_only</td><td>4</td><td>11</td><td>0.9396</td></tr>
</tbody>
</table>

Three points follow from the table:

- The hardware model raises the estimated fidelity from `0.8543` to `0.9208` without
  `sampling_only`, and from `0.8824` to `0.9396` with it. The circuits have the same number of
  gates, and the difference comes from the qubits and the couplings that they use.
- `sampling_only` raises it from `0.8543` to `0.8824` without the hardware model, and from
  `0.9208` to `0.9396` with it, because it removes two `CZ` gates and two `U3` gates. The
  distribution of the measured qubits is the same, but the state before the measurements is not,
  so it can only be used if the circuit is only run to sample from it.
- It only has an effect if there are diagonal gates before the measurements. If the data of
  Step 1 is real, such as the real parts alone, the encoder has no `RZ` gates, and the four
  transpilations have four `CZ` gates and ten `U3` gates, with the estimated fidelities `0.8834`,
  without the hardware model, and `0.9406`, with it.
