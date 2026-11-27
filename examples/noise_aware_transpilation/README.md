# Noise-aware transpilation to a chip with star connectivity

Code at: [https://github.com/qiboteam/qibo/tree/main/examples/noise_aware_transpilation](https://github.com/qiboteam/qibo/tree/main/examples/noise_aware_transpilation).

This tutorial transpiles a Hamming-weight encoder to a 5-qubit chip with star connectivity and
the native gates `CZ` and `U3`. The placement of the qubits and the routing depend on the noise
of the chip, which is read from the calibration data that `qibocal` exports. It has four steps:

1. build the circuit with `qibo.models.encodings.hamming_weight_encoder`,
2. get the calibration data from `qibocal` and build a `HardwareModel`,
3. transpile with the noise-aware pipeline of `qibo.transpiler`,
4. return the final, transpiled circuit.

The folder has two files: `main.py`, which runs all the steps, and `calibration.json`, which is
the input of the second step. Run the script from this folder, *i.e.* `python main.py`.

#### Step 1: the Hamming-weight encoder

The encoder of weight `k = 1` on `n = 3` qubits loads `3 choose 1 = 3` amplitudes in the
bitstrings `100`, `010` and `001`. Without `data`, the circuit is returned with all the phases
set to `0.0`, so we pass three amplitudes.

```python
from qibo.backends import get_backend
from qibo.models.encodings import hamming_weight_encoder

backend = get_backend()

nqubits, weight = 3, 1
# one amplitude per bitstring of Hamming weight 1, which are 3 choose 1 = 3
data = backend.cast([0.6, 0.3, 0.74])
circuit = hamming_weight_encoder(nqubits, weight, data=data, backend=backend)
circuit.draw()
```

```
0: ───RBS─RZ─RBS─RZ─RZ─
1: ───|──────RBS─RZ─o──
2: ─X─RBS─RZ───────────
```

The circuit has gates, such as `RBS`, that are not native, and it acts on qubits that are not
connected in a star, so it has to be transpiled.

#### Step 2: calibration data from `qibocal`

`qibocal` writes a `calibration.json` file with the coherence times, the randomized
benchmarking fidelities, the readout fidelities and the fidelities of the `CZ` gate of each
coupling. The chip of this tutorial has qubits `0` to `4`, with qubit `2` at the center of the
star, so the couplings are `0-2`, `1-2`, `2-3` and `2-4`.

On a chip, the file is produced by a runcard like the following one. The platform name is a
placeholder, and the depths and the delays have to be adapted to the chip.

```yaml
platform: my_star_chip
targets: [0, 1, 2, 3, 4]

actions:
  - id: readout characterization
    operation: readout_characterization
    parameters: {nshots: 1000}

  - id: t1
    operation: t1
    parameters:
      delay_before_readout_start: 50
      delay_before_readout_end: 20000
      delay_before_readout_step: 500

  - id: t2
    operation: t2
    parameters:
      delay_between_pulses_start: 50
      delay_between_pulses_end: 20000
      delay_between_pulses_step: 500

  - id: standard rb
    operation: standard_rb
    parameters: {depths: [1, 5, 10, 20, 50], niter: 5, nshots: 100}

  # one pair of actions for each of the couplings: 0-2, 1-2, 2-3 and 2-4
  - id: standard_rb_2q 0-2
    operation: standard_rb_2q
    targets: [[0, 2]]
    parameters: {depths: [1, 2, 3, 5, 8], niter: 5, nshots: 100}

  - id: standard_rb_2q_inter 0-2
    operation: standard_rb_2q_inter
    targets: [[0, 2]]
    parameters: {depths: [1, 2, 3, 5, 8], niter: 5, nshots: 100}
```

```
qq run runcard.yml -o calibration_run
```

A few points about the use of `qibocal`:

- The targets of an action cannot share a qubit, so each coupling needs its own action. All the
  couplings of a star share the center qubit.
- `standard_rb_2q` has to run before `standard_rb_2q_inter` on each coupling, because the
  interleaved benchmark is the one that writes the fidelity of the `CZ` gate, `cz_fidelity`.
- The platform folder must already have a `calibration.json` with the initial parameters of the
  qubits, such as their frequencies.
- If the output folder already has the original platform, the updated calibration is written
  to `calibration_run/new_platform/calibration.json`.

> **Note:** the `calibration.json` of this folder follows the format of `qibocal`, and it was
> validated with its `Calibration` model, but its values are **illustrative**, they were not
> measured on a chip. The `dummy` platform of `qibolab` has the same star of five qubits, but its
> instrument returns random data, so it cannot be used to get meaningful values.

The file is loaded with `json` and converted to a `qibo.transpiler.hardware.HardwareModel`. The
native gates are `CZ` and `U3`, and the durations, in ns, are used to split the error of each
gate into the thermal relaxation of its qubits, from `t1` and `t2`, and a depolarizing error.
The connectivity is the set of couplings of the file, so it is the star around qubit `2`.

```python
import json
from pathlib import Path

from qibo.transpiler import HardwareModel, NativeGates

calibration = json.loads(Path("calibration.json").read_text())
native_gates = NativeGates.CZ | NativeGates.U3
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

| Qubit | `t1` (µs) | `t2` (µs) | Readout error |
|-------|-----------|-----------|---------------|
| 0     | 32        | 24        | 0.035         |
| 1     | 21        | 15        | 0.060         |
| 2     | 45        | 31        | 0.025         |
| 3     | 38        | 29        | 0.030         |
| 4     | 27        | 18        | 0.050         |

| Coupling | `CZ` error |
|----------|------------|
| 0-2      | 0.015      |
| 1-2      | 0.038      |
| 2-3      | 0.009      |
| 2-4      | 0.028      |

#### Step 3: noise-aware transpilation

`qibo.transpiler.transpile` builds the pipeline of the chosen `optimization_level` and applies
it to the circuit. At level `2`, if `hardware` is given, the pipeline uses its errors to:

- search the initial layout, where a layout without SWAP gates is not kept just for being
  free of SWAP gates, so that the one with the least noise is chosen,
- route with the `"noise"` heuristic of the `"sabre"` router, which adds the SWAP gates on the
  least noisy couplings,
- choose other physical qubits for the routed circuit with
  `qibo.transpiler.placer.NoiseAwarePostLayout`,
- optimize the single-qubit gates, and, if `approximation_degree=None`, remove the gates whose
  infidelity is lower than their error.

```python
from qibo.transpiler import transpile
from qibo.transpiler.asserts import assert_transpiling

transpiled_circuit, layout = transpile(
    circuit,
    connectivity=hardware.connectivity,
    native_gates=native_gates,
    optimization_level=2,
    seed=42,
    hardware=hardware,
    backend=backend,
)

# Raises an error if the circuits are not equivalent, or if the transpiled circuit does not
# respect the connectivity and the native gates
assert_transpiling(
    original_circuit=circuit,
    transpiled_circuit=transpiled_circuit,
    connectivity=hardware.connectivity,
    final_layout=layout,
    native_gates=native_gates,
)
```

#### Step 4: the transpiled circuit

The transpiled circuit has one wire for each qubit of the chip. The qubits of the encoder are
placed on the physical qubits given by `layout.initial_layout`, and the other two qubits of the
chip are added as auxiliary qubits that are not used. The wire `i` of the circuit is the
physical qubit `transpiled_circuit.wire_names[i]`.

```python
# hardware qubits on which the three qubits of the circuit are placed
print(layout.initial_layout[:nqubits])
transpiled_circuit.draw()
```

```
[2, 0, 3]
Global phase = exp(i * π/2)
2: ─U3─Z─U3─Z────o─U3─o─U3─
0: ────|────|─U3─Z─U3─Z─U3─
3: ─U3─o─U3─o─U3───────────
1: ────────────────────────
4: ────────────────────────
```

The circuit only has the native gates `U3` and `CZ`, up to a global phase, and the `CZ` gates act
on the couplings `2-3` and `0-2` only, which are the two with the lowest error. The couplings
`1-2` and `2-4` are the noisiest ones, and the qubits `1` and `4` are the noisiest qubits, so none
of them is used. No SWAP gate was needed in this run.

#### Comparison with a noise-unaware transpilation

The same transpilation without `hardware` does not know the errors of the chip. The two
results are compared with a rough estimate of the fidelity, which is the product of
`1 - error` over the `U3` and `CZ` gates, and does not include the readout error nor the
decoherence of the idle qubits. Each `U3` gate is charged one single-qubit error.

```python
from qibo import gates

baseline_circuit, _ = transpile(
    circuit,
    connectivity=hardware.connectivity,
    native_gates=native_gates,
    optimization_level=2,
    seed=42,
    backend=backend,
)
for label, result in (
    ("noise-aware", transpiled_circuit),
    ("noise-unaware", baseline_circuit),
):
    fidelity = 1.0
    for gate in result.queue:
        if isinstance(gate, (gates.CZ, gates.U3)):
            qubits = tuple(result.wire_names[qubit] for qubit in gate.qubits)
            fidelity *= 1 - (hardware.error(qubits) or 0.0)
    print(f"{label}: estimated fidelity {fidelity:.4f}")
```

```
noise-aware: estimated fidelity 0.9406
noise-unaware: estimated fidelity 0.8834
```

Both circuits have four `CZ` gates, but the noise-unaware one places a qubit of the encoder on
the qubit `1`, which has the noisiest coupling to the center of the star. In this example, the
choice of the qubits and of the couplings changes the estimated fidelity by about 6 percentage
points.
