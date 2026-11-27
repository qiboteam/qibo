"""Noise-aware transpilation of a Hamming-weight encoder to a star-connectivity chip."""

import json
from pathlib import Path

from qibo import gates
from qibo.backends import get_backend
from qibo.models.encodings import hamming_weight_encoder
from qibo.transpiler import HardwareModel, NativeGates, transpile
from qibo.transpiler.asserts import assert_transpiling

backend = get_backend()

# (i) Circuit that loads three amplitudes in the Hamming-weight-1 subspace of 3 qubits
nqubits, weight = 3, 1
# one amplitude per bitstring of Hamming weight 1, which are 3 choose 1 = 3
data = backend.cast([0.6, 0.3, 0.74])
circuit = hamming_weight_encoder(nqubits, weight, data=data, backend=backend)
circuit.draw()

# (ii) Hardware model from the ``calibration.json`` that ``qibocal`` exports. The couplings
# of the file, which are the ones of the chip, define the star connectivity of qubit 2.
calibration = json.loads(Path("calibration.json").read_text())
native_gates = NativeGates.CZ | NativeGates.U3
hardware = HardwareModel.from_calibration(
    calibration, durations={"1q": 40.0, "2q": 60.0}, native_gates=native_gates
)
print(sorted(hardware.connectivity.edges))

# (iii) Noise-aware transpilation. At optimization level 2, ``hardware`` makes the placement
# search, the router and the post layout choose the qubits and the couplings with less noise.
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

# (iv) Final, transpiled circuit. Each wire is the hardware qubit of ``wire_names``
# hardware qubits on which the three qubits of the circuit are placed
print(layout.initial_layout[:nqubits])
transpiled_circuit.draw()

# Check: the same transpilation without the hardware model, whose result is estimated
# with the product of the fidelities of the gates, without readout and idle errors
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
