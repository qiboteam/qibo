"""Noise-aware transpilation of a Hamming-weight encoder to a star-connectivity chip."""

import json
from pathlib import Path

from qibo import gates
from qibo.backends import get_backend
from qibo.models.encodings import hamming_weight_encoder
from qibo.transpiler import HardwareModel, NativeGates, transpile
from qibo.transpiler.asserts import assert_transpiling

backend = get_backend()

# (i) Circuit that loads three amplitudes in the Hamming-weight-1 subspace of 3 qubits, with
# a measurement of all the qubits. There is one amplitude per bitstring of Hamming weight 1,
# which are 3 choose 1 = 3. The real and imaginary parts of the amplitudes are random numbers
# in [-1, 1) with a fixed seed, and the complex data makes the encoder add the RZ gates that
# load the phases.
nqubits, weight = 3, 1
samples = 2 * backend.random_sample((2, 3), seed=10, dtype=backend.float64) - 1
data = backend.cast(samples[0] + 1j * samples[1], dtype=backend.complex128)
circuit = hamming_weight_encoder(nqubits, weight, data=data, backend=backend)
circuit.add(gates.M(*range(nqubits)))
circuit.draw()

# (ii) Hardware model from the ``calibration.json`` that ``qibocal`` exports. The couplings
# of the file, which are the ones of the chip, define the star connectivity of qubit 2.
calibration = json.loads(Path("calibration.json").read_text())
native_gates = NativeGates.CZ | NativeGates.U3 | NativeGates.M
hardware = HardwareModel.from_calibration(
    calibration, durations={"1q": 40.0, "2q": 60.0}, native_gates=native_gates
)
print(sorted(hardware.connectivity.edges))

# (iii) Transpilation at optimization level 2, without and with the hardware model, and
# without and with ``sampling_only``. The hardware model makes the placement search, the
# router and the post layout choose the qubits and the couplings with less noise.
# ``sampling_only`` also removes the diagonal gates, such as RZ, placed right before the
# measurements, which keeps the distribution of the measured qubits.
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

    # Raises an error if the transpiled circuit does not respect the connectivity and the
    # native gates, or if it is not equivalent to the original circuit. With
    # ``sampling_only`` it is not, so the distribution of the measured qubits is compared.
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

    # Estimated fidelity: product of the fidelities of the gates, without readout and idle
    # errors. Each wire is the hardware qubit of ``wire_names``.
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

# (iv) Final, transpiled circuit: noise-aware, with ``sampling_only``
final_circuit = transpiled_circuits["noise-aware, sampling_only"]

# Summary of the four transpilations
print(f"\n{'transpilation':<30} {'CZ':>3} {'U3':>3} {'estimated fidelity':>19}")
for label, ncz, nu3, fidelity in summary:
    print(f"{label:<30} {ncz:>3} {nu3:>3} {fidelity:>19.4f}")
