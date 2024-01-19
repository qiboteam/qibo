from itertools import product

import numpy as np

from qibo import gates, symbols
from qibo.backends import GlobalBackend
from qibo.hamiltonians import SymbolicHamiltonian
from qibo.models import Circuit


def prepare_states(k, nqubits):
    """Prepares a quantum circuit in a particular state indexed by k.

        Args:
        k (int): The index of the state to be prepared.
        nqubits (int): Number of qubits in the circuit

    Returns:
        circuit (:class:`qibo.models.Circuit`): Qibo circuit.
    """

    if nqubits != 1 and nqubits != 2:
        raise ValueError(
            f"nqubits given as {nqubits}. nqubits needs to be either 1 or 2."
        )

    gates_list = [(gates.I,), (gates.X,), (gates.H,), (gates.H, gates.S)]
    k_to_gates = list(product(gates_list, repeat=nqubits))[k]
    circ = Circuit(nqubits, density_matrix=True)
    for q in range(len(k_to_gates)):
        gate_q = [gate(q) for gate in k_to_gates[q]]
        circ.add(gate_q)
    return circ


def measurement_basis(j, circ):
    """Implements a measurement basis for circuit.

        Args:
        circuit (:class:`qibo.models.Circuit`): Qibo circuit.
        j (int): The index of the measurement basis.

    Returns:
        circuit (:class:`qibo.models.Circuit`): Qibo circuit.
    """

    nqubits = circ.nqubits
    if nqubits != 1 and nqubits != 2:
        raise ValueError(
            f"nqubits given as {nqubits}. nqubits needs to be either 1 or 2."
        )

    meas_list = [gates.Z, gates.X, gates.Y, gates.Z]
    j_to_measurements = list(product(meas_list, repeat=nqubits))[j]
    new_circ = circ.copy()
    for q in range(len(j_to_measurements)):
        meas_q = j_to_measurements[q]
        new_circ.add(gates.M(q, basis=meas_q))

    return new_circ


def reset_register(circuit, invert_register):
    """Returns an inverse circuit of the selected register.
        One can then add inverse_circuit to the original circuit by addition:
            circ_with_inverse = circ.copy()
            circ_with_inverse.add(inverse_circuit.on_qubits(invert_register))
        where register_to_reset = 0, 1, or [0,1].

        Args:
        circuit (:class:`qibo.models.Circuit`): original circuit
        invert_register (string): Qubit(s) to reset:
            'sp_0' (qubit 0);
            'sp_1' (qubit 1); or
            'sp_t' (both qubits)
    Returns:
        inverse_circuit (:class:`qibo.models.Circuit`): Inverse of the input circuit's register.
    """

    if invert_register == "sp_0" or invert_register == "sp_1":
        if invert_register == "sp_0":
            register_to_reset = 0
        elif invert_register == "sp_1":
            register_to_reset = 1

        new_circ = Circuit(1)
        for data in circuit.raw["queue"]:
            theta = data.get("init_kwargs", {}).get("theta", None)
            if data["_target_qubits"][0] == register_to_reset:
                if theta is None:
                    new_circ.add(getattr(gates, data["_class"])(0))
                else:
                    new_circ.add(getattr(gates, data["_class"])(0, theta))

    elif invert_register == "sp_t":
        new_circ = circuit.copy()

    else:
        raise NameError(
            f"{invert_register} not recognized. Input "
            "sp_0"
            " to reset qubit 0, "
            "sp_1"
            " to reset qubit 1, or "
            "sp_t"
            " to reset both qubits."
        )

    inverse_circuit = new_circ.invert()

    return inverse_circuit


def GST_execute_circuit(circuit, k, j, nshots=int(1e4), backend=None):
    """Executes a circuit used in gate set tomography and processes the
        measurement outcomes for the Pauli Transfer Matrix notation. The circuit
        should already have noise models implemented, if any, prior to using this
        function.

        Args:
        circuit (:class:`qibo.models.Circuit`): The Qibo circuit to be executed.
        k (int): The index of the state prepared.
        j (int): The index of the measurement basis.
        nshots (int, optional): Number of shots to execute circuit with.
    Returns:
        numpy.float: Expectation value given by either Tr(Q_j rho_k) or Tr(Q_j O_l rho_k).
    """

    if j == 0:
        return 1.0

    nqubits = circuit.nqubits
    if nqubits != 1 and nqubits != 2:
        raise ValueError(
            f"nqubits given as {nqubits}. nqubits needs to be either 1 or 2."
        )

    if backend is None:  # pragma: no cover
        backend = GlobalBackend()

    result = backend.execute_circuit(circuit, nshots=nshots)
    observables = [symbols.I, symbols.Z, symbols.Z, symbols.Z]
    observables_list = list(product(observables, repeat=nqubits))[j]
    observable = 1
    for q, obs in enumerate(observables_list):
        if obs is not symbols.I:
            observable *= obs(q)
    observable = SymbolicHamiltonian(observable, nqubits=nqubits)
    expectation_val = result.expectation_from_samples(observable)

    return expectation_val


def GST(
    nqubits=None,
    gate=None,
    nshots=int(1e4),
    invert_register=None,
    noise_model=None,
    backend=None,
):
    """Runs gate set tomography for a 1 or 2 qubit gate/circuit.

        Args:
        nshots (int, optional): Number of shots used in Gate Set Tomography.
        gate (:class:`qibo.gates.abstract.Gate`, optional): The gate to perform gate set tomography on.
            If gate=None, then gate set tomography will be performed for an empty circuit.
        noise_model (:class:`qibo.noise.NoiseModel`, optional): Noise model applied
            to simulate noisy computation.
        backend (:class:`qibo.backends.abstract.Backend`, optional): Calculation engine.
    Returns:
        numpy.array: Numpy array with elements jk equivalent to either Tr(Q_j rho_k) or Tr(Q_j O_l rho_k).
    """

    # Check if class_qubit_gate corresponds to nqubits-gate.
    if nqubits != 1 and nqubits != 2:
        raise ValueError(
            f"nqubits given as {nqubits}. nqubits needs to be either 1 or 2."
        )

    if backend is None:  # pragma: no cover
        backend = GlobalBackend()

    if gate is not None:
        qb_gate = len(gate.qubits)
        if nqubits != qb_gate:
            raise ValueError(
                f"Mismatched inputs: nqubits given as {nqubits}. {gate} is a {qb_gate}-qubit gate."
            )
        gate = gate.__class__(*list(range(qb_gate)), **gate.init_kwargs)

    # GST for empty circuit or with gates
    matrix_jk = np.zeros((4**nqubits, 4**nqubits))
    for k in range(4**nqubits):
        circ = prepare_states(k, nqubits)

        if invert_register is not None:
            inverted_circuit = reset_register(circ, invert_register)
            if invert_register == "sp_0":
                circ.add(inverted_circuit.on_qubits(0))
            elif invert_register == "sp_1":
                circ.add(inverted_circuit.on_qubits(1))
            elif invert_register == "sp_t":
                circ.add(inverted_circuit.on_qubits(0, 1))
            else:
                raise NameError(
                    f"{invert_register} not recognized. Input "
                    "sp_0"
                    " to reset qubit 0, "
                    "sp_1"
                    " to reset qubit 1, or "
                    "sp_t"
                    " to reset both qubits."
                )

        if gate is not None:
            circ.add(gate)

        for j in range(4**nqubits):
            new_circ = measurement_basis(j, circ)

            if noise_model is not None and backend.name != "qibolab":
                new_circ = noise_model.apply(new_circ)

            expectation_val = GST_execute_circuit(
                new_circ, k, j, nshots, backend=backend
            )
            matrix_jk[j, k] = expectation_val

    return matrix_jk
