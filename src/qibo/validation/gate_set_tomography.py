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


def reset_register(circuit, state_prep_string):
    """Returns an inverse circuit of the selected register.
        One can then add inverse_circuit to the original circuit by addition:
            circ_with_inverse = circ.copy()
            circ_with_inverse.add(inverse_circuit.on_qubits(register_to_reset))
        where register_to_reset = 0, 1, or [0,1].

        Args:
        circuit (:class:`qibo.models.Circuit`): original circuit
        state_prep_string (string): String to reset qubit:
            'sp_0' (qubit 0);
            'sp_1' (qubit 1); or
            'sp_t' (both qubits)
    Returns:
        inverse_circuit (:class:`qibo.models.Circuit`): Inverse of the input circuit's register.
    """

    if state_prep_string == "sp_0" or state_prep_string == "sp_1":
        if state_prep_string == "sp_0":
            register_to_reset = 0
        elif state_prep_string == "sp_1":
            register_to_reset = 1

        new_circ = Circuit(1)
        for data in circuit.raw["queue"]:
            theta = data.get("init_kwargs", {}).get("theta", None)
            if data["_target_qubits"][0] == register_to_reset:
                if theta is None:
                    new_circ.add(getattr(gates, data["_class"])(0))
                else:
                    new_circ.add(getattr(gates, data["_class"])(0, theta))

    elif state_prep_string == "sp_t":
        new_circ = circuit.copy()

    inverse_circuit = new_circ.invert()

    return inverse_circuit  # inverse_circuit.unitary()


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
    # print(observables_list)
    observable = 1
    for q, obs in enumerate(observables_list):
        if obs is not symbols.I:
            observable *= obs(q)
    observable = SymbolicHamiltonian(observable, nqubits=nqubits)
    expectation_val = result.expectation_from_samples(observable)

    return expectation_val


def GST(nqubits=None, gate=None, nshots=int(1e4), noise_model=None, backend=None):
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


def GST_basis_operations(
    nqubits=None, gjk=None, nshots=int(1e4), noise_model=None, backend=None
):
    """Runs gate set tomography for the basis operations. There are 13 basis operations for a single
        qubit and 241 basis operations for 2 qubits.

        Args:
        nshots (int, optional): Number of shots used in Gate Set Tomography.
        noise_model (:class:`qibo.noise.NoiseModel`, optional): Noise model applied
            to simulate noisy computation.
        gjk (numpy.array): Numpy array with elements Tr(Q_j rho_k) for with no gate.
            If None, call GST function with gate=None.
        backend (:class:`qibo.backends.abstract.Backend`, optional): Calculation engine.

        The 13 and 241 basis operations are based on Physical Review Research 3, 033178 (2021).
    Returns:
        numpy.array: Numpy array with elements jk equivalent to either Tr(Q_j rho_k) or Tr(Q_j O_l rho_k).
    """

    if backend is None:  # pragma: no cover
        backend = GlobalBackend()

    # Implement a check. If nqubits==1, gjk needs to be 4x4 matrix. If nqubits==2, gjk needs to be 16x16 matrix.
    if gjk is None:
        gjk = GST(
            nqubits, gate=None, nshots=nshots, noise_model=noise_model, backend=backend
        )
        print("gjk done")

    backend = GlobalBackend()

    # 1 qubit basis operations
    hadamard = gates.H(0).matrix(backend=backend)
    identity = gates.I(0).matrix(backend=backend)
    xgate = gates.X(0).matrix(backend=backend)
    ygate = gates.Y(0).matrix(backend=backend)
    zgate = gates.Z(0).matrix(backend=backend)
    sgate = gates.S(0).matrix(backend=backend)
    sdggate = gates.SDG(0).matrix(backend=backend)
    CNOT = gates.CNOT(0, 1).matrix(backend=backend)
    SWAP = gates.SWAP(0, 1).matrix(backend=backend)
    ry_piby4 = gates.RY(0, np.pi / 4).matrix(backend=backend)

    # One qubit basis operations
    BO_1qb = []
    BO_1qb.append(identity)
    BO_1qb.append(xgate)
    BO_1qb.append(ygate)
    BO_1qb.append(zgate)
    BO_1qb.append((1 / np.sqrt(2)) * np.array([[1, 1j], [1j, 1]]))
    BO_1qb.append((1 / np.sqrt(2)) * np.array([[1, 1], [-1, 1]]))
    BO_1qb.append((1 / np.sqrt(2)) * np.array([[(1 + 1j), 0], [0, (1 - 1j)]]))
    BO_1qb.append((1 / np.sqrt(2)) * np.array([[1, -1j], [1j, -1]]))
    BO_1qb.append(hadamard)
    BO_1qb.append((1 / np.sqrt(2)) * np.array([[0, (1 - 1j)], [(1 + 1j), 0]]))
    BO_1qb.append(hadamard)
    BO_1qb.append(hadamard @ sgate)
    BO_1qb.append(identity)

    # if nqubits == 1:
    BO_1qb
    gatenames_1qb = [f"BasisOp{j}" for j in range(len(BO_1qb))]
    gatenames_1qb[10] += "_sp_0"
    gatenames_1qb[11] += "_sp_0"
    gatenames_1qb[12] += "_sp_0"

    # 2 qubit basis operations
    # Intermediate operations: "interm_ops"
    BO_2qubit = []
    interm_ops = []
    interm_ops.append(CNOT)
    interm_ops.append(np.kron(xgate, identity) @ CNOT @ np.kron(xgate, identity))
    interm_ops.append(
        np.array(np.block([[identity, np.zeros((2, 2))], [np.zeros((2, 2)), sgate]]))
    )
    interm_ops.append(
        np.array(np.block([[identity, np.zeros((2, 2))], [np.zeros((2, 2)), hadamard]]))
    )
    interm_ops.append(np.kron(ry_piby4, identity) @ CNOT @ np.kron(ry_piby4, identity))
    interm_ops.append(CNOT @ np.kron(hadamard, identity))
    interm_ops.append(SWAP)
    interm_ops.append(
        np.array([[1, 0, 0, 0], [0, 0, 1j, 0], [0, 1j, 0, 0], [0, 0, 0, 1]])
    )
    interm_ops.append(SWAP @ np.kron(hadamard, identity))

    # Two qubit basis operations: "Easy"
    BO_2qb_easy = []
    for ii in range(0, 13):
        for jj in range(0, 13):
            temp_matrix = np.kron(BO_1qb[ii], BO_1qb[jj])
            BO_2qb_easy.append(temp_matrix)

    # Two qubit basis operations: "Hard"
    BO_2qb_hard = []
    for ii in range(0, 9):
        if ii == 6:
            temp_matrix1 = (
                np.kron(identity, sgate @ hadamard)
                @ interm_ops[ii]
                @ np.kron(identity, hadamard @ sdggate)
            )
            temp_matrix2 = (
                np.kron(identity, hadamard @ sdggate)
                @ interm_ops[ii]
                @ np.kron(identity, sgate @ hadamard)
            )
            temp_matrix3 = (
                np.kron(identity, identity)
                @ interm_ops[ii]
                @ np.kron(identity, identity)
            )

            BO_2qb_hard.append(temp_matrix1)
            BO_2qb_hard.append(temp_matrix2)
            BO_2qb_hard.append(temp_matrix3)

        elif ii == 7:
            temp_matrix1 = (
                np.kron(sgate @ hadamard, sgate @ hadamard)
                @ interm_ops[ii]
                @ np.kron(hadamard @ sdggate, hadamard @ sdggate)
            )
            temp_matrix2 = (
                np.kron(sgate @ hadamard, hadamard @ sdggate)
                @ interm_ops[ii]
                @ np.kron(hadamard @ sdggate, sgate @ hadamard)
            )
            temp_matrix3 = (
                np.kron(sgate @ hadamard, identity)
                @ interm_ops[ii]
                @ np.kron(hadamard @ sdggate, identity)
            )
            temp_matrix4 = (
                np.kron(identity @ hadamard, sgate @ hadamard)
                @ interm_ops[ii]
                @ np.kron(identity @ sdggate, hadamard @ sdggate)
            )
            temp_matrix5 = (
                np.kron(identity @ hadamard, hadamard @ sdggate)
                @ interm_ops[ii]
                @ np.kron(identity @ sdggate, sgate @ hadamard)
            )
            temp_matrix6 = (
                np.kron(identity, identity)
                @ interm_ops[ii]
                @ np.kron(identity, identity)
            )

            BO_2qb_hard.append(temp_matrix1)
            BO_2qb_hard.append(temp_matrix2)
            BO_2qb_hard.append(temp_matrix3)
            BO_2qb_hard.append(temp_matrix4)
            BO_2qb_hard.append(temp_matrix5)
            BO_2qb_hard.append(temp_matrix6)
        else:
            temp_matrix1 = (
                np.kron(sgate @ hadamard, sgate @ hadamard)
                @ interm_ops[ii]
                @ np.kron(hadamard @ sdggate, hadamard @ sdggate)
            )
            temp_matrix2 = (
                np.kron(sgate @ hadamard, hadamard @ sdggate)
                @ interm_ops[ii]
                @ np.kron(hadamard @ sdggate, sgate @ hadamard)
            )
            temp_matrix3 = (
                np.kron(sgate @ hadamard, identity)
                @ interm_ops[ii]
                @ np.kron(hadamard @ sdggate, identity)
            )

            temp_matrix4 = (
                np.kron(hadamard @ sdggate, sgate @ hadamard)
                @ interm_ops[ii]
                @ np.kron(sgate @ hadamard, hadamard @ sdggate)
            )
            temp_matrix5 = (
                np.kron(hadamard @ sdggate, hadamard @ sdggate)
                @ interm_ops[ii]
                @ np.kron(sgate @ hadamard, sgate @ hadamard)
            )
            temp_matrix6 = (
                np.kron(hadamard @ sdggate, identity)
                @ interm_ops[ii]
                @ np.kron(sgate @ hadamard, identity)
            )

            temp_matrix7 = (
                np.kron(identity, sgate @ hadamard)
                @ interm_ops[ii]
                @ np.kron(identity, hadamard @ sdggate)
            )
            temp_matrix8 = (
                np.kron(identity, hadamard @ sdggate)
                @ interm_ops[ii]
                @ np.kron(identity, sgate @ hadamard)
            )
            temp_matrix9 = (
                np.kron(identity, identity)
                @ interm_ops[ii]
                @ np.kron(identity, identity)
            )

            BO_2qb_hard.append(temp_matrix1)
            BO_2qb_hard.append(temp_matrix2)
            BO_2qb_hard.append(temp_matrix3)
            BO_2qb_hard.append(temp_matrix4)
            BO_2qb_hard.append(temp_matrix5)
            BO_2qb_hard.append(temp_matrix6)
            BO_2qb_hard.append(temp_matrix7)
            BO_2qb_hard.append(temp_matrix8)
            BO_2qb_hard.append(temp_matrix9)

    BO_2qb = BO_2qb_easy + BO_2qb_hard

    gatenames_2qb = [f"BasisOp{j}" for j in range(len(BO_2qb))]
    index = 0
    for idx_1 in range(len(BO_1qb)):
        for idx_2 in range(len(BO_1qb)):
            # PARTIALLY REPREPARE QUBIT_0
            if idx_1 in [10, 11, 12] and idx_2 < 10:
                gatenames_2qb[index] += "_sp_0"
            # PARTIALLY REPREPARE QUBIT_1
            elif idx_2 in [10, 11, 12] and idx_1 < 10:
                gatenames_2qb[index] += "_sp_1"
            elif idx_1 in [10, 11, 12] and idx_2 in [10, 11, 12]:
                gatenames_2qb[index] += "_sp_t"
            index += 1

    if nqubits is not None:
        if nqubits == 1:
            Bjk_tilde_1qb = np.zeros((13, 4, 4))
            for idx_basis_ops in range(0, 13):
                # Initialise 4 different initial states
                for k in range(0, 4):
                    circ = prepare_states(k, nqubits)

                    # Basis operation 0 to 9
                    if idx_basis_ops < 10:
                        circ.add(
                            gates.Unitary(
                                BO_1qb[idx_basis_ops],
                                0,
                                trainable=False,
                                name="%s" % (gatenames_1qb[idx_basis_ops]),
                            )
                        )

                    # Basis operation 10, 11, 12
                    elif idx_basis_ops == 10:  # Reset, prepare |+> state
                        inverse_circuit = reset_register(circ, "sp_0")
                        circ.add(inverse_circuit.on_qubits(0))
                        circ.add(
                            gates.Unitary(
                                BO_1qb[idx_basis_ops],
                                0,
                                trainable=False,
                                name="%s" % (gatenames_1qb[idx_basis_ops]),
                            )
                        )

                    elif idx_basis_ops == 11:  # Reset, prepare |y+> state
                        inverse_circuit = reset_register(circ, "sp_0")
                        circ.add(inverse_circuit.on_qubits(0))
                        circ.add(
                            gates.Unitary(
                                BO_1qb[idx_basis_ops],
                                0,
                                trainable=False,
                                name="%s" % (gatenames_1qb[idx_basis_ops]),
                            )
                        )

                    elif idx_basis_ops == 12:  # Reset, prepare |0> state
                        inverse_circuit = reset_register(circ, "sp_0")
                        circ.add(inverse_circuit.on_qubits(0))
                        circ.add(
                            gates.Unitary(
                                BO_1qb[idx_basis_ops],
                                0,
                                trainable=False,
                                name="%s" % (gatenames_1qb[idx_basis_ops]),
                            )
                        )

                    # ================= START OF 4 MEASUREMENT BASES =================|||
                    column_of_js = np.zeros((4, 1))
                    for j in range(0, 4):
                        new_circ = measurement_basis(j, circ)
                        if noise_model is not None and backend.name != "qibolab":
                            new_circ = noise_model.apply(new_circ)

                        expectation_val = GST_execute_circuit(
                            new_circ, k, j, nshots, backend=backend
                        )
                        column_of_js[j, :] = expectation_val

                    Bjk_tilde_1qb[idx_basis_ops, :, k] = column_of_js.reshape(
                        4,
                    )
                idx_basis_ops += 1

            return Bjk_tilde_1qb

        elif nqubits == 2:
            Bjk_tilde_2qb = np.zeros((241, 16, 16))

            idx_basis_ops = 0
            for idx_gatenames_1 in range(0, 13):
                for idx_gatenames_2 in range(0, 13):
                    # 169 basis operations
                    for k in range(0, 16):
                        # Initial state
                        circ = prepare_states(k, nqubits)

                        # 169 Basis operations
                        if (
                            idx_gatenames_1 < 10 and idx_gatenames_2 < 10
                        ):  ### NEW SCENARIO 1
                            circ.add(
                                gates.Unitary(
                                    BO_1qb[idx_gatenames_1],
                                    0,
                                    trainable=False,
                                    name="%s" % (gatenames_2qb[idx_basis_ops]),
                                )
                            )
                            circ.add(
                                gates.Unitary(
                                    BO_1qb[idx_gatenames_2],
                                    1,
                                    trainable=False,
                                    name="%s" % (gatenames_2qb[idx_basis_ops]),
                                )
                            )

                        ### PARTIALLY REPREPARE QUBIT_1
                        elif idx_gatenames_1 == 10 and idx_gatenames_2 < 10:
                            inverse_circuit = reset_register(circ, "sp_0")
                            circ.add(inverse_circuit.on_qubits(0))
                            circ.add(
                                gates.Unitary(
                                    BO_1qb[idx_gatenames_1],
                                    0,
                                    trainable=False,
                                    name="%s" % (gatenames_2qb[idx_basis_ops]),
                                )
                            )
                            circ.add(
                                gates.Unitary(
                                    BO_1qb[idx_gatenames_2],
                                    1,
                                    trainable=False,
                                    name="%s" % (gatenames_2qb[idx_basis_ops]),
                                )
                            )

                        elif idx_gatenames_1 == 11 and idx_gatenames_2 < 10:
                            inverse_circuit = reset_register(circ, "sp_0")
                            circ.add(inverse_circuit.on_qubits(0))
                            circ.add(
                                gates.Unitary(
                                    BO_1qb[idx_gatenames_1],
                                    0,
                                    trainable=False,
                                    name="%s" % (gatenames_2qb[idx_basis_ops]),
                                )
                            )
                            circ.add(
                                gates.Unitary(
                                    BO_1qb[idx_gatenames_2],
                                    1,
                                    trainable=False,
                                    name="%s" % (gatenames_2qb[idx_basis_ops]),
                                )
                            )

                        elif idx_gatenames_1 == 12 and idx_gatenames_2 < 10:
                            inverse_circuit = reset_register(circ, "sp_0")
                            circ.add(inverse_circuit.on_qubits(0))
                            circ.add(
                                gates.Unitary(
                                    BO_1qb[idx_gatenames_1],
                                    0,
                                    trainable=False,
                                    name="%s" % (gatenames_2qb[idx_basis_ops]),
                                )
                            )
                            circ.add(
                                gates.Unitary(
                                    BO_1qb[idx_gatenames_2],
                                    1,
                                    trainable=False,
                                    name="%s" % (gatenames_2qb[idx_basis_ops]),
                                )
                            )

                        ### PARTIALLY REPREPARE QUBIT_2
                        elif idx_gatenames_1 < 10 and idx_gatenames_2 == 10:
                            circ.add(
                                gates.Unitary(
                                    BO_1qb[idx_gatenames_1],
                                    0,
                                    trainable=False,
                                    name="%s" % (gatenames_2qb[idx_basis_ops]),
                                )
                            )
                            inverse_circuit = reset_register(circ, "sp_1")
                            circ.add(inverse_circuit.on_qubits(1))
                            circ.add(
                                gates.Unitary(
                                    BO_1qb[idx_gatenames_2],
                                    1,
                                    trainable=False,
                                    name="%s" % (gatenames_2qb[idx_basis_ops]),
                                )
                            )

                        elif idx_gatenames_1 < 10 and idx_gatenames_2 == 11:
                            circ.add(
                                gates.Unitary(
                                    BO_1qb[idx_gatenames_1],
                                    0,
                                    trainable=False,
                                    name="%s" % (gatenames_2qb[idx_basis_ops]),
                                )
                            )
                            inverse_circuit = reset_register(circ, "sp_1")
                            circ.add(inverse_circuit.on_qubits(1))
                            circ.add(
                                gates.Unitary(
                                    BO_1qb[idx_gatenames_2],
                                    1,
                                    trainable=False,
                                    name="%s" % (gatenames_2qb[idx_basis_ops]),
                                )
                            )

                        elif idx_gatenames_1 < 10 and idx_gatenames_2 == 12:
                            circ.add(
                                gates.Unitary(
                                    BO_1qb[idx_gatenames_1],
                                    0,
                                    trainable=False,
                                    name="%s" % (gatenames_2qb[idx_basis_ops]),
                                )
                            )
                            inverse_circuit = reset_register(circ, "sp_1")
                            circ.add(inverse_circuit.on_qubits(1))
                            circ.add(
                                gates.Unitary(
                                    BO_1qb[idx_gatenames_2],
                                    1,
                                    trainable=False,
                                    name="%s" % (gatenames_2qb[idx_basis_ops]),
                                )
                            )

                        ### FULLY REPREPARE QUBIT_1 AND QUBIT_2
                        elif idx_gatenames_1 == 10 and idx_gatenames_2 == 10:
                            inverse_circuit = reset_register(circ, "sp_t")
                            circ.add(inverse_circuit.on_qubits(0, 1))
                            circ.add(
                                gates.Unitary(
                                    BO_1qb[idx_gatenames_1],
                                    0,
                                    trainable=False,
                                    name="%s" % (gatenames_2qb[idx_basis_ops]),
                                )
                            )
                            circ.add(
                                gates.Unitary(
                                    BO_1qb[idx_gatenames_2],
                                    1,
                                    trainable=False,
                                    name="%s" % (gatenames_2qb[idx_basis_ops]),
                                )
                            )

                        elif idx_gatenames_1 == 10 and idx_gatenames_2 == 11:
                            inverse_circuit = reset_register(circ, "sp_t")
                            circ.add(inverse_circuit.on_qubits(0, 1))
                            circ.add(
                                gates.Unitary(
                                    BO_1qb[idx_gatenames_1],
                                    0,
                                    trainable=False,
                                    name="%s" % (gatenames_2qb[idx_basis_ops]),
                                )
                            )
                            circ.add(
                                gates.Unitary(
                                    BO_1qb[idx_gatenames_2],
                                    1,
                                    trainable=False,
                                    name="%s" % (gatenames_2qb[idx_basis_ops]),
                                )
                            )

                        elif idx_gatenames_1 == 10 and idx_gatenames_2 == 12:
                            inverse_circuit = reset_register(circ, "sp_t")
                            circ.add(inverse_circuit.on_qubits(0, 1))
                            circ.add(
                                gates.Unitary(
                                    BO_1qb[idx_gatenames_1],
                                    0,
                                    trainable=False,
                                    name="%s" % (gatenames_2qb[idx_basis_ops]),
                                )
                            )
                            circ.add(
                                gates.Unitary(
                                    BO_1qb[idx_gatenames_2],
                                    1,
                                    trainable=False,
                                    name="%s" % (gatenames_2qb[idx_basis_ops]),
                                )
                            )

                        # ==================================================================================
                        elif idx_gatenames_1 == 11 and idx_gatenames_2 == 10:
                            inverse_circuit = reset_register(circ, "sp_t")
                            circ.add(inverse_circuit.on_qubits(0, 1))
                            circ.add(
                                gates.Unitary(
                                    BO_1qb[idx_gatenames_1],
                                    0,
                                    trainable=False,
                                    name="%s" % (gatenames_2qb[idx_basis_ops]),
                                )
                            )
                            circ.add(
                                gates.Unitary(
                                    BO_1qb[idx_gatenames_2],
                                    1,
                                    trainable=False,
                                    name="%s" % (gatenames_2qb[idx_basis_ops]),
                                )
                            )

                        elif idx_gatenames_1 == 11 and idx_gatenames_2 == 11:
                            inverse_circuit = reset_register(circ, "sp_t")
                            circ.add(inverse_circuit.on_qubits(0, 1))
                            circ.add(
                                gates.Unitary(
                                    BO_1qb[idx_gatenames_1],
                                    0,
                                    trainable=False,
                                    name="%s" % (gatenames_2qb[idx_basis_ops]),
                                )
                            )
                            circ.add(
                                gates.Unitary(
                                    BO_1qb[idx_gatenames_2],
                                    1,
                                    trainable=False,
                                    name="%s" % (gatenames_2qb[idx_basis_ops]),
                                )
                            )

                        elif idx_gatenames_1 == 11 and idx_gatenames_2 == 12:
                            inverse_circuit = reset_register(circ, "sp_t")
                            circ.add(inverse_circuit.on_qubits(0, 1))
                            circ.add(
                                gates.Unitary(
                                    BO_1qb[idx_gatenames_1],
                                    0,
                                    trainable=False,
                                    name="%s" % (gatenames_2qb[idx_basis_ops]),
                                )
                            )
                            circ.add(
                                gates.Unitary(
                                    BO_1qb[idx_gatenames_2],
                                    1,
                                    trainable=False,
                                    name="%s" % (gatenames_2qb[idx_basis_ops]),
                                )
                            )

                        # ==================================================================================
                        elif idx_gatenames_1 == 12 and idx_gatenames_2 == 10:
                            inverse_circuit = reset_register(circ, "sp_t")
                            circ.add(inverse_circuit.on_qubits(0, 1))
                            circ.add(
                                gates.Unitary(
                                    BO_1qb[idx_gatenames_1],
                                    0,
                                    trainable=False,
                                    name="%s" % (gatenames_2qb[idx_basis_ops]),
                                )
                            )
                            circ.add(
                                gates.Unitary(
                                    BO_1qb[idx_gatenames_2],
                                    1,
                                    trainable=False,
                                    name="%s" % (gatenames_2qb[idx_basis_ops]),
                                )
                            )

                        elif idx_gatenames_1 == 12 and idx_gatenames_2 == 11:
                            inverse_circuit = reset_register(circ, "sp_t")
                            circ.add(inverse_circuit.on_qubits(0, 1))
                            circ.add(
                                gates.Unitary(
                                    BO_1qb[idx_gatenames_1],
                                    0,
                                    trainable=False,
                                    name="%s" % (gatenames_2qb[idx_basis_ops]),
                                )
                            )
                            circ.add(
                                gates.Unitary(
                                    BO_1qb[idx_gatenames_2],
                                    1,
                                    trainable=False,
                                    name="%s" % (gatenames_2qb[idx_basis_ops]),
                                )
                            )

                        elif idx_gatenames_1 == 12 and idx_gatenames_2 == 12:
                            inverse_circuit = reset_register(circ, "sp_t")
                            circ.add(inverse_circuit.on_qubits(0, 1))
                            circ.add(
                                gates.Unitary(
                                    BO_1qb[idx_gatenames_1],
                                    0,
                                    trainable=False,
                                    name="%s" % (gatenames_2qb[idx_basis_ops]),
                                )
                            )
                            circ.add(
                                gates.Unitary(
                                    BO_1qb[idx_gatenames_2],
                                    1,
                                    trainable=False,
                                    name="%s" % (gatenames_2qb[idx_basis_ops]),
                                )
                            )

                        column_of_js = np.zeros((4**nqubits, 1))
                        # ==================================================================================

                        #     # Measurements
                        column_of_js = np.zeros((4**nqubits, 1))
                        for j in range(0, 16):
                            new_circ = measurement_basis(j, circ)

                            if noise_model is not None and backend.name != "qibolab":
                                new_circ = noise_model.apply(new_circ)

                            expectation_val = GST_execute_circuit(
                                new_circ, k, j, nshots, backend=backend
                            )
                            column_of_js[j, :] = expectation_val

                        Bjk_tilde_2qb[idx_basis_ops, :, k] = column_of_js.reshape(
                            16,
                        )

                    idx_basis_ops += 1

            idx_basis_ops = 169
            for idx_basis_ops in range(169, 241):
                for k in range(0, 16):
                    # Initial states
                    circ = prepare_states(k, nqubits)

                    # 72 Basis operations
                    circ.add(
                        gates.Unitary(
                            BO_2qb[idx_basis_ops],
                            0,
                            1,
                            trainable=False,
                            name="%s" % (gatenames_2qb[idx_basis_ops]),
                        )
                    )

                    # Measurements
                    for j in range(0, 16):
                        new_circ = measurement_basis(j, circ)
                        expectation_val = GST_execute_circuit(
                            new_circ, k, j, nshots, backend=backend
                        )
                        column_of_js[j, :] = expectation_val

                        if noise_model is not None and backend.name != "qibolab":
                            new_circ = noise_model.apply(new_circ)

                    Bjk_tilde_2qb[idx_basis_ops, :, k] = column_of_js.reshape(
                        16,
                    )

                idx_basis_ops += 1

        return Bjk_tilde_2qb
