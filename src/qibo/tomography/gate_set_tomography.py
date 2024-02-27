from itertools import product

import numpy as np

from qibo import Circuit, gates, symbols
from qibo.backends import GlobalBackend
from qibo.config import raise_error
from qibo.hamiltonians import SymbolicHamiltonian


def prepare_states(k, nqubits):
    """Prepares a quantum circuit in a specific state indexed by `k`.

        Args:
        k (int): The index of the state to be prepared.
            For a single qubit, \\(k \\in \\{0, 1, 2, 3\\} \\equiv
                \\{| 0 \rangle \\langle 0 |,
                  | 1 \rangle \\langle 1 |,
                  | + \rangle \\langle + |,
                  | y+ \rangle \\langle y+ | \\).
            For two qubits, \\(k \\in \\{0, 1, 2, 3\\}^{\\otimes 2}\\).
        nqubits (int): Number of qubits in the circuit.

    Returns:
        circuit (:class:`qibo.models.Circuit`): Circuit prepared in the specified state.
    """

    if not nqubits in (1, 2):
        raise_error(
            ValueError,
            f"nqubits given as {nqubits}. nqubits needs to be either 1 or 2.",
        )

    gates_list = [(gates.I,), (gates.X,), (gates.H,), (gates.H, gates.S)]
    k_to_gates = list(product(gates_list, repeat=nqubits))[k]
    circ = Circuit(nqubits, density_matrix=True)
    for q in range(len(k_to_gates)):
        gate_q = [gate(q) for gate in k_to_gates[q]]
        circ.add(gate_q)
    return circ


def measurement_basis(j, circ):
    r"""Implements a measurement basis for circuit indexed by `j`.

        Args:
        j (int): The index of the measurement basis.
            For a single qubit, \(j \in \{0, 1, 2, 3\} \equiv \{I, X, Y, Z\}\).
            For two qubits, \(j \in \{0, 1, 2, 3\}^{\otimes 2}\).
        circuit (:class:`qibo.models.Circuit`): Circuit without measurement basis.

    Returns:
        circuit (:class:`qibo.models.Circuit`): Circuit with measurement basis.
    """

    nqubits = circ.nqubits
    if not nqubits in (1, 2):
        raise_error(
            ValueError,
            f"nqubits given as {nqubits}. nqubits needs to be either 1 or 2.",
        )

    meas_list = [gates.Z, gates.X, gates.Y, gates.Z]
    j_to_measurements = list(product(meas_list, repeat=nqubits))[j]
    new_circ = circ.copy()
    for q in range(len(j_to_measurements)):
        meas_q = j_to_measurements[q]
        new_circ.add(gates.M(q, basis=meas_q))

    return new_circ


def reset_register(circuit, invert_register):
    """Returns an inverse circuit of the selected register to prepare the zero state \\(|0\rangle\\).
        One can then add inverse_circuit to the original circuit by addition:
            circ_with_inverse = circ.copy()
            circ_with_inverse.add(inverse_circuit.on_qubits(invert_register))
        where register_to_reset = (0,), (1,) , or (0, 1).

        Args:
        circuit (:class:`qibo.models.Circuit`): original circuit
        invert_register (tuple): Qubit(s) to reset: Use a tuple to specify which qubit(s) to reset:
            - (0,) to reset qubit 0;
            - (1,) to reset qubit 1; or
            - (0,1) to reset both qubits.
    Returns:
        inverse_circuit (:class:`qibo.models.Circuit`): Inverse of the input circuit's register.
    """
    valid_registers = [(0,), (1,), (0, 1)]
    if invert_register is not None:
        if (
            not isinstance(invert_register, tuple)
            or invert_register not in valid_registers
        ):
            raise_error(
                NameError,
                f"{invert_register} not recognized.",
            )

        elif invert_register == (0,) or invert_register == (1,):
            register_to_reset = invert_register[0]
            new_circ = Circuit(1)
            for data in circuit.raw["queue"]:
                init_kwargs = data.get("init_kwargs", {})
                if data["_target_qubits"][0] == register_to_reset:
                    new_circ.add(getattr(gates, data["_class"])(0, **init_kwargs))

        else:
            new_circ = circuit.copy()

    return new_circ.invert()


# def reset_register(circuit, invert_register):
#     """Returns an inverse circuit of the selected register to prepare the zero state \\(|0\rangle\\).
#         One can then add inverse_circuit to the original circuit by addition:
#             circ_with_inverse = circ.copy()
#             circ_with_inverse.add(inverse_circuit.on_qubits(invert_register))
#         where register_to_reset = 0, 1, or [0,1].

#         Args:
#         circuit (:class:`qibo.models.Circuit`): original circuit
#         invert_register (string): Qubit(s) to reset:
#             'sp_0' (qubit 0);
#             'sp_1' (qubit 1); or
#             'sp_t' (both qubits)
#             where 'sp' is an abbreviation for state_preparation.
#     Returns:
#         inverse_circuit (:class:`qibo.models.Circuit`): Inverse of the input circuit's register.
#     """

#     if invert_register == "sp_0" or invert_register == "sp_1":
#         if invert_register == "sp_0":
#             register_to_reset = 0
#         elif invert_register == "sp_1":
#             register_to_reset = 1

#         new_circ = Circuit(1)
#         for data in circuit.raw["queue"]:
#             init_kwargs = data.get("init_kwargs", {})
#             if data["_target_qubits"][0] == register_to_reset:
#                 new_circ.add(getattr(gates, data["_class"])(0, **init_kwargs))

#     elif invert_register == "sp_t":
#         new_circ = circuit.copy()

#     else:
#         raise_error(
#             NameError,
#             f"{invert_register} not recognized. Input "
#             "sp_0"
#             " to reset qubit 0, "
#             "sp_1"
#             " to reset qubit 1, or "
#             "sp_t"
#             " to reset both qubits.",
#         )

#     return new_circ.invert()


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
        numpy.float: Expectation value given by either :math:`\\text{tr}(Q_j rho_k) \\` or
            :math:`\\Tr(Q_j O_l rho_k) \\`.
    """

    nqubits = circuit.nqubits
    if not nqubits in (1, 2):
        raise_error(
            ValueError,
            f"nqubits given as {nqubits}. nqubits needs to be either 1 or 2.",
        )

    else:
        if j == 0:
            return 1.0
        else:
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


def execute_GST(
    nqubits=None,
    gate=None,
    nshots=int(1e4),
    invert_register=None,
    noise_model=None,
    backend=None,
):
    """Runs gate set tomography for a 1 or 2 qubit gate.

        Args:
        nshots (int, optional): Number of shots used in Gate Set Tomography.
        gate (:class:`qibo.gates.abstract.Gate`, optional): The gate to perform gate set tomography on.
            If gate=None, then gate set tomography will be performed for an empty circuit.
        noise_model (:class:`qibo.noise.NoiseModel`, optional): Noise model applied
            to simulate noisy computation.
        backend (:class:`qibo.backends.abstract.Backend`, optional): Calculation engine.
    Returns:
        ndarray: array with elements ``jk`` equivalent to either :math:`\\text{tr}(Q_{j} \\, \\rho_{k})`
            or :math:`\\text{tr}(Q_{j} \\, O_{l} \\rho_{k})` where :math:`O_{l}` is the l-th operation
            in the original circuit.
    """

    # Check if gate is 1 or 2 qubit gate.
    if not nqubits in (1, 2):
        raise_error(
            ValueError,
            f"nqubits given as {nqubits}. nqubits needs to be either 1 or 2.",
        )

    # Check if invert_register has the correct register(s).
    valid_registers = [(0,), (1,), (0, 1)]
    if invert_register is not None:
        if (
            not isinstance(invert_register, tuple)
            or invert_register not in valid_registers
        ):
            raise_error(
                NameError,
                f"{invert_register} not recognized.",
            )

    # # Check if invert_register has the correct string.
    # if invert_register is not None:
    #     if (
    #         invert_register != "sp_0"
    #         and invert_register != "sp_1"
    #         and invert_register != "sp_t"
    #     ):
    #         raise_error(
    #             NameError,
    #             f"{invert_register} not recognized. Input "
    #             "sp_0"
    #             " to reset qubit 0, "
    #             "sp_1"
    #             " to reset qubit 1, or "
    #             "sp_t"
    #             " to reset both qubits.",
    #         )

    if backend is None:  # pragma: no cover
        backend = GlobalBackend()

    if gate is not None:
        if nqubits != len(gate.qubits):
            raise_error(
                ValueError,
                f"Mismatched inputs: nqubits given as {nqubits}. {gate} is a {len(gate.qubits)}-qubit gate.",
            )
        gate = gate.__class__(*gate.qubits, **gate.init_kwargs)

    # GST for empty circuit or with gates
    matrix_jk = np.zeros((4**nqubits, 4**nqubits))
    for k in range(4**nqubits):
        circ = prepare_states(k, nqubits)
        if invert_register is not None:
            inverted_circuit = reset_register(circ, invert_register)
            # if invert_register == "sp_0":
            #     circ.add(inverted_circuit.on_qubits(0))
            # elif invert_register == "sp_1":
            #     circ.add(inverted_circuit.on_qubits(1))
            # elif invert_register == "sp_t":
            #     circ.add(inverted_circuit.on_qubits(0, 1))
            circ.add(inverted_circuit.on_qubits(*invert_register))

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
