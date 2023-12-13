import os
import time

import numpy as np

from qibo import gates
from qibo.backends import GlobalBackend
from qibo.models import Circuit


def prepare_1qb_state_new(k):
    nqubits = 1

    k_to_gates_1qb = {
        0: (),
        1: (gates.X(0),),
        2: (gates.H(0),),
        3: (gates.H(0), gates.S(0)),
    }

    circ = Circuit(nqubits, density_matrix=True)
    for gate in k_to_gates_1qb[k]:
        circ.add(gate)
    return circ


def prepare_2qb_state_new(k):
    nqubits = 2

    k_to_gates_2qb = {
        0: (),
        1: (gates.X(1),),
        2: (gates.H(1),),
        3: (gates.H(1), gates.S(1)),
        4: (gates.X(0),),
        5: (
            gates.X(0),
            gates.X(1),
        ),
        6: (
            gates.X(0),
            gates.H(1),
        ),
        7: (gates.X(0), gates.H(1), gates.S(1)),
        8: (gates.H(0),),
        9: (
            gates.H(0),
            gates.X(1),
        ),
        10: (
            gates.H(0),
            gates.H(1),
        ),
        11: (gates.H(0), gates.H(1), gates.S(1)),
        12: (
            gates.H(0),
            gates.S(0),
        ),
        13: (
            gates.H(0),
            gates.S(0),
            gates.X(1),
        ),
        14: (
            gates.H(0),
            gates.S(0),
            gates.H(1),
        ),
        15: (gates.H(0), gates.S(0), gates.H(1), gates.S(1)),
    }

    circ = Circuit(nqubits, density_matrix=True)
    for gate in k_to_gates_2qb[k]:
        circ.add(gate)
    return circ


def measure_1qb_state_new(j, circ, noise_model=None):
    nqubits = 1

    j_to_measurements_1qb = {
        0: (gates.M(0, basis=gates.Z),),
        1: (gates.M(0, basis=gates.X),),
        2: (gates.M(0, basis=gates.Y),),
        3: (gates.M(0, basis=gates.Z),),
    }

    new_circ = circ.copy()

    for meas in j_to_measurements_1qb[j]:
        new_circ.add(meas)

    if noise_model is not None and backend.name != "qibolab":
        new_circ = noise_model.apply(new_circ)

    return new_circ


def measure_2qb_state_new(j, circ, noise_model=None):
    nqubits = 2

    j_to_measurements_2qb = {
        0: (
            gates.M(0, basis=gates.Z),
            gates.M(1, basis=gates.Z),
        ),
        1: (
            gates.M(0, basis=gates.Z),
            gates.M(1, basis=gates.X),
        ),
        2: (
            gates.M(0, basis=gates.Z),
            gates.M(1, basis=gates.Y),
        ),
        3: (
            gates.M(0, basis=gates.Z),
            gates.M(1, basis=gates.Z),
        ),
        4: (
            gates.M(0, basis=gates.X),
            gates.M(1, basis=gates.Z),
        ),
        5: (
            gates.M(0, basis=gates.X),
            gates.M(1, basis=gates.X),
        ),
        6: (
            gates.M(0, basis=gates.X),
            gates.M(1, basis=gates.Y),
        ),
        7: (
            gates.M(0, basis=gates.X),
            gates.M(1, basis=gates.Z),
        ),
        8: (
            gates.M(0, basis=gates.Y),
            gates.M(1, basis=gates.Z),
        ),
        9: (
            gates.M(0, basis=gates.Y),
            gates.M(1, basis=gates.X),
        ),
        10: (
            gates.M(0, basis=gates.Y),
            gates.M(1, basis=gates.Y),
        ),
        11: (
            gates.M(0, basis=gates.Y),
            gates.M(1, basis=gates.Z),
        ),
        12: (
            gates.M(0, basis=gates.Z),
            gates.M(1, basis=gates.Z),
        ),
        13: (
            gates.M(0, basis=gates.Z),
            gates.M(1, basis=gates.X),
        ),
        14: (
            gates.M(0, basis=gates.Z),
            gates.M(1, basis=gates.Y),
        ),
        15: (
            gates.M(0, basis=gates.Z),
            gates.M(1, basis=gates.Z),
        ),
    }

    new_circ = circ.copy()

    for meas in j_to_measurements_2qb[j]:
        new_circ.add(meas)

    if noise_model is not None and backend.name != "qibolab":
        new_circ = noise_model.apply(new_circ)

    return new_circ


def sort_counts(matrix):
    """Sorts input matrix according to resultant states.
        e.g. input = np.array([[0, 439], [11, 3023], [10, 1942], [1,  830]])
            output = np.array([[0, 439], [1,  830],  [19, 1942], [11, 3023]])

        Args:
        matrix (numpy.array): Array with first column as the resultant states
        and second column as the number of measurement outcomes (int).

    Returns:
        numpy.matrix: Array with resultant states sorted.

    """
    counts_matrix = np.zeros((matrix.shape[0], matrix.shape[1]))

    for ii in range(0, matrix.shape[0]):
        word = bin(ii)[2:]
        counts_matrix[ii, 0] = word

    for row in matrix:
        row_number = int(str(int(row[0])), 2)
        counts_matrix[row_number, 1] = row[1]

    return counts_matrix


def GST_measure(
    circuit,
    k,
    j,
    nshots,
    nqubits=None,
    class_qubit_gate=None,
    theta=None,
    idx_basis_ops=None,
    save_data=None,
):
    if nqubits == 1:
        idx_basis = ["I", "X", "Y", "Z"]
        expectation_signs = np.array([[1, 1, 1, 1], [1, -1, -1, -1]])
        matrix = np.array([[0, 0], [1, 0]])

    elif nqubits == 2:
        idx_basis = [
            "II",
            "IX",
            "IY",
            "IZ",
            "XI",
            "XX",
            "XY",
            "XZ",
            "YI",
            "YX",
            "YY",
            "YZ",
            "ZI",
            "ZX",
            "ZY",
            "ZZ",
        ]
        expectation_signs = np.array(
            [[1, 1, 1, 1], [1, -1, 1, -1], [1, 1, -1, -1], [1, -1, -1, 1]]
        )
        matrix = np.array([[0, 0], [1, 0], [10, 0], [11, 0]])

    else:
        raise ValueError(
            f"nqubits given as {nqubits}. nqubits needs to be either 1 or 2."
        )

    result = circuit.execute(nshots=nshots)
    counts = result.frequencies(binary=True)

    if save_data is not None:
        if class_qubit_gate is None and idx_basis_ops is None:
            filedirectory = "GST_nogates_qibo_%.1dqb/" % (nqubits)
            fname = "No gates (k=%.2d,j=%.2d) %s_basis.txt" % (k, j, idx_basis[j])
            folder_name = "GST_nogates_qibo_%1.dqb" % (nqubits)

        elif class_qubit_gate is not None and idx_basis_ops is None:
            filedirectory = "GST_gates_qibo_%.1dqb/" % (nqubits)
            if theta is None:
                fname = "%s gate (k=%.2d,j=%.2d) %s_basis.txt" % (
                    class_qubit_gate,
                    k,
                    j,
                    idx_basis[j],
                )
            elif theta is not None:
                fname = "%s(%.12f) gate (k=%.2d,j=%.2d) %s_basis.txt" % (
                    class_qubit_gate,
                    theta,
                    k,
                    j,
                    idx_basis[j],
                )
            folder_name = "GST_gates_qibo_%.1dqb" % (nqubits)

        elif class_qubit_gate is None and idx_basis_ops is not None:
            filedirectory = "GST_basisops_qibo_%.1dqb/" % (nqubits)
            if nqubits == 1:
                fname = "BasisOp%.2d (k=%.2d,j=%.2d) %s_basis.txt" % (
                    idx_basis_ops,
                    k,
                    j,
                    idx_basis[j],
                )
            elif nqubits == 2:
                fname = "BasisOp%.3d (k=%.2d,j=%.2d) %s_basis.txt" % (
                    idx_basis_ops,
                    k,
                    j,
                    idx_basis[j],
                )
            folder_name = f"GST_basisops_qibo_%.1dqb" % (nqubits)

        elif class_qubit_gate is not None and idx_basis_ops is not None:
            raise ValueError(
                "Both 'class_qubit_gate' and 'idx_basis_ops' cannot be specified at the same time."
            )

        if not os.path.exists(folder_name):
            os.makedirs(folder_name)

        # Save counts into a matrix. 0th column: word; 1st column: count value
        # --------------------------------------------------------------------
        fout = filedirectory + fname
        fo = open(fout, "w")
        for v1, v2 in counts.items():
            fo.write(str(v1) + "  " + str(v2) + "\n")
        fo.close()

    # Find value for matrix_jk
    # -------------------------

    for v1, v2 in counts.items():
        # print(v1, v2)
        row = int(v1, 2)
        col = v2
        matrix[row, 1] = col

    sorted_matrix = sort_counts(matrix)
    probs = sorted_matrix[:, 1] / np.sum(sorted_matrix[:, 1])
    sorted_matrix = np.hstack((sorted_matrix, probs.reshape(-1, 1)))

    k = k + 1
    j = j + 1

    if j == 1:
        sorted_matrix = np.hstack(
            (sorted_matrix, expectation_signs[:, 0].reshape(-1, 1))
        )
    elif j == 2 or j == 3 or j == 4:
        sorted_matrix = np.hstack(
            (sorted_matrix, expectation_signs[:, 1].reshape(-1, 1))
        )
    elif j == 5 or j == 9 or j == 13:
        sorted_matrix = np.hstack(
            (sorted_matrix, expectation_signs[:, 2].reshape(-1, 1))
        )
    else:
        sorted_matrix = np.hstack(
            (sorted_matrix, expectation_signs[:, 3].reshape(-1, 1))
        )

    temp = np.array(sorted_matrix[:, 2]) * np.array(sorted_matrix[:, 3])
    sorted_matrix = np.hstack((sorted_matrix, temp.reshape(-1, 1)))

    expectation_val = np.sum(temp)

    return expectation_val


def GST_1qb(
    nshots=int(1e4),
    noise_model=None,
    class_qubit_gate=None,
    theta=None,
    operator_name=None,
    backend=None,
    save_data=None,
):
    """Runs gate set tomography for a single qubit gate/circuit.

        Args:
        nshots (int, optional): Number of shots used in Gate Set Tomography.
        noise_model (:class:`qibo.noise.NoiseModel`, optional): Noise model applied
            to simulate noisy computation.
        save_data: To save gate set tomography data. If None, skip.
        class_qubit_gate: Identifier for the type of gate given in qibo.raw. If None,
            do gate set tomography with empty circuit.
        theta: Identifier for the gate parameter given in qibo.raw. If None, do gate
            set tomography with only gate provided by class_qubit_gate.
        operator_name: Name of gate given in qibo.raw used for saving data.
        backend (:class:`qibo.backends.abstract.Backend`, optional): Calculation engine.

    Returns:
        numpy.matrix: Matrix with elements jk equivalent to either Tr(Q_j rho_k) or Tr(Q_j O_l rho_k).
    """

    if backend is None:  # pragma: no cover
        backend = GlobalBackend()

    # Keep for testing, remove when push.
    if class_qubit_gate is None:
        print("Empty circuit without any operator (1 qb)")
    elif class_qubit_gate is not None:
        if theta is None:
            print(f"Gate set tomography for 1 qubit gate: {class_qubit_gate}.")
        elif theta is not None:
            print(f"Gate set tomography for 1 qubit gate: {class_qubit_gate}({theta}).")

    nqubits = 1
    idx_basis_ops = None
    matrix_jk_1qb = np.zeros((4**nqubits, 4**nqubits))
    for k in range(0, 4):
        # Prepare state using "prepare_1qb_state(k)"
        circ = prepare_1qb_state_new(k)

        # If class_qubit_gate is present (with/without theta)
        if class_qubit_gate is not None:
            if theta is None:
                circ.add(getattr(gates, class_qubit_gate)(0))
            elif theta is not None:
                circ.add(getattr(gates, class_qubit_gate)(0, theta))

        # 4 measurements
        for j in range(0, 4):
            new_circ = measure_1qb_state_new(j, circ)
            # expectation_val = GST_measure_1qb(new_circ, k, j, nshots)
            expectation_val = GST_measure(
                new_circ,
                k,
                j,
                nshots,
                nqubits,
                class_qubit_gate,
                theta,
                idx_basis_ops,
                save_data,
            )
            matrix_jk_1qb[j, k] = expectation_val

    return matrix_jk_1qb


def GST_2qb(
    nshots=int(1e4),
    noise_model=None,
    class_qubit_gate=None,
    ctrl_qb=None,
    targ_qb=None,
    theta=None,
    operator_name=None,
    backend=None,
    save_data=None,
):
    """Runs gate set tomography for a two qubit gate/circuit.

        Args:
        nshots (int, optional): Number of shots used in Gate Set Tomography.
        noise_model (:class:`qibo.noise.NoiseModel`, optional): Noise model applied
            to simulate noisy computation.
        save_data: To save gate set tomography data. If None, skip.
        class_qubit_gate: Identifier for the type of gate given in qibo.raw. If None,
            do gate set tomography with empty circuit.
        ctrl_qb: Parameter for the control qubit.
        targ_qb: Parameter for the target qubit.
        theta: Identifier for the gate parameter given in qibo.raw. If None, do gate
            set tomography with only gate provided by class_qubit_gate.
        operator_name: Name of gate given in qibo.raw used for saving data.
        backend (:class:`qibo.backends.abstract.Backend`, optional): Calculation engine.

    Returns:
        numpy.matrix: Matrix with elements jk equivalent to either Tr(Q_j rho_k) or Tr(Q_j O_l rho_k).
    """

    if backend is None:  # pragma: no cover
        backend = GlobalBackend()

    # Keep for testing, remove when push.
    if class_qubit_gate is None:
        print("Empty circuit without any operator (2 qb)")
    elif class_qubit_gate is not None:
        if theta is None:
            print(f"Gate set tomography for 2 qubit gate: {class_qubit_gate}.")
        elif theta is not None:
            print(f"Gate set tomography for 2 qubit gate: {class_qubit_gate}({theta}).")

    nqubits = 2
    idx_basis_ops = None
    matrix_jk_2qb = np.zeros((4**nqubits, 4**nqubits))
    for k in range(0, 16):
        # Prepare states using function "prepare_2qb_states(k)".
        circ = prepare_2qb_state_new(k)

        # If class_qubit_gate is present (with/without theta).
        if class_qubit_gate is not None:
            if theta is None:
                circ.add(getattr(gates, class_qubit_gate)(targ_qb[0], ctrl_qb[0]))
            elif theta is not None:
                circ.add(
                    getattr(gates, class_qubit_gate)(targ_qb[0], ctrl_qb[0], theta)
                )

        # Measure states. Measurement basis using function "measure_2qb_states(j, circ, noise_model)".
        for j in range(0, 16):
            new_circ = measure_2qb_state_new(j, circ)
            # expectation_val = GST_measure_2qb(new_circ, k, j, nshots)
            expectation_val = GST_measure(
                new_circ,
                k,
                j,
                nshots,
                nqubits,
                class_qubit_gate,
                theta,
                idx_basis_ops,
                save_data,
            )
            matrix_jk_2qb[j, k] = expectation_val

    return matrix_jk_2qb


def GST_1qb_basis_operations(
    nshots=int(1e4), noise_model=None, gjk_1qb=None, save_data=None, backend=None
):
    """Runs gate set tomography for the 13 basis operations for a single qubit.

        Args:
        nshots (int, optional): Number of shots used in Gate Set Tomography.
        noise_model (:class:`qibo.noise.NoiseModel`, optional): Noise model applied
            to simulate noisy computation.
        gjk_1qb (numpy.matrix): Matrix with elements Tr(Q_j rho_k) for one qubit with no gate.
            If None, call GST_1qb without gate.
        save_data: Flag to save gate set tomography data. If None, skip.
        backend (:class:`qibo.backends.abstract.Backend`, optional): Calculation engine.

        The 13 basis operations are based on Physical Review Research 3, 033178 (2021).
    Returns:
        numpy.matrix: Matrix with elements jk equivalent to either Tr(Q_j rho_k) or Tr(Q_j O_l rho_k).
    """

    if backend is None:  # pragma: no cover
        backend = GlobalBackend()

    if gjk_1qb is None:
        gjk_1qb = GST_1qb(nshots, noise_model, save_data=None)

    nqubits = 1
    class_qubit_gate = None
    theta = None

    BasisOps_13 = [
        gates.I(0).matrix(backend=GlobalBackend()),
        gates.X(0).matrix(backend=GlobalBackend()),
        gates.Y(0).matrix(backend=GlobalBackend()),
        gates.Z(0).matrix(backend=GlobalBackend()),
        1 / np.sqrt(2) * np.array([[1, 1j], [1j, 1]]),
        1 / np.sqrt(2) * np.array([[1, 1], [-1, 1]]),
        1 / np.sqrt(2) * np.array([[(1 + 1j), 0], [0, (1 - 1j)]]),
        1 / np.sqrt(2) * np.array([[1, -1j], [1j, -1]]),
        gates.H(0).matrix(backend=GlobalBackend()),
        1 / np.sqrt(2) * np.array([[0, (1 - 1j)], [(1 + 1j), 0]]),
        np.array([[1, 0], [0, 0]]),
        np.array([[0.5, 0.5], [0.5, 0.5]]),
        np.array([[0.5, -0.5j], [0.5j, 0.5]]),
    ]

    ###################################
    ### 13 scenarios # (2023-11-06) ###
    ###################################

    gatenames = (
        "BasisOp00",
        "BasisOp01",
        "BasisOp02",
        "BasisOp03",
        "BasisOp04",
        "BasisOp05",
        "BasisOp06",
        "BasisOp07",
        "BasisOp08",
        "BasisOp09",
        "BasisOp10",
        "BasisOp11",
        "BasisOp12",
    )
    import time

    tic = time.time()
    idx_basis_ops = 0

    Bjk_tilde_1qb = np.zeros((13, 4, 4))

    for idx_basis_ops in range(0, 13):
        print("GST 1qb 13 Basis Op. Basis Op idx = %d" % (idx_basis_ops))

        # ===============#
        # 13 SCENARIOS  # -- BIG ENDIAN NOTATION FOR THE 13 BASIS GATES TENSORED PRODUCTS
        # ===============#

        # Initialise 4 different initial states
        for k in range(0, 4):
            circ = prepare_1qb_state_new(k)

            # ==================================================================================
            # Basis operation 0 to 9
            if idx_basis_ops < 10:
                circ.add(
                    gates.Unitary(
                        BasisOps_13[idx_basis_ops],
                        0,
                        trainable=False,
                        name="%s" % (gatenames[idx_basis_ops]),
                    )
                )

            # ==================================================================================
            # Basis operation 10, 11, 12
            elif idx_basis_ops == 10:  # Reset, prepare |+> state
                circ.add(
                    gates.ResetChannel(0, [1, 0])
                )  # ResetChannel(qubit, [prob0, prob1])
                circ.add(gates.H(0))
                circ.add(
                    gates.Unitary(
                        BasisOps_13[idx_basis_ops],
                        0,
                        trainable=False,
                        name="%s" % (gatenames[idx_basis_ops]),
                    )
                )

            elif idx_basis_ops == 11:  # Reset, prepare |y+> state
                circ.add(
                    gates.ResetChannel(0, [1, 0])
                )  # ResetChannel(qubit, [prob0, prob1])
                circ.add(gates.H(0))
                circ.add(gates.S(0))
                circ.add(
                    gates.Unitary(
                        BasisOps_13[idx_basis_ops],
                        0,
                        trainable=False,
                        name="%s" % (gatenames[idx_basis_ops]),
                    )
                )

            elif idx_basis_ops == 12:  # Reset, prepare |0> state
                circ.add(
                    gates.ResetChannel(0, [1, 0])
                )  # ResetChannel(qubit, [prob0, prob1])
                circ.add(
                    gates.Unitary(
                        BasisOps_13[idx_basis_ops],
                        0,
                        trainable=False,
                        name="%s" % (gatenames[idx_basis_ops]),
                    )
                )

            column_of_js = np.zeros((4**nqubits, 1))

            # ================= START OF 4 MEASUREMENT BASES =================|||
            for j in range(0, 4**nqubits):
                new_circ = measure_1qb_state_new(j, circ)
                # expectation_val = GST_measure_13(new_circ, idx_basis_ops, k, j, nshots, save_data)
                expectation_val = GST_measure(
                    new_circ,
                    k,
                    j,
                    nshots,
                    nqubits,
                    class_qubit_gate,
                    theta,
                    idx_basis_ops,
                    save_data,
                )
                column_of_js[j, :] = expectation_val

            Bjk_tilde_1qb[idx_basis_ops, :, k] = column_of_js.reshape(
                4,
            )
        idx_basis_ops += 1

    toc = time.time() - tic
    print("13 basis operations tomography done. %.4f seconds\n" % (toc))

    #######################################################################
    ### Get noisy Pauli Transfer Matrix for each of 13 basis operations ###
    #######################################################################

    # T = np.matrix([[1,1,1,1],[0,0,1,0],[0,0,0,1],[1,-1,0,0]])
    T = np.array([[1, 1, 1, 1], [0, 0, 1, 0], [0, 0, 0, 1], [1, -1, 0, 0]])
    Bjk_hat_1qb = np.zeros((13, 4**nqubits, 4**nqubits))
    for idx_BO in range(0, 13):
        Bjk_hat_1qb[idx_BO, :, :] = (
            T @ np.linalg.inv(gjk_1qb) @ Bjk_tilde_1qb[idx_BO, :, :] @ np.linalg.inv(T)
        )

    # Reshape
    Bjk_hat_1qb_reshaped = np.zeros((16, 13))
    for idx_basis_ops in range(0, 13):
        temp = np.reshape(
            Bjk_hat_1qb[idx_basis_ops, :, :], [16, 1], order="F"
        )  # 'F' so that it stacks columns.
        Bjk_hat_1qb_reshaped[:, idx_basis_ops] = temp[:, 0].flatten()

    return Bjk_hat_1qb, Bjk_hat_1qb_reshaped, BasisOps_13


def GST_2qb_basis_operations(
    nshots=int(1e4), noise_model=None, gjk_2qb=None, save_data=None, backend=None
):
    """Runs gate set tomography for the 13 basis operations for a single qubit.

        Args:
        nshots (int, optional): Number of shots used in Gate Set Tomography.
        noise_model (:class:`qibo.noise.NoiseModel`, optional): Noise model applied
            to simulate noisy computation.
        gjk_2qb (numpy.matrix): Matrix with elements Tr(Q_j rho_k) for two qubits with no gate.
            If None, call GST_2qb without gate.
        save_data: Flag to save gate set tomography data. If None, skip.
        backend (:class:`qibo.backends.abstract.Backend`, optional): Calculation engine.

        The 241 basis operations are based on Physical Review Research 3, 033178 (2021).
    Returns:
        numpy.matrix: Matrix with elements jk equivalent to either Tr(Q_j rho_k) or Tr(Q_j O_l rho_k).
    """

    if backend is None:  # pragma: no cover
        backend = GlobalBackend()

    if gjk_2qb is None:
        gjk_2qb = GST_2qb(nshots, noise_model, save_data=None)

    nqubits = 2
    class_qubit_gate = None
    theta = None

    hadamard = gates.H(0).matrix(backend=GlobalBackend())
    identity = gates.I(0).matrix(backend=GlobalBackend())
    xgate = gates.X(0).matrix(backend=GlobalBackend())
    sgate = gates.S(0).matrix(backend=GlobalBackend())
    sdggate = gates.SDG(0).matrix(backend=GlobalBackend())
    CNOT = gates.CNOT(0, 1).matrix(backend=GlobalBackend())
    SWAP = gates.SWAP(0, 1).matrix(backend=GlobalBackend())
    ry_piby4 = gates.RY(0, np.pi / 4).matrix(backend=GlobalBackend())

    # One qubit basis operations
    BO_1qubit = []
    BO_1qubit.append(gates.I(0).matrix(backend=GlobalBackend()))
    BO_1qubit.append(gates.X(0).matrix(backend=GlobalBackend()))
    BO_1qubit.append(gates.Y(0).matrix(backend=GlobalBackend()))
    BO_1qubit.append(gates.Z(0).matrix(backend=GlobalBackend()))
    BO_1qubit.append((1 / np.sqrt(2)) * np.array([[1, 1j], [1j, 1]]))
    BO_1qubit.append((1 / np.sqrt(2)) * np.array([[1, 1], [-1, 1]]))
    BO_1qubit.append((1 / np.sqrt(2)) * np.array([[(1 + 1j), 0], [0, (1 - 1j)]]))
    BO_1qubit.append((1 / np.sqrt(2)) * np.array([[1, -1j], [1j, -1]]))
    BO_1qubit.append(gates.H(0).matrix(backend=GlobalBackend()))
    BO_1qubit.append((1 / np.sqrt(2)) * np.array([[0, (1 - 1j)], [(1 + 1j), 0]]))
    BO_1qubit.append(identity)
    BO_1qubit.append(identity)
    BO_1qubit.append(identity)

    # Intermediate operations: "interm_ops"
    interm_ops = []
    interm_ops.append(CNOT)
    interm_ops.append(np.kron(xgate, identity) @ CNOT @ np.kron(xgate, identity))
    interm_ops.append(
        np.matrix(np.block([[identity, np.zeros((2, 2))], [np.zeros((2, 2)), sgate]]))
    )
    interm_ops.append(
        np.matrix(
            np.block([[identity, np.zeros((2, 2))], [np.zeros((2, 2)), hadamard]])
        )
    )
    interm_ops.append(np.kron(ry_piby4, identity) @ CNOT @ np.kron(ry_piby4, identity))
    interm_ops.append(CNOT @ np.kron(hadamard, identity))
    interm_ops.append(SWAP)
    interm_ops.append(
        np.matrix([[1, 0, 0, 0], [0, 0, 1j, 0], [0, 1j, 0, 0], [0, 0, 0, 1]])
    )
    interm_ops.append(SWAP @ np.kron(hadamard, identity))

    # Two qubit basis operations
    # "Easy"
    BO_2qubits_easy = []
    for ii in range(0, 13):
        for jj in range(0, 13):
            temp_matrix = np.kron(BO_1qubit[ii], BO_1qubit[jj])
            BO_2qubits_easy.append(temp_matrix)

    # "Hard"
    BO_2qubits_hard = []
    for ii in range(0, 9):
        if ii == 6:
            # print(f'interm_op {ii}, 3 combinations')
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

            BO_2qubits_hard.append(temp_matrix1)
            BO_2qubits_hard.append(temp_matrix2)
            BO_2qubits_hard.append(temp_matrix3)

        elif ii == 7:
            # print(f'interm_op {ii}, 6 combinations')
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

            BO_2qubits_hard.append(temp_matrix1)
            BO_2qubits_hard.append(temp_matrix2)
            BO_2qubits_hard.append(temp_matrix3)
            BO_2qubits_hard.append(temp_matrix4)
            BO_2qubits_hard.append(temp_matrix5)
            BO_2qubits_hard.append(temp_matrix6)
        else:
            # print(f'interm_op {ii}, 9 combinations')
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

            BO_2qubits_hard.append(temp_matrix1)
            BO_2qubits_hard.append(temp_matrix2)
            BO_2qubits_hard.append(temp_matrix3)
            BO_2qubits_hard.append(temp_matrix4)
            BO_2qubits_hard.append(temp_matrix5)
            BO_2qubits_hard.append(temp_matrix6)
            BO_2qubits_hard.append(temp_matrix7)
            BO_2qubits_hard.append(temp_matrix8)
            BO_2qubits_hard.append(temp_matrix9)

    BasisOps_241 = BO_2qubits_easy + BO_2qubits_hard

    ###########################################################################################################################################

    ####################################
    ### 169 scenarios # (2020-12-05) ###
    ####################################

    BasisOps_169 = BO_1qubit
    gatenames = (
        "BasisOp00",
        "BasisOp01",
        "BasisOp02",
        "BasisOp03",
        "BasisOp04",
        "BasisOp05",
        "BasisOp06",
        "BasisOp07",
        "BasisOp08",
        "BasisOp09",
        "BasisOp10",
        "BasisOp11",
        "BasisOp12",
    )

    tic = time.time()
    idx_basis_ops = 0
    nqubits = 2

    Bjk_tilde_2qb = np.zeros((241, 16, 16))

    for idx_gatenames_1 in range(0, 13):
        for idx_gatenames_2 in range(0, 13):
            print("GST 2qb 241 Basis Op. Basis Op idx = %d" % (idx_basis_ops))

            # =========================================#
            # 16 SCENARIOS  #  13^2 BASIS OPERATIONS  #  -- BIG ENDIAN NOTATION FOR THE 13 BASIS GATES TENSORED PRODUCTS
            # =========================================#

            # Initialise 16 different initial states
            for k in range(0, 16):
                circ = prepare_2qb_state_new(k)

                # ==================================================================================

                if idx_gatenames_1 < 10 and idx_gatenames_2 < 10:  ### NEW SCENARIO 1
                    circ.add(
                        gates.Unitary(
                            BasisOps_169[idx_gatenames_1],
                            0,
                            trainable=False,
                            name="%s" % (gatenames[idx_gatenames_1]),
                        )
                    )
                    circ.add(
                        gates.Unitary(
                            BasisOps_169[idx_gatenames_2],
                            1,
                            trainable=False,
                            name="%s" % (gatenames[idx_gatenames_2]),
                        )
                    )

                ### PARTIALLY REPREPARE QUBIT_1

                elif idx_gatenames_1 == 10 and idx_gatenames_2 < 10:
                    # print('Partially reprepare Qubit_1: idx_gatenames_1 = %d, idx_gatenames_2 = %d ' %(idx_gatenames_1, idx_gatenames_1))
                    circ.add(
                        gates.ResetChannel(0, [1, 0])
                    )  # ResetChannel(qubit, [prob0, prob1])
                    circ.add(gates.H(0))
                    circ.add(
                        gates.Unitary(
                            BasisOps_169[idx_gatenames_2],
                            1,
                            trainable=False,
                            name="%s" % (gatenames[idx_gatenames_2]),
                        )
                    )

                elif idx_gatenames_1 == 11 and idx_gatenames_2 < 10:
                    # print('Partially reprepare Qubit_1: idx_gatenames_1 = %d, idx_gatenames_2 = %d ' %(idx_gatenames_1, idx_gatenames_2))
                    circ.add(
                        gates.ResetChannel(0, [1, 0])
                    )  # ResetChannel(qubit, [prob0, prob1])
                    circ.add(gates.H(0))
                    circ.add(gates.S(0))
                    circ.add(
                        gates.Unitary(
                            BasisOps_169[idx_gatenames_2],
                            1,
                            trainable=False,
                            name="%s" % (gatenames[idx_gatenames_2]),
                        )
                    )

                elif idx_gatenames_1 == 12 and idx_gatenames_2 < 10:
                    # print('Partially reprepare Qubit_1: idx_gatenames_1 = %d, idx_gatenames_2 = %d ' %(idx_gatenames_1,idx_gatenames_2))
                    circ.add(
                        gates.ResetChannel(0, [1, 0])
                    )  # ResetChannel(qubit, [prob0, prob1])
                    circ.add(
                        gates.Unitary(
                            BasisOps_169[idx_gatenames_2],
                            1,
                            trainable=False,
                            name="%s" % (gatenames[idx_gatenames_2]),
                        )
                    )

                ### PARTIALLY REPREPARE QUBIT_2

                elif idx_gatenames_1 < 10 and idx_gatenames_2 == 10:
                    # print('Partially reprepare Qubit_2: idx_gatenames_1 = %d, idx_gatenames_2 = %d ' %(idx_gatenames_1, idx_gatenames_2))
                    circ.add(
                        gates.Unitary(
                            BasisOps_169[idx_gatenames_1],
                            0,
                            trainable=False,
                            name="%s" % (gatenames[idx_gatenames_1]),
                        )
                    )
                    circ.add(
                        gates.ResetChannel(1, [1, 0])
                    )  # ResetChannel(qubit, [prob0, prob1])
                    circ.add(gates.H(1))

                elif idx_gatenames_1 < 10 and idx_gatenames_2 == 11:
                    # print('Partially reprepare Qubit_2: idx_gatenames_1 = %d, idx_gatenames_2 = %d ' %(idx_gatenames_1,idx_gatenames_2))
                    circ.add(
                        gates.Unitary(
                            BasisOps_169[idx_gatenames_1],
                            0,
                            trainable=False,
                            name="%s" % (gatenames[idx_gatenames_1]),
                        )
                    )
                    circ.add(
                        gates.ResetChannel(1, [1, 0])
                    )  # ResetChannel(qubit, [prob0, prob1])
                    circ.add(gates.H(1))
                    circ.add(gates.S(1))

                elif idx_gatenames_1 < 10 and idx_gatenames_2 == 12:
                    # print('Partially reprepare Qubit_2: idx_gatenames_1 = %d, idx_gatenames_2 = %d ' %(idx_gatenames_1,idx_gatenames_2))
                    circ.add(
                        gates.Unitary(
                            BasisOps_169[idx_gatenames_1],
                            0,
                            trainable=False,
                            name="%s" % (gatenames[idx_gatenames_1]),
                        )
                    )
                    circ.add(
                        gates.ResetChannel(1, [1, 0])
                    )  # ResetChannel(qubit, [prob0, prob1])

                ### FULLY REPREPARE QUBIT_1 AND QUBIT_2

                elif idx_gatenames_1 == 10 and idx_gatenames_2 == 10:
                    # qubit_1 = reinitialised |+> state
                    # qubit_2 = reinitialised |+> state
                    circ.add(
                        gates.ResetChannel(0, [1, 0])
                    )  # ResetChannel(qubit, [prob0, prob1])
                    circ.add(
                        gates.ResetChannel(1, [1, 0])
                    )  # ResetChannel(qubit, [prob0, prob1])
                    circ.add(gates.H(0))
                    circ.add(gates.H(1))

                elif idx_gatenames_1 == 10 and idx_gatenames_2 == 11:
                    # qubit_1 = reinitialised |+> state
                    # qubit_2 = reinitialised |y+> state
                    circ.add(
                        gates.ResetChannel(0, [1, 0])
                    )  # ResetChannel(qubit, [prob0, prob1])
                    circ.add(
                        gates.ResetChannel(1, [1, 0])
                    )  # ResetChannel(qubit, [prob0, prob1])
                    circ.add(gates.H(0))
                    circ.add(gates.H(1))
                    circ.add(gates.S(1))

                elif idx_gatenames_1 == 10 and idx_gatenames_2 == 12:
                    # qubit_1 = reinitialised |+> state
                    # qubit_2 = reinitialised |0> state
                    circ.add(
                        gates.ResetChannel(0, [1, 0])
                    )  # ResetChannel(qubit, [prob0, prob1])
                    circ.add(
                        gates.ResetChannel(1, [1, 0])
                    )  # ResetChannel(qubit, [prob0, prob1])
                    circ.add(gates.H(0))

                # ==================================================================================
                elif idx_gatenames_1 == 11 and idx_gatenames_2 == 10:
                    # qubit_1 = reinitialised |y+> state
                    # qubit_2 = reinitialised |+> state
                    circ.add(
                        gates.ResetChannel(0, [1, 0])
                    )  # ResetChannel(qubit, [prob0, prob1])
                    circ.add(
                        gates.ResetChannel(1, [1, 0])
                    )  # ResetChannel(qubit, [prob0, prob1])
                    circ.add(gates.H(0))
                    circ.add(gates.S(0))
                    circ.add(gates.H(1))

                elif idx_gatenames_1 == 11 and idx_gatenames_2 == 11:
                    # qubit_1 = reinitialised |y+> state
                    # qubit_2 = reinitialised |y+> state
                    circ.add(
                        gates.ResetChannel(0, [1, 0])
                    )  # ResetChannel(qubit, [prob0, prob1])
                    circ.add(
                        gates.ResetChannel(1, [1, 0])
                    )  # ResetChannel(qubit, [prob0, prob1])
                    circ.add(gates.H(0))
                    circ.add(gates.S(0))
                    circ.add(gates.H(1))
                    circ.add(gates.S(1))

                elif idx_gatenames_1 == 11 and idx_gatenames_2 == 12:
                    # qubit_1 = reinitialised |y+> state
                    # qubit_2 = reinitialised |0> state
                    circ.add(
                        gates.ResetChannel(0, [1, 0])
                    )  # ResetChannel(qubit, [prob0, prob1])
                    circ.add(
                        gates.ResetChannel(1, [1, 0])
                    )  # ResetChannel(qubit, [prob0, prob1])
                    circ.add(gates.H(0))
                    circ.add(gates.S(0))

                # ==================================================================================
                elif idx_gatenames_1 == 12 and idx_gatenames_2 == 10:
                    # qubit_1 = reinitialised |0> state
                    # qubit_2 = reinitialised |+> state
                    circ.add(
                        gates.ResetChannel(0, [1, 0])
                    )  # ResetChannel(qubit, [prob0, prob1])
                    circ.add(
                        gates.ResetChannel(1, [1, 0])
                    )  # ResetChannel(qubit, [prob0, prob1])
                    circ.add(gates.H(1))

                elif idx_gatenames_1 == 12 and idx_gatenames_2 == 11:
                    # qubit_1 = reinitialised |0> state
                    # qubit_2 = reinitialised |y+> state
                    circ.add(
                        gates.ResetChannel(0, [1, 0])
                    )  # ResetChannel(qubit, [prob0, prob1])
                    circ.add(
                        gates.ResetChannel(1, [1, 0])
                    )  # ResetChannel(qubit, [prob0, prob1])
                    circ.add(gates.H(1))
                    circ.add(gates.S(1))

                elif idx_gatenames_1 == 12 and idx_gatenames_2 == 12:
                    # qubit_1 = reinitialised |0> state
                    # qubit_2 = reinitialised |0> state
                    circ.add(
                        gates.ResetChannel(0, [1, 0])
                    )  # ResetChannel(qubit, [prob0, prob1])
                    circ.add(
                        gates.ResetChannel(1, [1, 0])
                    )  # ResetChannel(qubit, [prob0, prob1])

                # ==================================================================================

                column_of_js = np.zeros((4**nqubits, 1))
                # ================= START OF 16 MEASUREMENT BASES =================|||
                for j in range(0, 4**nqubits):
                    new_circ = measure_2qb_state_new(j, circ, noise_model=None)
                    # expectation_val = GST_measure_241(new_circ, idx_basis_ops, k, j, nshots, save_data)
                    expectation_val = GST_measure(
                        new_circ,
                        k,
                        j,
                        nshots,
                        nqubits,
                        class_qubit_gate,
                        theta,
                        idx_basis_ops,
                        save_data,
                    )
                    column_of_js[j, :] = expectation_val

                Bjk_tilde_2qb[idx_basis_ops, :, k] = column_of_js.reshape(
                    16,
                )

            idx_basis_ops += 1

    toc = time.time() - tic

    ###################################
    ### 72 scenarios # (2020-12-05) ###
    ###################################

    tic = time.time()
    idx_basis_ops = 169
    for idx_basis_ops_72 in range(169, 241):
        print("GST 2qb 241 Basis Op. Basis Op idx = %d" % (idx_basis_ops_72))

        # ===============#
        # 72 SCENARIOS  #
        # ===============#

        # Initialise 16 different initial states
        for k in range(0, 16):
            circ = prepare_2qb_state_new(k)

            # ==================================================================================

            circ.add(
                gates.Unitary(
                    BasisOps_241[idx_basis_ops_72],
                    0,
                    1,
                    trainable=False,
                    name="%s" % (idx_basis_ops_72),
                )
            )

            # ================= START OF 16 MEASUREMENT BASES =================|||
            for j in range(0, 4**nqubits):
                new_circ = measure_2qb_state_new(j, circ, noise_model=None)
                # expectation_val = GST_measure_241(new_circ, idx_basis_ops, k, j, nshots, save_data)
                expectation_val = GST_measure(
                    new_circ,
                    k,
                    j,
                    nshots,
                    nqubits,
                    class_qubit_gate,
                    theta,
                    idx_basis_ops,
                    save_data,
                )
                column_of_js[j, :] = expectation_val

            Bjk_tilde_2qb[idx_basis_ops, :, k] = column_of_js.reshape(
                16,
            )

        idx_basis_ops += 1

    toc = time.time() - tic

    # ########################################################################
    # ### Get noisy Pauli Transfer Matrix for each of 241 basis operations ###
    # ########################################################################

    T = np.array([[1, 1, 1, 1], [0, 0, 1, 0], [0, 0, 0, 1], [1, -1, 0, 0]])
    Bjk_hat_2qb = np.zeros((241, 4**nqubits, 4**nqubits))
    for idx_BO in range(0, 241):
        Bjk_hat_2qb[idx_BO, :, :] = (
            np.kron(T, T)
            @ np.linalg.inv(gjk_2qb)
            @ Bjk_tilde_2qb[idx_BO, :, :]
            @ np.linalg.inv(np.kron(T, T))
        )

    # Reshape
    Bjk_hat_2qb_reshaped = np.zeros((256, 241))
    for idx_basis_ops in range(0, 241):
        temp = np.reshape(
            Bjk_hat_2qb[idx_basis_ops, :, :], [256, 1], order="F"
        )  # 'F' so that it stacks columns.
        Bjk_hat_2qb_reshaped[:, idx_basis_ops] = temp[:, 0].flatten()

    return Bjk_hat_2qb, Bjk_hat_2qb_reshaped, BasisOps_241
