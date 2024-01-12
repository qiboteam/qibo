def sort_counts_1qb(matrix):
    import numpy as np

    counts_matrix = np.zeros((2, 2))

    for ii in range(0, 2):
        word = bin(ii)[2:]
        counts_matrix[ii, 0] = word

    for row in matrix:
        row_number = int(str(int(row[0])), 2)
        counts_matrix[row_number, 1] = row[1]

    return counts_matrix


def sort_counts_2qb(matrix):
    import numpy as np

    counts_matrix = np.zeros((4, 2))

    for ii in range(0, 4):
        word = bin(ii)[2:]
        counts_matrix[ii, 0] = word

    for row in matrix:
        row_number = int(str(int(row[0])), 2)
        counts_matrix[row_number, 1] = row[1]

    return counts_matrix


def GST_1qb(
    NshotsGST=1e4,
    noise_model=None,
    save_data=None,
    class_qubit_gate=None,
    theta=None,
    operator_name=None,
    backend=None,
):
    """Runs gate set tomography for a single qubit gate/circuit.

        Args:
        NshotsGST (int, optional): Number of shots used in Gate Set Tomography.
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

    import numpy as np

    from qibo import gates
    from qibo.backends import GlobalBackend
    from qibo.models import Circuit

    if backend is None:  # pragma: no cover
        backend = GlobalBackend()

    if class_qubit_gate is None:
        print("Empty circuit without any operator (1 qb)")

    def GST_measure_1qb(circuit, k, j):
        idx_basis = ["I", "X", "Y", "Z"]

        result = circuit.execute(nshots=NshotsGST)
        counts = dict(result.frequencies(binary=True))

        if save_data is not None:
            import os

            # If location does not exist, create location.
            if class_qubit_gate is None:
                filedirectory = "GST_nogates_qibo_1qb/"
                fname = "No gates (k=%.2d,j=%.2d) %s_basis.txt" % (k, j, idx_basis[j])
                GST_nogates_qibo_1qb = "GST_nogates_qibo_1qb"
                if not os.path.exists(GST_nogates_qibo_1qb):
                    os.makedirs(GST_nogates_qibo_1qb)

            elif class_qubit_gate is not None:
                filedirectory = "GST_gates_qibo_1qb/"
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
                GST_gates_qibo_1qb = "GST_gates_qibo_1qb"
                if not os.path.exists(GST_gates_qibo_1qb):
                    os.makedirs(GST_gates_qibo_1qb)

            # Save counts into a matrix. 0th column: word; 1st column: count value
            # --------------------------------------------------------------------
            fout = filedirectory + fname
            fo = open(fout, "w")
            for v1, v2 in counts.items():
                fo.write(str(v1) + "  " + str(v2) + "\n")
            fo.close()

        # Find value for matrix_jk
        # -------------------------

        expectation_signs = np.matrix([[1, 1, 1, 1], [1, -1, -1, -1]])

        matrix = np.zeros((2, 2))
        for ii in range(0, 2):
            word = bin(ii)[2:]
            matrix[ii, 0] = word

        for v1, v2 in counts.items():
            # print(v1, v2)
            row = int(v1, 2)
            col = v2
            matrix[row, 1] = col

        sorted_matrix = sort_counts_1qb(matrix)
        probs = sorted_matrix[:, 1] / np.sum(sorted_matrix[:, 1])
        sorted_matrix = np.column_stack((sorted_matrix, probs))

        k = k + 1
        j = j + 1

        if j == 1:
            sorted_matrix = np.column_stack((sorted_matrix, expectation_signs[:, 0]))
        elif j == 2:
            sorted_matrix = np.column_stack((sorted_matrix, expectation_signs[:, 1]))
        elif j == 3:
            sorted_matrix = np.column_stack((sorted_matrix, expectation_signs[:, 1]))
        elif j == 4:
            sorted_matrix = np.column_stack((sorted_matrix, expectation_signs[:, 1]))
        elif j == 5:
            sorted_matrix = np.column_stack((sorted_matrix, expectation_signs[:, 2]))
        elif j == 9:
            sorted_matrix = np.column_stack((sorted_matrix, expectation_signs[:, 2]))
        elif j == 13:
            sorted_matrix = np.column_stack((sorted_matrix, expectation_signs[:, 2]))
        else:
            sorted_matrix = np.column_stack((sorted_matrix, expectation_signs[:, 3]))

        temp = np.array(sorted_matrix[:, 2]) * np.array(sorted_matrix[:, 3])
        sorted_matrix = np.column_stack((sorted_matrix, temp))

        expectation_val = np.sum(temp)

        return expectation_val

    nqubits = 1
    matrix_jk_1qb = np.zeros((4**nqubits, 4**nqubits))
    for k in range(0, 4):
        if k == 0:  # |0>
            circ = Circuit(nqubits, density_matrix=True)

        elif k == 1:  # |1>
            circ = Circuit(nqubits, density_matrix=True)
            circ.add(gates.X(0))

        elif k == 2:  # |+>
            circ = Circuit(nqubits, density_matrix=True)
            circ.add(gates.H(0))

        elif k == 3:  # |y+>
            circ = Circuit(nqubits, density_matrix=True)
            circ.add(gates.H(0))
            circ.add(gates.S(0))

        # If no class_qubit_gate
        if class_qubit_gate is None:
            pass

        # If class_qubit_gate is present (with/without theta)
        elif class_qubit_gate is not None:
            if theta is None:
                print(f"Gate set tomography for 1 qubit gate: {class_qubit_gate}.")
                circ.add(getattr(gates, class_qubit_gate)(0))
            elif theta is not None:
                print(
                    f"Gate set tomography for 1 qubit gate: {class_qubit_gate}({theta})."
                )
                circ.add(getattr(gates, class_qubit_gate)(0, theta))

        # ================= START OF 4 MEASUREMENT BASES =================
        for j in range(0, 4):
            if j == 0:  # Identity basis
                circ2 = Circuit(nqubits, density_matrix=True)
                new_circ = circ + circ2
                if noise_model is not None and backend.name != "qibolab":
                    new_circ = noise_model.apply(new_circ)
                new_circ.add(gates.M(0, basis=gates.Z))
                expectation_val = GST_measure_1qb(new_circ, k, j)

            elif j == 1:  # X basis
                circ2 = Circuit(nqubits, density_matrix=True)
                new_circ = circ + circ2
                if noise_model is not None and backend.name != "qibolab":
                    new_circ = noise_model.apply(new_circ)
                new_circ.add(gates.M(0, basis=gates.X))
                expectation_val = GST_measure_1qb(new_circ, k, j)

            elif j == 2:  # Y basis
                circ2 = Circuit(nqubits, density_matrix=True)
                new_circ = circ + circ2
                if noise_model is not None and backend.name != "qibolab":
                    new_circ = noise_model.apply(new_circ)
                new_circ.add(gates.M(0, basis=gates.Y))
                expectation_val = GST_measure_1qb(new_circ, k, j)

            elif j == 3:  # Z basis
                circ2 = Circuit(nqubits, density_matrix=True)
                new_circ = circ + circ2
                if noise_model is not None and backend.name != "qibolab":
                    new_circ = noise_model.apply(new_circ)
                new_circ.add(gates.M(0, basis=gates.Z))
                expectation_val = GST_measure_1qb(new_circ, k, j)

            matrix_jk_1qb[j, k] = expectation_val

    matrix_jk_1qb = np.matrix(matrix_jk_1qb)

    return matrix_jk_1qb


def GST_2qb(
    NshotsGST=1e4,
    noise_model=None,
    save_data=None,
    class_qubit_gate=None,
    ctrl_qb=None,
    targ_qb=None,
    theta=None,
    operator_name=None,
    backend=None,
):
    """Runs gate set tomography for a two qubit gate/circuit.

        Args:
        NshotsGST (int, optional): Number of shots used in Gate Set Tomography.
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

    import numpy as np

    from qibo import gates
    from qibo.backends import GlobalBackend
    from qibo.models import Circuit

    if backend is None:  # pragma: no cover
        backend = GlobalBackend()

    if class_qubit_gate is None:
        print("Empty circuit without any operator (2 qb)")

    def GST_measure_2qb(circuit, k, j):
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

        result = circuit.execute(nshots=NshotsGST)
        counts = dict(result.frequencies(binary=True))

        if save_data is not None:
            import os

            # If location does not exist, create location.
            if class_qubit_gate is None:
                filedirectory = "GST_nogates_qibo_2qb/"
                fname = "No gates (k=%.2d,j=%.2d) %s_basis.txt" % (k, j, idx_basis[j])
                GST_nogates_qibo_2qb = "GST_nogates_qibo_2qb"
                if not os.path.exists(GST_nogates_qibo_2qb):
                    os.makedirs(GST_nogates_qibo_2qb)

            elif class_qubit_gate is not None:
                filedirectory = "GST_gates_qibo_2qb/"
                fname = "%s gate (k=%.2d,j=%.2d) %s_basis.txt" % (
                    class_qubit_gate,
                    k,
                    j,
                    idx_basis[j],
                )
                GST_gates_qibo_2qb = "GST_gates_qibo_2qb"
                if not os.path.exists(GST_gates_qibo_2qb):
                    os.makedirs(GST_gates_qibo_2qb)

            # Save counts into a matrix. 0th column: word; 1st column: count value
            # --------------------------------------------------------------------
            fout = filedirectory + fname
            fo = open(fout, "w")
            for v1, v2 in counts.items():
                fo.write(str(v1) + "  " + str(v2) + "\n")
            fo.close()

        # Find value for matrix_jk
        # -------------------------

        expectation_signs = np.matrix(
            [[1, 1, 1, 1], [1, -1, 1, -1], [1, 1, -1, -1], [1, -1, -1, 1]]
        )

        matrix = np.zeros((2**2, 2))
        for ii in range(0, 2**2):
            word = bin(ii)[2:]
            matrix[ii, 0] = word

        for v1, v2 in counts.items():
            # print(v1, v2)
            row = int(v1, 2)
            col = v2
            matrix[row, 1] = col

        sorted_matrix = sort_counts_2qb(matrix)
        probs = sorted_matrix[:, 1] / np.sum(sorted_matrix[:, 1])
        sorted_matrix = np.column_stack((sorted_matrix, probs))

        k = k + 1
        j = j + 1

        if j == 1:
            sorted_matrix = np.column_stack((sorted_matrix, expectation_signs[:, 0]))
        elif j == 2:
            sorted_matrix = np.column_stack((sorted_matrix, expectation_signs[:, 1]))
        elif j == 3:
            sorted_matrix = np.column_stack((sorted_matrix, expectation_signs[:, 1]))
        elif j == 4:
            sorted_matrix = np.column_stack((sorted_matrix, expectation_signs[:, 1]))
        elif j == 5:
            sorted_matrix = np.column_stack((sorted_matrix, expectation_signs[:, 2]))
        elif j == 9:
            sorted_matrix = np.column_stack((sorted_matrix, expectation_signs[:, 2]))
        elif j == 13:
            sorted_matrix = np.column_stack((sorted_matrix, expectation_signs[:, 2]))
        else:
            sorted_matrix = np.column_stack((sorted_matrix, expectation_signs[:, 3]))

        temp = np.array(sorted_matrix[:, 2]) * np.array(sorted_matrix[:, 3])
        sorted_matrix = np.column_stack((sorted_matrix, temp))

        expectation_val = np.sum(temp)

        return expectation_val

    nqubits = 2
    matrix_jk_2qb = np.zeros((4**nqubits, 4**nqubits))
    for k in range(0, 16):
        if k == 0:  # |0> |0>
            circ = Circuit(nqubits, density_matrix=True)

        elif k == 1:  # |0> |1>
            circ = Circuit(nqubits, density_matrix=True)
            circ.add(gates.X(1))

        elif k == 2:  # |0> |+>
            circ = Circuit(nqubits, density_matrix=True)
            circ.add(gates.H(1))

        elif k == 3:  # |0> |y+>
            circ = Circuit(nqubits, density_matrix=True)
            circ.add(gates.H(1))
            circ.add(gates.S(1))
        # --------------------------------------------

        elif k == 4:  # |1> |0>
            circ = Circuit(nqubits, density_matrix=True)
            circ.add(gates.X(0))

        elif k == 5:  # |1> |1>
            circ = Circuit(nqubits, density_matrix=True)
            circ.add(gates.X(0))
            circ.add(gates.X(1))

        elif k == 6:  # |1> |+>
            circ = Circuit(nqubits, density_matrix=True)
            circ.add(gates.X(0))
            circ.add(gates.H(1))

        elif k == 7:  # |1> |y+>
            circ = Circuit(nqubits, density_matrix=True)
            circ.add(gates.X(0))
            circ.add(gates.H(1))
            circ.add(gates.S(1))

        # --------------------------------------------

        elif k == 8:  # |+> |0>
            circ = Circuit(nqubits, density_matrix=True)
            circ.add(gates.H(0))

        elif k == 9:  # |+> |1>
            circ = Circuit(nqubits, density_matrix=True)
            circ.add(gates.H(0))
            circ.add(gates.X(1))

        elif k == 10:  # |+> |+>
            circ = Circuit(nqubits, density_matrix=True)
            circ.add(gates.H(0))
            circ.add(gates.H(1))

        elif k == 11:  # |+> |y+>
            circ = Circuit(nqubits, density_matrix=True)
            circ.add(gates.H(0))
            circ.add(gates.H(1))
            circ.add(gates.S(1))

        # --------------------------------------------

        elif k == 12:  # |y+> |0>
            circ = Circuit(nqubits, density_matrix=True)
            circ.add(gates.H(0))
            circ.add(gates.S(0))

        elif k == 13:  # |y+> |1>
            circ = Circuit(nqubits, density_matrix=True)
            circ.add(gates.H(0))
            circ.add(gates.S(0))
            circ.add(gates.X(1))

        elif k == 14:  # |y+> |+>
            circ = Circuit(nqubits, density_matrix=True)
            circ.add(gates.H(0))
            circ.add(gates.S(0))
            circ.add(gates.H(1))

        elif k == 15:  # |y+> |y+>
            circ = Circuit(nqubits, density_matrix=True)
            circ.add(gates.H(0))
            circ.add(gates.S(0))
            circ.add(gates.H(1))
            circ.add(gates.S(1))

        # --------------------------------------------

        # If no class_qubit_gate
        if class_qubit_gate is None:
            pass

        # If class_qubit_gate is present (with/without theta)
        elif class_qubit_gate is not None:
            if theta is None:
                print(f"Gate set tomography for 2-qubit gate: {class_qubit_gate}.")
                circ.add(getattr(gates, class_qubit_gate)(targ_qb[0], ctrl_qb[0]))
            elif theta is not None:
                print(
                    f"Gate set tomography for 2-qubit gate: {class_qubit_gate}({theta})."
                )
                circ.add(
                    getattr(gates, class_qubit_gate)(targ_qb[0], ctrl_qb[0], theta)
                )

        # --------------------------------------------

        # initial_state = np.zeros((2**nqubits))
        # initial_state[0] = 1
        # ================= START OF 16 MEASUREMENT BASES =================|||
        for j in range(0, 16):
            # print('GST 2qb blank circ: initial state k=%d, measurement basis %d' %(k,j))

            if j == 0:  # Identity basis  # Identity basis
                circ2 = Circuit(nqubits, density_matrix=True)
                new_circ = circ + circ2
                if noise_model is not None and backend.name != "qibolab":
                    new_circ = noise_model.apply(new_circ)
                new_circ.add(gates.M(0, basis=gates.Z))
                new_circ.add(gates.M(1, basis=gates.Z))

                expectation_val = GST_measure_2qb(new_circ, k, j)

            elif j == 1:  # Identity basis # X basis
                circ2 = Circuit(nqubits, density_matrix=True)
                new_circ = circ + circ2
                if noise_model is not None and backend.name != "qibolab":
                    new_circ = noise_model.apply(new_circ)
                new_circ.add(gates.M(0, basis=gates.Z))
                new_circ.add(gates.M(1, basis=gates.X))

                expectation_val = GST_measure_2qb(new_circ, k, j)

            elif j == 2:  # Identity basis # Y basis
                circ2 = Circuit(nqubits, density_matrix=True)
                new_circ = circ + circ2
                if noise_model is not None and backend.name != "qibolab":
                    new_circ = noise_model.apply(new_circ)
                new_circ.add(gates.M(0, basis=gates.Z))
                new_circ.add(gates.M(1, basis=gates.Y))

                expectation_val = GST_measure_2qb(new_circ, k, j)

            elif j == 3:  # Identity basis # Z basis
                circ2 = Circuit(nqubits, density_matrix=True)
                new_circ = circ + circ2
                if noise_model is not None and backend.name != "qibolab":
                    new_circ = noise_model.apply(new_circ)
                new_circ.add(gates.M(0, basis=gates.Z))
                new_circ.add(gates.M(1, basis=gates.Z))

                expectation_val = GST_measure_2qb(new_circ, k, j)

            # =================================================================================================================

            elif j == 4:  # X basis# Identity basis
                circ2 = Circuit(nqubits, density_matrix=True)
                new_circ = circ + circ2
                if noise_model is not None and backend.name != "qibolab":
                    new_circ = noise_model.apply(new_circ)
                new_circ.add(gates.M(0, basis=gates.X))
                new_circ.add(gates.M(1, basis=gates.Z))

                expectation_val = GST_measure_2qb(new_circ, k, j)

            elif j == 5:  # X basis # X basis
                circ2 = Circuit(nqubits, density_matrix=True)
                new_circ = circ + circ2
                if noise_model is not None and backend.name != "qibolab":
                    new_circ = noise_model.apply(new_circ)
                new_circ.add(gates.M(0, basis=gates.X))
                new_circ.add(gates.M(1, basis=gates.X))

                expectation_val = GST_measure_2qb(new_circ, k, j)

            elif j == 6:  # X basis # Y basis
                circ2 = Circuit(nqubits, density_matrix=True)
                new_circ = circ + circ2
                if noise_model is not None and backend.name != "qibolab":
                    new_circ = noise_model.apply(new_circ)
                new_circ.add(gates.M(0, basis=gates.X))
                new_circ.add(gates.M(1, basis=gates.Y))

                expectation_val = GST_measure_2qb(new_circ, k, j)

            elif j == 7:  # X basis  # Z basis
                circ2 = Circuit(nqubits, density_matrix=True)
                new_circ = circ + circ2
                if noise_model is not None and backend.name != "qibolab":
                    new_circ = noise_model.apply(new_circ)
                new_circ.add(gates.M(0, basis=gates.X))
                new_circ.add(gates.M(1, basis=gates.Z))

                expectation_val = GST_measure_2qb(new_circ, k, j)

            # =================================================================================================================

            elif j == 8:  # Y basis # Identity basis
                circ2 = Circuit(nqubits, density_matrix=True)
                new_circ = circ + circ2
                if noise_model is not None and backend.name != "qibolab":
                    new_circ = noise_model.apply(new_circ)
                new_circ.add(gates.M(0, basis=gates.Y))
                new_circ.add(gates.M(1, basis=gates.Z))

                expectation_val = GST_measure_2qb(new_circ, k, j)

            elif j == 9:  # Y basis  # X basis
                circ2 = Circuit(nqubits, density_matrix=True)
                new_circ = circ + circ2
                if noise_model is not None and backend.name != "qibolab":
                    new_circ = noise_model.apply(new_circ)
                new_circ.add(gates.M(0, basis=gates.Y))
                new_circ.add(gates.M(1, basis=gates.X))

                expectation_val = GST_measure_2qb(new_circ, k, j)

            elif j == 10:  # Y basis  # Y basis
                circ2 = Circuit(nqubits, density_matrix=True)
                new_circ = circ + circ2
                if noise_model is not None and backend.name != "qibolab":
                    new_circ = noise_model.apply(new_circ)
                new_circ.add(gates.M(0, basis=gates.Y))
                new_circ.add(gates.M(1, basis=gates.Y))

                expectation_val = GST_measure_2qb(new_circ, k, j)

            elif j == 11:  # Y basis  # Z basis
                circ2 = Circuit(nqubits, density_matrix=True)
                new_circ = circ + circ2
                if noise_model is not None and backend.name != "qibolab":
                    new_circ = noise_model.apply(new_circ)
                new_circ.add(gates.M(0, basis=gates.Y))
                new_circ.add(gates.M(1, basis=gates.Z))

                expectation_val = GST_measure_2qb(new_circ, k, j)

            # =================================================================================================================

            elif j == 12:  # Z basis# Identity basis
                circ2 = Circuit(nqubits, density_matrix=True)
                new_circ = circ + circ2
                if noise_model is not None and backend.name != "qibolab":
                    new_circ = noise_model.apply(new_circ)
                new_circ.add(gates.M(0, basis=gates.Z))
                new_circ.add(gates.M(1, basis=gates.Z))

                expectation_val = GST_measure_2qb(new_circ, k, j)

            elif j == 13:  # Z basis  # X basis
                circ2 = Circuit(nqubits, density_matrix=True)
                new_circ = circ + circ2
                if noise_model is not None and backend.name != "qibolab":
                    new_circ = noise_model.apply(new_circ)
                new_circ.add(gates.M(0, basis=gates.Z))
                new_circ.add(gates.M(1, basis=gates.X))

                expectation_val = GST_measure_2qb(new_circ, k, j)

            elif j == 14:  # Z basis  # Y basis
                circ2 = Circuit(nqubits, density_matrix=True)
                new_circ = circ + circ2
                if noise_model is not None and backend.name != "qibolab":
                    new_circ = noise_model.apply(new_circ)
                new_circ.add(gates.M(0, basis=gates.Z))
                new_circ.add(gates.M(1, basis=gates.Y))

                expectation_val = GST_measure_2qb(new_circ, k, j)

            elif j == 15:  # Z basis # Z basis
                circ2 = Circuit(nqubits, density_matrix=True)
                new_circ = circ + circ2
                if noise_model is not None and backend.name != "qibolab":
                    new_circ = noise_model.apply(new_circ)
                new_circ.add(gates.M(0, basis=gates.Z))
                new_circ.add(gates.M(1, basis=gates.Z))

                expectation_val = GST_measure_2qb(new_circ, k, j)

            # print(expectation_val)
            matrix_jk_2qb[j, k] = expectation_val

    matrix_jk_2qb = np.matrix(matrix_jk_2qb)

    return matrix_jk_2qb


def GST_1qb_basis_operations(
    NshotsGST=1e4, noise_model=None, gjk_1qb=None, save_data=None, backend=None
):
    """Runs gate set tomography for the 13 basis operations for a single qubit.

        Args:
        NshotsGST (int, optional): Number of shots used in Gate Set Tomography.
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

    import numpy as np

    from qibo import gates
    from qibo.backends import GlobalBackend
    from qibo.models import Circuit

    if backend is None:  # pragma: no cover
        backend = GlobalBackend()

    if gjk_1qb is None:
        gjk_1qb = GST_1qb(NshotsGST, noise_model, save_data=None)

    BasisOps_13 = [
        np.matrix(gates.I(0).matrix(backend=GlobalBackend())),
        np.matrix(gates.X(0).matrix(backend=GlobalBackend())),
        np.matrix(gates.Y(0).matrix(backend=GlobalBackend())),
        np.matrix(gates.Z(0).matrix(backend=GlobalBackend())),
        np.matrix(
            [
                [1 / np.sqrt(2) * 1, 1 / np.sqrt(2) * 1j],
                [1 / np.sqrt(2) * 1j, 1 / np.sqrt(2) * 1],
            ]
        ),
        np.matrix(
            [
                [1 / np.sqrt(2) * 1, 1 / np.sqrt(2) * 1],
                [1 / np.sqrt(2) * (-1), 1 / np.sqrt(2) * 1],
            ]
        ),
        np.matrix([[1 / np.sqrt(2) * (1 + 1j), 0], [0, 1 / np.sqrt(2) * (1 - 1j)]]),
        np.matrix(
            [
                [1 / np.sqrt(2) * 1, 1 / np.sqrt(2) * (-1j)],
                [1 / np.sqrt(2) * 1j, 1 / np.sqrt(2) * (-1)],
            ]
        ),
        np.matrix(
            [
                [1 / np.sqrt(2) * 1, 1 / np.sqrt(2) * 1],
                [1 / np.sqrt(2) * 1, 1 / np.sqrt(2) * (-1)],
            ]
        ),
        np.matrix([[0, 1 / np.sqrt(2) * (1 - 1j)], [1 / np.sqrt(2) * (1 + 1j), 0]]),
        np.matrix([[1, 0], [0, 0]]),
        np.matrix([[0.5, 0.5], [0.5, 0.5]]),
        np.matrix([[0.5, -0.5j], [0.5j, 0.5]]),
    ]

    def GST_measure_13(circuit, idx_basis_op, k, j):
        nqubits = 1
        idx_basis = ["I", "X", "Y", "Z"]

        result = circuit.execute(nshots=NshotsGST)
        counts = dict(result.frequencies(binary=True))

        if save_data is not None:
            import os

            filedirectory = "GST_basisops_qibo_1qb/"
            fname = "BasisOp%.2d (k=%.2d,j=%.2d) %s_basis.txt" % (
                idx_basis_op,
                k,
                j,
                idx_basis[j],
            )
            # If location does not exist, create location.
            GST_basisops_qibo_1qb = "GST_basisops_qibo_1qb"
            if not os.path.exists(GST_basisops_qibo_1qb):
                os.makedirs(GST_basisops_qibo_1qb)

            # Save counts into a matrix. 0th column: word; 1st column: count value
            # --------------------------------------------------------------------
            fout = filedirectory + fname
            fo = open(fout, "w")
            for v1, v2 in counts.items():
                fo.write(str(v1) + "  " + str(v2) + "\n")
            fo.close()

        # Find value for Bjk tilde
        # -------------------------

        expectation_signs = np.matrix([[1, 1, 1, 1], [1, -1, -1, -1]])

        matrix = np.zeros((2**nqubits, 2**nqubits))
        for ii in range(0, 2):
            word = bin(ii)[2:]
            matrix[ii, 0] = word

        for v1, v2 in counts.items():
            # print(v1, v2)
            row = int(v1, 2)
            col = v2
            matrix[row, 1] = col

        sorted_matrix = sort_counts_1qb(matrix)
        probs = sorted_matrix[:, 1] / np.sum(sorted_matrix[:, 1])
        sorted_matrix = np.column_stack((sorted_matrix, probs))

        k = k + 1
        j = j + 1

        if j == 1:
            sorted_matrix = np.column_stack((sorted_matrix, expectation_signs[:, 0]))
        elif j == 2:
            sorted_matrix = np.column_stack((sorted_matrix, expectation_signs[:, 1]))
        elif j == 3:
            sorted_matrix = np.column_stack((sorted_matrix, expectation_signs[:, 1]))
        elif j == 4:
            sorted_matrix = np.column_stack((sorted_matrix, expectation_signs[:, 1]))
        elif j == 5:
            sorted_matrix = np.column_stack((sorted_matrix, expectation_signs[:, 2]))
        elif j == 9:
            sorted_matrix = np.column_stack((sorted_matrix, expectation_signs[:, 2]))
        elif j == 13:
            sorted_matrix = np.column_stack((sorted_matrix, expectation_signs[:, 2]))
        else:
            sorted_matrix = np.column_stack((sorted_matrix, expectation_signs[:, 3]))

        temp = np.array(sorted_matrix[:, 2]) * np.array(sorted_matrix[:, 3])
        sorted_matrix = np.column_stack((sorted_matrix, temp))

        expectation_val = np.sum(temp)

        return expectation_val

    def GSTBasisOpMeasurements_13(circ):
        nqubits = 1
        initial_state = np.zeros(2**nqubits)
        initial_state[0] = 1

        column_of_js = np.zeros((4**nqubits, 1))

        # ================= START OF 16 MEASUREMENT BASES =================|||
        for j in range(0, 4**nqubits):
            # print('[def GSTBasisOpMeasurements]: initial state k=%d, idx_gatenames_1 = %d, idx_gatenames_2 = %d, measurement basis %d' %(k,idx_gatenames_1,idx_gatenames_2,j))

            if j == 0:  # Identity basis
                circ2 = Circuit(nqubits, density_matrix=True)
                new_circ = circ + circ2
                if noise_model is not None and backend.name != "qibolab":
                    new_circ = noise_model.apply(new_circ)
                new_circ.add(gates.M(0, basis=gates.Z))
                expectation_val = GST_measure_13(new_circ, idx_basis_op, k, j)

            elif j == 1:  # X basis
                circ2 = Circuit(nqubits, density_matrix=True)
                new_circ = circ + circ2
                if noise_model is not None and backend.name != "qibolab":
                    new_circ = noise_model.apply(new_circ)
                new_circ.add(gates.M(0, basis=gates.X))
                expectation_val = GST_measure_13(new_circ, idx_basis_op, k, j)

            elif j == 2:  # Y basis
                circ2 = Circuit(nqubits, density_matrix=True)
                new_circ = circ + circ2
                if noise_model is not None and backend.name != "qibolab":
                    new_circ = noise_model.apply(new_circ)
                new_circ.add(gates.M(0, basis=gates.Y))
                expectation_val = GST_measure_13(new_circ, idx_basis_op, k, j)

            elif j == 3:  # Z basis
                circ2 = Circuit(nqubits, density_matrix=True)
                new_circ = circ + circ2
                if noise_model is not None and backend.name != "qibolab":
                    new_circ = noise_model.apply(new_circ)
                new_circ.add(gates.M(0, basis=gates.Z))
                expectation_val = GST_measure_13(new_circ, idx_basis_op, k, j)

            column_of_js[j, :] = expectation_val

        return column_of_js

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
    idx_basis_op = 0
    nqubits = 1

    Bjk_tilde_1qb = np.zeros((13, 4, 4))

    for idx_basis_op in range(0, 13):
        print("GST 1qb 13 Basis Op. Basis Op idx = %d" % (idx_basis_op))

        time.sleep(0)

        # ===============#=========================#
        # 13 SCENARIOS -- BIG ENDIAN NOTATION FOR THE 13 BASIS GATES TENSORED PRODUCTS
        # ==================================================================================

        # Initialise 4 different initial states
        for k in range(0, 4):
            if k == 0:  # Initial state is |0>
                circ = Circuit(nqubits, density_matrix=True)

            elif k == 1:  # Initial state is |1>
                circ = Circuit(nqubits, density_matrix=True)
                circ.add(gates.X(0))

            elif k == 2:  # Initial state is |+>
                circ = Circuit(nqubits, density_matrix=True)
                circ.add(gates.H(0))

            elif k == 3:  # Initial state is |y+>
                circ = Circuit(nqubits, density_matrix=True)
                circ.add(gates.H(0))
                circ.add(gates.S(0))

            # ==================================================================================
            # Basis operation 0 to 9
            if idx_basis_op < 10:
                circ.add(
                    gates.Unitary(
                        BasisOps_13[idx_basis_op],
                        0,
                        trainable=False,
                        name="%s" % (gatenames[idx_basis_op]),
                    )
                )

                column_of_js = GSTBasisOpMeasurements_13(circ)

            # ==================================================================================
            # Basis operation 10, 11, 12
            elif idx_basis_op == 10:  # Reset, prepare |+> state
                circ.add(
                    gates.ResetChannel(0, [1, 0])
                )  # ResetChannel(qubit, [prob0, prob1])
                circ.add(gates.H(0))
                circ.add(
                    gates.Unitary(
                        BasisOps_13[idx_basis_op],
                        0,
                        trainable=False,
                        name="%s" % (gatenames[idx_basis_op]),
                    )
                )

                column_of_js = GSTBasisOpMeasurements_13(circ)

            elif idx_basis_op == 11:  # Reset, prepare |y+> state
                circ.add(
                    gates.ResetChannel(0, [1, 0])
                )  # ResetChannel(qubit, [prob0, prob1])
                circ.add(gates.H(0))
                circ.add(gates.S(0))
                circ.add(
                    gates.Unitary(
                        BasisOps_13[idx_basis_op],
                        0,
                        trainable=False,
                        name="%s" % (gatenames[idx_basis_op]),
                    )
                )

                column_of_js = GSTBasisOpMeasurements_13(circ)

            elif idx_basis_op == 12:  # Reset, prepare |0> state
                circ.add(
                    gates.ResetChannel(0, [1, 0])
                )  # ResetChannel(qubit, [prob0, prob1])
                circ.add(
                    gates.Unitary(
                        BasisOps_13[idx_basis_op],
                        0,
                        trainable=False,
                        name="%s" % (gatenames[idx_basis_op]),
                    )
                )

                column_of_js = GSTBasisOpMeasurements_13(circ)

            Bjk_tilde_1qb[idx_basis_op, :, k] = column_of_js.reshape(
                4,
            )
        idx_basis_op += 1

    toc = time.time() - tic
    print("13 basis operations tomography done. %.4f seconds\n" % (toc))

    #######################################################################
    ### Get noisy Pauli Transfer Matrix for each of 13 basis operations ###
    #######################################################################

    nqubits = 1
    T = np.matrix([[1, 1, 1, 1], [0, 0, 1, 0], [0, 0, 0, 1], [1, -1, 0, 0]])
    Bjk_hat_1qb = np.zeros((13, 4**nqubits, 4**nqubits))
    for idx_BO in range(0, 13):
        Bjk_hat_1qb[idx_BO, :, :] = (
            T * np.linalg.inv(gjk_1qb) * Bjk_tilde_1qb[idx_BO, :, :] * np.linalg.inv(T)
        )

    # Reshape
    Bjk_hat_1qb_reshaped = np.zeros((16, 13))
    for idx_basis_op in range(0, 13):
        temp = np.reshape(
            Bjk_hat_1qb[idx_basis_op, :, :], [16, 1], order="F"
        )  # 'F' so that it stacks columns.
        Bjk_hat_1qb_reshaped[:, idx_basis_op] = temp[:, 0].flatten()

    return Bjk_hat_1qb, Bjk_hat_1qb_reshaped, BasisOps_13


def GST_2qb_basis_operations(
    NshotsGST=1e4, noise_model=None, gjk_2qb=None, save_data=None, backend=None
):
    """Runs gate set tomography for the 13 basis operations for a single qubit.

        Args:
        NshotsGST (int, optional): Number of shots used in Gate Set Tomography.
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

    import numpy as np

    from qibo import gates
    from qibo.backends import GlobalBackend
    from qibo.models import Circuit

    if backend is None:  # pragma: no cover
        backend = GlobalBackend()

    if gjk_2qb is None:
        gjk_2qb = GST_2qb(NshotsGST, noise_model, save_data=None)

    hadamard = np.matrix(gates.H(0).matrix(backend=GlobalBackend()))
    identity = np.matrix(gates.I(0).matrix(backend=GlobalBackend()))
    xgate = np.matrix(gates.X(0).matrix(backend=GlobalBackend()))
    sgate = np.matrix(gates.S(0).matrix(backend=GlobalBackend()))
    sdggate = np.matrix(gates.SDG(0).matrix(backend=GlobalBackend()))
    CNOT = np.matrix(gates.CNOT(0, 1).matrix(backend=GlobalBackend()))
    SWAP = np.matrix(gates.SWAP(0, 1).matrix(backend=GlobalBackend()))
    ry_piby4 = np.matrix(gates.RY(0, np.pi / 4).matrix(backend=GlobalBackend()))

    # One qubit basis operations
    BO_1qubit = []
    BO_1qubit.append(np.matrix(gates.I(0).matrix(backend=GlobalBackend())))
    BO_1qubit.append(np.matrix(gates.X(0).matrix(backend=GlobalBackend())))
    BO_1qubit.append(np.matrix(gates.Y(0).matrix(backend=GlobalBackend())))
    BO_1qubit.append(np.matrix(gates.Z(0).matrix(backend=GlobalBackend())))
    BO_1qubit.append((1 / np.sqrt(2)) * np.matrix([[1, 1j], [1j, 1]]))
    BO_1qubit.append((1 / np.sqrt(2)) * np.matrix([[1, 1], [-1, 1]]))
    BO_1qubit.append((1 / np.sqrt(2)) * np.matrix([[(1 + 1j), 0], [0, (1 - 1j)]]))
    BO_1qubit.append((1 / np.sqrt(2)) * np.matrix([[1, -1j], [1j, -1]]))
    BO_1qubit.append((1 / np.sqrt(2)) * np.matrix([[1, 1], [1, -1]]))
    BO_1qubit.append((1 / np.sqrt(2)) * np.matrix([[0, (1 - 1j)], [(1 + 1j), 0]]))
    BO_1qubit.append(identity)
    BO_1qubit.append(identity)
    BO_1qubit.append(identity)

    # Intermediate operations: "interm_ops"
    interm_ops = []
    interm_ops.append(CNOT)
    interm_ops.append(np.kron(xgate, identity) * CNOT * np.kron(xgate, identity))
    interm_ops.append(
        np.matrix(np.block([[identity, np.zeros((2, 2))], [np.zeros((2, 2)), sgate]]))
    )
    interm_ops.append(
        np.matrix(
            np.block([[identity, np.zeros((2, 2))], [np.zeros((2, 2)), hadamard]])
        )
    )
    interm_ops.append(np.kron(ry_piby4, identity) * CNOT * np.kron(ry_piby4, identity))
    interm_ops.append(CNOT * np.kron(hadamard, identity))
    interm_ops.append(SWAP)
    interm_ops.append(
        np.matrix([[1, 0, 0, 0], [0, 0, 1j, 0], [0, 1j, 0, 0], [0, 0, 0, 1]])
    )
    interm_ops.append(SWAP * np.kron(hadamard, identity))

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
                np.kron(identity, sgate * hadamard)
                * interm_ops[ii]
                * np.kron(identity, hadamard * sdggate)
            )
            temp_matrix2 = (
                np.kron(identity, hadamard * sdggate)
                * interm_ops[ii]
                * np.kron(identity, sgate * hadamard)
            )
            temp_matrix3 = (
                np.kron(identity, identity)
                * interm_ops[ii]
                * np.kron(identity, identity)
            )

            BO_2qubits_hard.append(temp_matrix1)
            BO_2qubits_hard.append(temp_matrix2)
            BO_2qubits_hard.append(temp_matrix3)

        elif ii == 7:
            # print(f'interm_op {ii}, 6 combinations')
            temp_matrix1 = (
                np.kron(sgate * hadamard, sgate * hadamard)
                * interm_ops[ii]
                * np.kron(hadamard * sdggate, hadamard * sdggate)
            )
            temp_matrix2 = (
                np.kron(sgate * hadamard, hadamard * sdggate)
                * interm_ops[ii]
                * np.kron(hadamard * sdggate, sgate * hadamard)
            )
            temp_matrix3 = (
                np.kron(sgate * hadamard, identity)
                * interm_ops[ii]
                * np.kron(hadamard * sdggate, identity)
            )
            temp_matrix4 = (
                np.kron(identity * hadamard, sgate * hadamard)
                * interm_ops[ii]
                * np.kron(identity * sdggate, hadamard * sdggate)
            )
            temp_matrix5 = (
                np.kron(identity * hadamard, hadamard * sdggate)
                * interm_ops[ii]
                * np.kron(identity * sdggate, sgate * hadamard)
            )
            temp_matrix6 = (
                np.kron(identity, identity)
                * interm_ops[ii]
                * np.kron(identity, identity)
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
                np.kron(sgate * hadamard, sgate * hadamard)
                * interm_ops[ii]
                * np.kron(hadamard * sdggate, hadamard * sdggate)
            )
            temp_matrix2 = (
                np.kron(sgate * hadamard, hadamard * sdggate)
                * interm_ops[ii]
                * np.kron(hadamard * sdggate, sgate * hadamard)
            )
            temp_matrix3 = (
                np.kron(sgate * hadamard, identity)
                * interm_ops[ii]
                * np.kron(hadamard * sdggate, identity)
            )

            temp_matrix4 = (
                np.kron(hadamard * sdggate, sgate * hadamard)
                * interm_ops[ii]
                * np.kron(sgate * hadamard, hadamard * sdggate)
            )
            temp_matrix5 = (
                np.kron(hadamard * sdggate, hadamard * sdggate)
                * interm_ops[ii]
                * np.kron(sgate * hadamard, sgate * hadamard)
            )
            temp_matrix6 = (
                np.kron(hadamard * sdggate, identity)
                * interm_ops[ii]
                * np.kron(sgate * hadamard, identity)
            )

            temp_matrix7 = (
                np.kron(identity, sgate * hadamard)
                * interm_ops[ii]
                * np.kron(identity, hadamard * sdggate)
            )
            temp_matrix8 = (
                np.kron(identity, hadamard * sdggate)
                * interm_ops[ii]
                * np.kron(identity, sgate * hadamard)
            )
            temp_matrix9 = (
                np.kron(identity, identity)
                * interm_ops[ii]
                * np.kron(identity, identity)
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

    # 169 basis operation names
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

    # def sort_counts_2qb(matrix):
    #     counts_matrix = np.zeros((4,2))

    #     for ii in range(0,4):
    #         word = bin(ii)[2:]
    #         counts_matrix[ii,0] = word

    #     for row in matrix:
    #         row_number = int(str(int(row[0])), 2)
    #         counts_matrix[row_number,1] = row[1]

    #     return counts_matrix

    def GST_measure_241(circuit, idx_basis_ops, k, j):
        nqubits = 2
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

        result = circuit.execute(nshots=NshotsGST)
        counts = dict(result.frequencies(binary=True))

        if save_data is not None:
            import os

            filedirectory = "GST_basisops_qibo_2qb/"
            fname = "BasisOp%.3d (k=%.2d,j=%.2d) %s_basis.txt" % (
                idx_basis_op,
                k,
                j,
                idx_basis[j],
            )
            # If location does not exist, create location.
            GST_basisops_qibo_2qb = "GST_basisops_qibo_2qb"
            if not os.path.exists(GST_basisops_qibo_2qb):
                os.makedirs(GST_basisops_qibo_2qb)

            # Save counts into a matrix. 0th column: word; 1st column: count value
            # --------------------------------------------------------------------
            fout = filedirectory + fname
            fo = open(fout, "w")
            for v1, v2 in counts.items():
                fo.write(str(v1) + "  " + str(v2) + "\n")
            fo.close()

        # Find value for Bjk tilde
        # -------------------------

        expectation_signs = np.matrix(
            [[1, 1, 1, 1], [1, -1, 1, -1], [1, 1, -1, -1], [1, -1, -1, 1]]
        )

        matrix = np.zeros((2**nqubits, 2))
        for ii in range(0, 2**nqubits):
            word = bin(ii)[2:]
            matrix[ii, 0] = word

        for v1, v2 in counts.items():
            # print(v1, v2)
            row = int(v1, 2)
            col = v2
            matrix[row, 1] = col

        sorted_matrix = sort_counts_2qb(matrix)
        probs = sorted_matrix[:, 1] / np.sum(sorted_matrix[:, 1])
        sorted_matrix = np.column_stack((sorted_matrix, probs))

        k = k + 1
        j = j + 1

        if j == 1:
            sorted_matrix = np.column_stack((sorted_matrix, expectation_signs[:, 0]))
        elif j == 2:
            sorted_matrix = np.column_stack((sorted_matrix, expectation_signs[:, 1]))
        elif j == 3:
            sorted_matrix = np.column_stack((sorted_matrix, expectation_signs[:, 1]))
        elif j == 4:
            sorted_matrix = np.column_stack((sorted_matrix, expectation_signs[:, 1]))
        elif j == 5:
            sorted_matrix = np.column_stack((sorted_matrix, expectation_signs[:, 2]))
        elif j == 9:
            sorted_matrix = np.column_stack((sorted_matrix, expectation_signs[:, 2]))
        elif j == 13:
            sorted_matrix = np.column_stack((sorted_matrix, expectation_signs[:, 2]))
        else:
            sorted_matrix = np.column_stack((sorted_matrix, expectation_signs[:, 3]))

        temp = np.array(sorted_matrix[:, 2]) * np.array(sorted_matrix[:, 3])
        sorted_matrix = np.column_stack((sorted_matrix, temp))

        expectation_val = np.sum(temp)

        return expectation_val

    def GSTBasisOpMeasurements_241(circ, idx_basis_op):
        nqubits = 2
        # initial_state = np.zeros((2**nqubits))
        # initial_state[0] = 1

        column_of_js = np.zeros((4**nqubits, 1))

        # ================= START OF 16 MEASUREMENT BASES =================|||
        for j in range(0, 4**nqubits):
            # print('[def GSTBasisOpMeasurements]: initial state k=%d, idx_gatenames_1 = %d, idx_gatenames_2 = %d, measurement basis %d' %(k,idx_gatenames_1,idx_gatenames_2,j))

            if j == 0:  # Identity basis  # Identity basis
                circ2 = Circuit(nqubits, density_matrix=True)
                new_circ = circ + circ2
                if noise_model is not None and backend.name != "qibolab":
                    new_circ = noise_model.apply(new_circ)
                new_circ.add(gates.M(0, basis=gates.Z))
                new_circ.add(gates.M(1, basis=gates.Z))

                expectation_val = GST_measure_241(new_circ, idx_basis_op, k, j)

            elif j == 1:  # Identity basis # X basis
                circ2 = Circuit(nqubits, density_matrix=True)
                new_circ = circ + circ2
                if noise_model is not None and backend.name != "qibolab":
                    new_circ = noise_model.apply(new_circ)
                new_circ.add(gates.M(0, basis=gates.Z))
                new_circ.add(gates.M(1, basis=gates.X))

                expectation_val = GST_measure_241(new_circ, idx_basis_op, k, j)

            elif j == 2:  # Identity basis # Y basis
                circ2 = Circuit(nqubits, density_matrix=True)
                new_circ = circ + circ2
                if noise_model is not None and backend.name != "qibolab":
                    new_circ = noise_model.apply(new_circ)
                new_circ.add(gates.M(0, basis=gates.Z))
                new_circ.add(gates.M(1, basis=gates.Y))

                expectation_val = GST_measure_241(new_circ, idx_basis_op, k, j)

            elif j == 3:  # Identity basis # Z basis
                circ2 = Circuit(nqubits, density_matrix=True)
                new_circ = circ + circ2
                if noise_model is not None and backend.name != "qibolab":
                    new_circ = noise_model.apply(new_circ)
                new_circ.add(gates.M(0, basis=gates.Z))
                new_circ.add(gates.M(1, basis=gates.Z))

                expectation_val = GST_measure_241(new_circ, idx_basis_op, k, j)

            # =================================================================================================================

            elif j == 4:  # X basis# Identity basis
                circ2 = Circuit(nqubits, density_matrix=True)
                new_circ = circ + circ2
                if noise_model is not None and backend.name != "qibolab":
                    new_circ = noise_model.apply(new_circ)
                new_circ.add(gates.M(0, basis=gates.X))
                new_circ.add(gates.M(1, basis=gates.Z))

                expectation_val = GST_measure_241(new_circ, idx_basis_op, k, j)

            elif j == 5:  # X basis # X basis
                circ2 = Circuit(nqubits, density_matrix=True)
                new_circ = circ + circ2
                if noise_model is not None and backend.name != "qibolab":
                    new_circ = noise_model.apply(new_circ)
                new_circ.add(gates.M(0, basis=gates.X))
                new_circ.add(gates.M(1, basis=gates.X))

                expectation_val = GST_measure_241(new_circ, idx_basis_op, k, j)

            elif j == 6:  # X basis # Y basis
                circ2 = Circuit(nqubits, density_matrix=True)
                new_circ = circ + circ2
                if noise_model is not None and backend.name != "qibolab":
                    new_circ = noise_model.apply(new_circ)
                new_circ.add(gates.M(0, basis=gates.X))
                new_circ.add(gates.M(1, basis=gates.Y))

                expectation_val = GST_measure_241(new_circ, idx_basis_op, k, j)

            elif j == 7:  # X basis  # Z basis
                circ2 = Circuit(nqubits, density_matrix=True)
                new_circ = circ + circ2
                if noise_model is not None and backend.name != "qibolab":
                    new_circ = noise_model.apply(new_circ)
                new_circ.add(gates.M(0, basis=gates.X))
                new_circ.add(gates.M(1, basis=gates.Z))

                expectation_val = GST_measure_241(new_circ, idx_basis_op, k, j)

            # =================================================================================================================

            elif j == 8:  # Y basis # Identity basis
                circ2 = Circuit(nqubits, density_matrix=True)
                new_circ = circ + circ2
                if noise_model is not None and backend.name != "qibolab":
                    new_circ = noise_model.apply(new_circ)
                new_circ.add(gates.M(0, basis=gates.Y))
                new_circ.add(gates.M(1, basis=gates.Z))

                expectation_val = GST_measure_241(new_circ, idx_basis_op, k, j)

            elif j == 9:  # Y basis  # X basis
                circ2 = Circuit(nqubits, density_matrix=True)
                new_circ = circ + circ2
                if noise_model is not None and backend.name != "qibolab":
                    new_circ = noise_model.apply(new_circ)
                new_circ.add(gates.M(0, basis=gates.Y))
                new_circ.add(gates.M(1, basis=gates.X))

                expectation_val = GST_measure_241(new_circ, idx_basis_op, k, j)

            elif j == 10:  # Y basis  # Y basis
                circ2 = Circuit(nqubits, density_matrix=True)
                new_circ = circ + circ2
                if noise_model is not None and backend.name != "qibolab":
                    new_circ = noise_model.apply(new_circ)
                new_circ.add(gates.M(0, basis=gates.Y))
                new_circ.add(gates.M(1, basis=gates.Y))

                expectation_val = GST_measure_241(new_circ, idx_basis_op, k, j)

            elif j == 11:  # Y basis  # Z basis
                circ2 = Circuit(nqubits, density_matrix=True)
                new_circ = circ + circ2
                if noise_model is not None and backend.name != "qibolab":
                    new_circ = noise_model.apply(new_circ)
                new_circ.add(gates.M(0, basis=gates.Y))
                new_circ.add(gates.M(1, basis=gates.Z))

                expectation_val = GST_measure_241(new_circ, idx_basis_op, k, j)

            # =================================================================================================================

            elif j == 12:  # Z basis# Identity basis
                circ2 = Circuit(nqubits, density_matrix=True)
                new_circ = circ + circ2
                if noise_model is not None and backend.name != "qibolab":
                    new_circ = noise_model.apply(new_circ)
                new_circ.add(gates.M(0, basis=gates.Z))
                new_circ.add(gates.M(1, basis=gates.Z))

                expectation_val = GST_measure_241(new_circ, idx_basis_op, k, j)

            elif j == 13:  # Z basis  # X basis
                circ2 = Circuit(nqubits, density_matrix=True)
                new_circ = circ + circ2
                if noise_model is not None and backend.name != "qibolab":
                    new_circ = noise_model.apply(new_circ)
                new_circ.add(gates.M(0, basis=gates.Z))
                new_circ.add(gates.M(1, basis=gates.X))

                expectation_val = GST_measure_241(new_circ, idx_basis_op, k, j)

            elif j == 14:  # Z basis  # Y basis
                circ2 = Circuit(nqubits, density_matrix=True)
                new_circ = circ + circ2
                if noise_model is not None and backend.name != "qibolab":
                    new_circ = noise_model.apply(new_circ)
                new_circ.add(gates.M(0, basis=gates.Z))
                new_circ.add(gates.M(1, basis=gates.Y))

                expectation_val = GST_measure_241(new_circ, idx_basis_op, k, j)

            elif j == 15:  # Z basis # Z basis
                circ2 = Circuit(nqubits, density_matrix=True)
                new_circ = circ + circ2
                if noise_model is not None and backend.name != "qibolab":
                    new_circ = noise_model.apply(new_circ)
                new_circ.add(gates.M(0, basis=gates.Z))
                new_circ.add(gates.M(1, basis=gates.Z))

                expectation_val = GST_measure_241(new_circ, idx_basis_op, k, j)

            column_of_js[j, :] = expectation_val

        return column_of_js

    ####################################
    ### 169 scenarios # (2020-12-05) ###
    ####################################

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
    idx_basis_op = 0
    nqubits = 2

    Bjk_tilde_2qb = np.zeros((241, 16, 16))

    for idx_gatenames_1 in range(0, 13):
        for idx_gatenames_2 in range(0, 13):
            # print('GST 2qb 241 Basis Op. Basis Op idx = %d (top reg idx = %d, bottom reg idx = %d)' %(idx_BO, idx_gatenames_1, idx_gatenames_2))
            print("GST 2qb 241 Basis Op. Basis Op idx = %d" % (idx_basis_op))

            # =========================================#
            # 16 SCENARIOS  #  13^2 BASIS OPERATIONS  #  -- BIG ENDIAN NOTATION FOR THE 13 BASIS GATES TENSORED PRODUCTS
            # ==================================================================================

            # Initialise 16 different initial states
            for k in range(0, 16):
                #     print('Tomography of initial state %.2d' %(k))
                if k == 0:  # |0> |0>
                    circ = Circuit(nqubits, density_matrix=True)

                elif k == 1:  # |0> |1>
                    circ = Circuit(nqubits, density_matrix=True)
                    circ.add(gates.X(1))

                elif k == 2:  # |0> |+>
                    circ = Circuit(nqubits, density_matrix=True)
                    circ.add(gates.H(1))

                elif k == 3:  # |0> |y+>
                    circ = Circuit(nqubits, density_matrix=True)
                    circ.add(gates.H(1))
                    circ.add(gates.S(1))
                # --------------------------------------------

                elif k == 4:  # |1> |0>
                    circ = Circuit(nqubits, density_matrix=True)
                    circ.add(gates.X(0))

                elif k == 5:  # |1> |1>
                    circ = Circuit(nqubits, density_matrix=True)
                    circ.add(gates.X(0))
                    circ.add(gates.X(1))

                elif k == 6:  # |1> |+>
                    circ = Circuit(nqubits, density_matrix=True)
                    circ.add(gates.X(0))
                    circ.add(gates.H(1))

                elif k == 7:  # |1> |y+>
                    circ = Circuit(nqubits, density_matrix=True)
                    circ.add(gates.X(0))
                    circ.add(gates.H(1))
                    circ.add(gates.S(1))

                # --------------------------------------------

                elif k == 8:  # |+> |0>
                    circ = Circuit(nqubits, density_matrix=True)
                    circ.add(gates.H(0))

                elif k == 9:  # |+> |1>
                    circ = Circuit(nqubits, density_matrix=True)
                    circ.add(gates.H(0))
                    circ.add(gates.X(1))

                elif k == 10:  # |+> |+>
                    circ = Circuit(nqubits, density_matrix=True)
                    circ.add(gates.H(0))
                    circ.add(gates.H(1))

                elif k == 11:  # |+> |y+>
                    circ = Circuit(nqubits, density_matrix=True)
                    circ.add(gates.H(0))
                    circ.add(gates.H(1))
                    circ.add(gates.S(1))

                # --------------------------------------------

                elif k == 12:  # |y+> |0>
                    circ = Circuit(nqubits, density_matrix=True)
                    circ.add(gates.H(0))
                    circ.add(gates.S(0))

                elif k == 13:  # |y+> |1>
                    circ = Circuit(nqubits, density_matrix=True)
                    circ.add(gates.H(0))
                    circ.add(gates.S(0))
                    circ.add(gates.X(1))

                elif k == 14:  # |y+> |+>
                    circ = Circuit(nqubits, density_matrix=True)
                    circ.add(gates.H(0))
                    circ.add(gates.S(0))
                    circ.add(gates.H(1))

                elif k == 15:  # |y+> |y+>
                    circ = Circuit(nqubits, density_matrix=True)
                    circ.add(gates.H(0))
                    circ.add(gates.S(0))
                    circ.add(gates.H(1))
                    circ.add(gates.S(1))

                # --------------------------------------------

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

                    column_of_js = GSTBasisOpMeasurements_241(circ, idx_basis_op)

                # ==================================================================================

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

                    column_of_js = GSTBasisOpMeasurements_241(circ, idx_basis_op)

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

                    column_of_js = GSTBasisOpMeasurements_241(circ, idx_basis_op)

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

                    column_of_js = GSTBasisOpMeasurements_241(circ, idx_basis_op)

                # ==================================================================================

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

                    column_of_js = GSTBasisOpMeasurements_241(circ, idx_basis_op)

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

                    column_of_js = GSTBasisOpMeasurements_241(circ, idx_basis_op)

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

                    column_of_js = GSTBasisOpMeasurements_241(circ, idx_basis_op)

                # ==================================================================================

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

                    column_of_js = GSTBasisOpMeasurements_241(circ, idx_basis_op)

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

                    column_of_js = GSTBasisOpMeasurements_241(circ, idx_basis_op)

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

                    column_of_js = GSTBasisOpMeasurements_241(circ, idx_basis_op)

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

                    column_of_js = GSTBasisOpMeasurements_241(circ, idx_basis_op)

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

                    column_of_js = GSTBasisOpMeasurements_241(circ, idx_basis_op)

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

                    column_of_js = GSTBasisOpMeasurements_241(circ, idx_basis_op)

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

                    column_of_js = GSTBasisOpMeasurements_241(circ, idx_basis_op)

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

                    column_of_js = GSTBasisOpMeasurements_241(circ, idx_basis_op)

                elif idx_gatenames_1 == 12 and idx_gatenames_2 == 12:
                    # qubit_1 = reinitialised |0> state
                    # qubit_2 = reinitialised |0> state
                    circ.add(
                        gates.ResetChannel(0, [1, 0])
                    )  # ResetChannel(qubit, [prob0, prob1])
                    circ.add(
                        gates.ResetChannel(1, [1, 0])
                    )  # ResetChannel(qubit, [prob0, prob1])

                    column_of_js = GSTBasisOpMeasurements_241(circ, idx_basis_op)

                # ==================================================================================

                Bjk_tilde_2qb[idx_basis_op, :, k] = column_of_js.reshape(
                    16,
                )

            idx_basis_op += 1

    toc = time.time() - tic
    print("0-168 basis operations tomography done. %.4f seconds\n" % (toc))

    ###################################
    ### 72 scenarios # (2020-12-05) ###
    ###################################
    nqubits = 2

    tic = time.time()
    idx_basis_op = 169
    for idx_basis_ops_72 in range(169, 241):
        # print('GST 2qb 241 Basis Op. Basis Op idx = %d [%d]' %(idx_basis_ops_72, idx_basis_ops_72-169))
        print("GST 2qb 241 Basis Op. Basis Op idx = %d" % (idx_basis_ops_72))

        for k in range(0, 16):
            if k == 0:  # |0> |0>
                circ = Circuit(nqubits, density_matrix=True)

            elif k == 1:  # |0> |1>
                circ = Circuit(nqubits, density_matrix=True)
                circ.add(gates.X(1))

            elif k == 2:  # |0> |+>
                circ = Circuit(nqubits, density_matrix=True)
                circ.add(gates.H(1))

            elif k == 3:  # |0> |y+>
                circ = Circuit(nqubits, density_matrix=True)
                circ.add(gates.H(1))
                circ.add(gates.S(1))
            # --------------------------------------------

            elif k == 4:  # |1> |0>
                circ = Circuit(nqubits, density_matrix=True)
                circ.add(gates.X(0))

            elif k == 5:  # |1> |1>
                circ = Circuit(nqubits, density_matrix=True)
                circ.add(gates.X(0))
                circ.add(gates.X(1))

            elif k == 6:  # |1> |+>
                circ = Circuit(nqubits, density_matrix=True)
                circ.add(gates.X(0))
                circ.add(gates.H(1))

            elif k == 7:  # |1> |y+>
                circ = Circuit(nqubits, density_matrix=True)
                circ.add(gates.X(0))
                circ.add(gates.H(1))
                circ.add(gates.S(1))

            # --------------------------------------------

            elif k == 8:  # |+> |0>
                circ = Circuit(nqubits, density_matrix=True)
                circ.add(gates.H(0))

            elif k == 9:  # |+> |1>
                circ = Circuit(nqubits, density_matrix=True)
                circ.add(gates.H(0))
                circ.add(gates.X(1))

            elif k == 10:  # |+> |+>
                circ = Circuit(nqubits, density_matrix=True)
                circ.add(gates.H(0))
                circ.add(gates.H(1))

            elif k == 11:  # |+> |y+>
                circ = Circuit(nqubits, density_matrix=True)
                circ.add(gates.H(0))
                circ.add(gates.H(1))
                circ.add(gates.S(1))

            # --------------------------------------------

            elif k == 12:  # |y+> |0>
                circ = Circuit(nqubits, density_matrix=True)
                circ.add(gates.H(0))
                circ.add(gates.S(0))

            elif k == 13:  # |y+> |1>
                circ = Circuit(nqubits, density_matrix=True)
                circ.add(gates.H(0))
                circ.add(gates.S(0))
                circ.add(gates.X(1))

            elif k == 14:  # |y+> |+>
                circ = Circuit(nqubits, density_matrix=True)
                circ.add(gates.H(0))
                circ.add(gates.S(0))
                circ.add(gates.H(1))

            elif k == 15:  # |y+> |y+>
                circ = Circuit(nqubits, density_matrix=True)
                circ.add(gates.H(0))
                circ.add(gates.S(0))
                circ.add(gates.H(1))
                circ.add(gates.S(1))

            # --------------------------------------------

            # INPUT BASIS OPERATION
            circ.add(
                gates.Unitary(
                    BasisOps_241[idx_basis_ops_72],
                    1,
                    0,
                    trainable=False,
                    name="%s" % (idx_basis_ops_72),
                )
            )
            # print('outside GST_measure_241: \n', circ.draw())
            column_of_js = GSTBasisOpMeasurements_241(circ, idx_basis_op)

            Bjk_tilde_2qb[idx_basis_op, :, k] = column_of_js.reshape(
                16,
            )

        idx_basis_op += 1

    toc = time.time() - tic
    print("169-240 basis operations tomography done. %.4f seconds\n" % (toc))

    # ########################################################################
    # ### Get noisy Pauli Transfer Matrix for each of 241 basis operations ###
    # ########################################################################

    nqubits = 2
    T = np.matrix([[1, 1, 1, 1], [0, 0, 1, 0], [0, 0, 0, 1], [1, -1, 0, 0]])
    Bjk_hat_2qb = np.zeros((241, 4**nqubits, 4**nqubits))
    for idx_BO in range(0, 241):
        Bjk_hat_2qb[idx_BO, :, :] = (
            np.kron(T, T)
            * np.linalg.inv(gjk_2qb)
            * Bjk_tilde_2qb[idx_BO, :, :]
            * np.linalg.inv(np.kron(T, T))
        )

    # Reshape
    Bjk_hat_2qb_reshaped = np.zeros((256, 241))
    for idx_basis_op in range(0, 241):
        temp = np.reshape(
            Bjk_hat_2qb[idx_basis_op, :, :], [256, 1], order="F"
        )  # 'F' so that it stacks columns.
        Bjk_hat_2qb_reshaped[:, idx_basis_op] = temp[:, 0].flatten()

    return Bjk_hat_2qb, Bjk_hat_2qb_reshaped, BasisOps_241
