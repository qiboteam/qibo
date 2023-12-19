import numpy as np
from qibo.models import Circuit
from qibo import gates
from qibo.backends import GlobalBackend

def prepare_states(k, nqubits=None):
    
    if nqubits is not None:
        if nqubits == 1:
            k_to_gates_1qb = {
                0: (),
                1: (gates.X(0), ),
                2: (gates.H(0), ),
                3: (gates.H(0), gates.S(0))
            }
            
            circ = Circuit(nqubits, density_matrix=True)
            for gate in k_to_gates_1qb[k]:
                circ.add(gate)
                
            return circ
            
        elif nqubits == 2:
            k_to_gates_2qb = {
                0: (),
                1: (gates.X(1), ),
                2: (gates.H(1), ),
                3: (gates.H(1), gates.S(1)),
                4: (gates.X(0), ),
                5: (gates.X(0), gates.X(1), ),
                6: (gates.X(0), gates.H(1), ),
                7: (gates.X(0), gates.H(1), gates.S(1)),
                8: (gates.H(0), ),
                9: (gates.H(0), gates.X(1), ),
                10: (gates.H(0), gates.H(1), ),
                11: (gates.H(0), gates.H(1), gates.S(1)),
                12: (gates.H(0), gates.S(0), ),
                13: (gates.H(0), gates.S(0), gates.X(1), ),
                14: (gates.H(0), gates.S(0), gates.H(1), ),
                15: (gates.H(0), gates.S(0), gates.H(1), gates.S(1))
            }
            
            circ = Circuit(nqubits, density_matrix=True)
            for gate in k_to_gates_2qb[k]:
                circ.add(gate)
                
            return circ

        else:
            raise ValueError(f"nqubits given as {nqubits}. nqubits needs to be either 1 or 2.")    
        
    else:
        raise ValueError(f"nqubits given as {nqubits}. nqubits needs to be either 1 or 2.")


def measurement_basis(j, circ, nqubits=None):
    if nqubits is not None:
        if nqubits == 1:
            j_to_measurements_1qb = {
                0: (gates.M(0, basis=gates.Z), ),
                1: (gates.M(0, basis=gates.X), ),
                2: (gates.M(0, basis=gates.Y), ),
                3: (gates.M(0, basis=gates.Z), ),
            }
        
            new_circ = circ.copy()
        
            for meas in j_to_measurements_1qb[j]:
                new_circ.add(meas)

            return new_circ

        elif nqubits == 2:
            j_to_measurements_2qb = {
                0: (gates.M(0, basis=gates.Z), gates.M(1, basis=gates.Z), ),
                1: (gates.M(0, basis=gates.Z), gates.M(1, basis=gates.X), ),
                2: (gates.M(0, basis=gates.Z), gates.M(1, basis=gates.Y), ),
                3: (gates.M(0, basis=gates.Z), gates.M(1, basis=gates.Z), ),
                4: (gates.M(0, basis=gates.X), gates.M(1, basis=gates.Z), ),
                5: (gates.M(0, basis=gates.X), gates.M(1, basis=gates.X), ),
                6: (gates.M(0, basis=gates.X), gates.M(1, basis=gates.Y), ),
                7: (gates.M(0, basis=gates.X), gates.M(1, basis=gates.Z), ),
                8: (gates.M(0, basis=gates.Y), gates.M(1, basis=gates.Z), ),
                9: (gates.M(0, basis=gates.Y), gates.M(1, basis=gates.X), ),
                10: (gates.M(0, basis=gates.Y), gates.M(1, basis=gates.Y), ),
                11: (gates.M(0, basis=gates.Y), gates.M(1, basis=gates.Z), ),
                12: (gates.M(0, basis=gates.Z), gates.M(1, basis=gates.Z), ),
                13: (gates.M(0, basis=gates.Z), gates.M(1, basis=gates.X), ),
                14: (gates.M(0, basis=gates.Z), gates.M(1, basis=gates.Y), ),
                15: (gates.M(0, basis=gates.Z), gates.M(1, basis=gates.Z), ),
            }
        
            new_circ = circ.copy()
        
            for meas in j_to_measurements_2qb[j]:
                new_circ.add(meas)
                
            return new_circ
            
        else:
            raise ValueError(f"nqubits given as {nqubits}. nqubits needs to be either 1 or 2.")    
                
    else:
        raise ValueError(f"nqubits given as {nqubits}. nqubits needs to be either 1 or 2.")    


def GST_execute_circuit(circuit, k, j, nshots=int(1e4), nqubits=None):
    """Executes a circuit used in gate set tomography and processes the
        measurement outcomes for the Pauli Transfer Matrix notation. The circuit
        should already have noise models implemented, if any, prior to using this
        function.
    
        Args:
        circuit: qibo circuit
        k (int): The index of the state prepared.
        j (int): The index of the measurement basis.
        nshots (int, optional): Number of shots to execute circuit with.

    Returns:
        numpy.float: Expectation value given by either Tr(Q_j rho_k) or Tr(Q_j O_l rho_k).
    """

    
    if nqubits == 1:
        #idx_basis = ['I', 'X', 'Y', 'Z']
        expectation_signs = np.array([[1,1,1,1], [1,-1,-1,-1]])
        matrix = np.array([[0, 0],[1, 0]])

    elif nqubits == 2:
        #idx_basis = ['II','IX','IY','IZ', 'XI','XX','XY','XZ', 'YI','YX','YY','YZ', 'ZI','ZX','ZY','ZZ']
        expectation_signs = np.array([[1,1,1,1], [1,-1,1,-1], [1,1,-1,-1], [1,-1,-1,1]])
        matrix = np.array([[0, 0],[1, 0], [10, 0], [11, 0]])
    
    else:
        raise ValueError(f"nqubits given as {nqubits}. nqubits needs to be either 1 or 2.")

    result = circuit.execute(nshots=nshots)
    counts = result.frequencies(binary=True)
    
    # Find value for matrix_jk 
    for v1, v2 in counts.items():
        row = int(v1,2)
        col = v2
        matrix[row,1] = col

    probs = matrix[:,1] / np.sum(matrix[:,1])
    matrix = np.hstack((matrix, probs.reshape(-1,1)))
    
    k = k+1
    j = j+1

    if j==1:
        matrix = np.hstack((matrix, expectation_signs[:,0].reshape(-1,1)))
    elif j==2 or j==3 or j==4:
        matrix = np.hstack((matrix, expectation_signs[:,1].reshape(-1,1)))
    elif j==5 or j==9 or j==13:
        matrix = np.hstack((matrix, expectation_signs[:,2].reshape(-1,1)))
    else:
        matrix = np.hstack((matrix, expectation_signs[:,3].reshape(-1,1)))

    temp = matrix[:,2] * matrix[:,3]
    matrix = np.hstack((matrix, temp.reshape(-1,1)))
    expectation_val = np.sum(temp)

    return expectation_val


def GST(nqubits=None, nshots=int(1e4), noise_model=None, class_qubit_gate=None, ctrl_qb=None, targ_qb=None, theta=None, operator_name=None, backend=None):
    """Runs gate set tomography for a single qubit gate/circuit.
    
        Args:
        nshots (int, optional): Number of shots used in Gate Set Tomography.
        noise_model (:class:`qibo.noise.NoiseModel`, optional): Noise model applied
            to simulate noisy computation.
        save_data: To save gate set tomography data. If None, skip.
        class_qubit_gate: Identifier for the type of gate given in circuit.raw. If None,
            do gate set tomography with empty circuit.
        theta: Identifier for the gate parameter given in circuit.raw. If None, do gate
            set tomography with only gate provided by class_qubit_gate.
        operator_name: Name of gate given in circuit.raw used for saving data.
        backend (:class:`qibo.backends.abstract.Backend`, optional): Calculation engine.

    Returns:
        numpy.matrix: Matrix with elements jk equivalent to either Tr(Q_j rho_k) or Tr(Q_j O_l rho_k).
    """

    # Check if class_qubit_gate corresponds to nqubits-gate.
    if class_qubit_gate is not None:
        if ctrl_qb is not None and targ_qb is not None:
            gate = getattr(gates, class_qubit_gate)(targ_qb[0], ctrl_qb[0]).matrix()
            qb_gate = int(np.log2(gate.shape[0]))
            if nqubits != qb_gate:
                raise ValueError(f"Mismatched inputs: nqubits given as {nqubits}. {class_qubit_gate} is a {qb_gate}-qubit gate.")

        else:
            gate = getattr(gates, class_qubit_gate)(0).matrix()
            qb_gate = int(np.log2(gate.shape[0]))
            if nqubits != qb_gate:
                raise ValueError(f"Mismatched inputs: nqubits given as {nqubits}. {class_qubit_gate} is a {qb_gate}-qubit gate.")
            
            

    
    # GST for empty circuit or operators
    if nqubits is not None:
        
        if backend is None:  # pragma: no cover
            backend = GlobalBackend()
        
        if nqubits == 1:
            matrix_jk = np.zeros((4, 4))
            for k in range(0,4):
        
                # Prepare state using "prepare_1qb_state(k)"
                circ = prepare_states(k, nqubits)
            
                # If class_qubit_gate is present (with/without theta)
                if class_qubit_gate is not None:
                    if theta is None:
                        circ.add(getattr(gates, class_qubit_gate)(0))
                    elif theta is not None:
                        circ.add(getattr(gates, class_qubit_gate)(0, theta))
                    
                # Measurements
                for j in range(0,4):
                    new_circ = measurement_basis(j, circ, nqubits)
                    if noise_model is not None and backend.name != "qibolab":
                        new_circ = noise_model.apply(new_circ)

                    expectation_val = GST_execute_circuit(new_circ, k, j, nshots, nqubits)
                    matrix_jk[j, k] = expectation_val

            return matrix_jk
            
        elif nqubits == 2:
            matrix_jk = np.zeros((16, 16))
            for k in range(0,16):
        
                # Prepare states using function "prepare_2qb_states(k)".
                circ = prepare_states(k, nqubits)
                
                # If class_qubit_gate is present (with/without theta).
                if class_qubit_gate is not None:
                    if theta is None:
                        circ.add(getattr(gates, class_qubit_gate)(targ_qb[0], ctrl_qb[0]))
                    else:
                        circ.add(getattr(gates, class_qubit_gate)(targ_qb[0], ctrl_qb[0], theta))
            
                # Measurements
                for j in range(0,16):
                    new_circ = measurement_basis(j, circ, nqubits)
                    if noise_model is not None and backend.name != "qibolab":
                        new_circ = noise_model.apply(new_circ)
                    
                    expectation_val = GST_execute_circuit(new_circ, k, j, nshots, nqubits)
                    matrix_jk[j, k] = expectation_val
            
            return matrix_jk

        else:
            raise ValueError(f"nqubits given as {nqubits}. nqubits needs to be either 1 or 2.")

    else:
        raise ValueError(f"nqubits given as {nqubits}. nqubits needs to be either 1 or 2.")


def GST_basis_operations(nqubits=None, nshots=int(1e4), noise_model=None, gjk=None, backend=None):
    """Runs gate set tomography for the basis operations. There are 13 basis operations for a single
        qubit and 241 basis operations for 2 qubits.

        Args:
        nshots (int, optional): Number of shots used in Gate Set Tomography.
        noise_model (:class:`qibo.noise.NoiseModel`, optional): Noise model applied
            to simulate noisy computation.
        gjk (numpy.matrix): Matrix with elements Tr(Q_j rho_k) for with no gate.
            If None, call GST without gate.
        save_data: Flag to save gate set tomography data. If None, skip.
        backend (:class:`qibo.backends.abstract.Backend`, optional): Calculation engine.

        The 13 and 241 basis operations are based on Physical Review Research 3, 033178 (2021).
    Returns:
        numpy.matrix: Matrix with elements jk equivalent to either Tr(Q_j rho_k) or Tr(Q_j O_l rho_k).
    """
    
    if backend is None:  # pragma: no cover
        backend = GlobalBackend()


    # Implement a check. If nqubits==1, gjk needs to be 4x4 matrix. If nqubits==2, gjk needs to be 16x16 matrix.
    if gjk is not None:
        if gjk.shape[0] != 4**nqubits and gjk.shape[1] != 4**nqubits:
            gjk = GST(nqubits, nshots, noise_model) 


    # GST for basis operations.
    if nqubits is not None:
        if nqubits == 1:
            if gjk is None:
                gjk = GST(nqubits, nshots, noise_model) 

            class_qubit_gate = None
            theta = None
            
            BasisOps_13 = [gates.I(0).matrix(backend=backend),
                       
                           gates.X(0).matrix(backend=backend),
                           gates.Y(0).matrix(backend=backend),
                           gates.Z(0).matrix(backend=backend),
                           
                           1/np.sqrt(2) * np.array([[1, 1j], [1j, 1]]),
                           1/np.sqrt(2) * np.array([[1, 1], [-1, 1]]),
                           1/np.sqrt(2) * np.array([[(1+1j), 0],[0, (1-1j)]]),
                           
                           1/np.sqrt(2) * np.array([[1, -1j], [1j, -1]]),
                           gates.H(0).matrix(backend=GlobalBackend()),
                           1/np.sqrt(2) * np.array([[0, (1-1j)], [(1+1j), 0]]),
                           
                           np.array([[1,0],[0,0]]),
                           np.array([[0.5,0.5],[0.5,0.5]]),
                           np.array([[0.5,-0.5j],[0.5j,0.5]])]
            
        
            gatenames = 'BasisOp00', 'BasisOp01', 'BasisOp02', 'BasisOp03', 'BasisOp04', 'BasisOp05', 'BasisOp06', 'BasisOp07', 'BasisOp08', 'BasisOp09', 'BasisOp10', 'BasisOp11', 'BasisOp12'
            
            Bjk_tilde_1qb = np.zeros((13,4,4))    
            for idx_basis_ops in range(0,13):
            
                # Initialise 4 different initial states
                for k in range(0,4):
                    
                    circ = prepare_states(k, nqubits)
                    
                    #==================================================================================
                    # Basis operation 0 to 9
                    if  idx_basis_ops < 10:
                        circ.add(gates.Unitary(BasisOps_13[idx_basis_ops], 0, trainable=False, name='%s' %(gatenames[idx_basis_ops])))
            
                    #==================================================================================
                    # Basis operation 10, 11, 12        
                    elif idx_basis_ops == 10: # Reset, prepare |+> state
                        circ.add(gates.ResetChannel(0, [1,0])) # ResetChannel(qubit, [prob0, prob1])
                        circ.add(gates.H(0))
                        circ.add(gates.Unitary(BasisOps_13[idx_basis_ops], 0, trainable=False, name='%s' %(gatenames[idx_basis_ops])))
                    
                    elif idx_basis_ops == 11:  # Reset, prepare |y+> state    
                        circ.add(gates.ResetChannel(0, [1,0])) # ResetChannel(qubit, [prob0, prob1])
                        circ.add(gates.H(0))
                        circ.add(gates.S(0))
                        circ.add(gates.Unitary(BasisOps_13[idx_basis_ops], 0, trainable=False, name='%s' %(gatenames[idx_basis_ops]))) 
                        
                    elif idx_basis_ops == 12: # Reset, prepare |0> state
                        circ.add(gates.ResetChannel(0, [1,0])) # ResetChannel(qubit, [prob0, prob1])
                        circ.add(gates.Unitary(BasisOps_13[idx_basis_ops], 0, trainable=False, name='%s' %(gatenames[idx_basis_ops])))
            
                    column_of_js = np.zeros((4,1))
                    
                    #================= START OF 4 MEASUREMENT BASES =================|||
                    for j in range(0,4):
                        new_circ = measurement_basis(j, circ, nqubits)
                        expectation_val = GST_execute_circuit(new_circ, k, j, nshots, nqubits)
                        column_of_js[j,:] = expectation_val
        
                    
                    Bjk_tilde_1qb[idx_basis_ops, :, k] = column_of_js.reshape(4,)
                idx_basis_ops += 1
                
            # Get noisy Pauli Transfer Matrix 
            T = np.array([[1,1,1,1],[0,0,1,0],[0,0,0,1],[1,-1,0,0]])
            Bjk_hat_1qb = np.zeros((13,4,4))
            for idx_BO in range(0,13):
                Bjk_hat_1qb[idx_BO, :, :] = T @ np.linalg.inv(gjk) @ Bjk_tilde_1qb[idx_BO, :, :] @ np.linalg.inv(T);
            
            # Reshape
            Bjk_hat_1qb_reshaped = np.zeros((16,13))
            for idx_basis_ops in range(0,13):
                temp = np.reshape(Bjk_hat_1qb[idx_basis_ops,:,:], [16,1], order='F') # 'F' so that it stacks columns. 
                Bjk_hat_1qb_reshaped[:,idx_basis_ops] = temp[:,0].flatten()
        
            return Bjk_hat_1qb, Bjk_hat_1qb_reshaped, BasisOps_13

        elif nqubits == 2:
            if gjk is None:
                gjk = GST(nqubits, nshots, noise_model) 
        
            class_qubit_gate = None
            theta = None
            
            hadamard = gates.H(0).matrix(backend=backend)
            identity = gates.I(0).matrix(backend=backend)
            xgate    = gates.X(0).matrix(backend=backend)
            sgate    = gates.S(0).matrix(backend=backend)
            sdggate  = gates.SDG(0).matrix(backend=backend)
            CNOT     = gates.CNOT(0,1).matrix(backend=backend)
            SWAP     = gates.SWAP(0,1).matrix(backend=backend)
            ry_piby4 = gates.RY(0, np.pi/4).matrix(backend=backend)
            
            # One qubit basis operations
            BO_1qubit = []
            BO_1qubit.append(gates.I(0).matrix(backend=backend))
            BO_1qubit.append(gates.X(0).matrix(backend=backend))
            BO_1qubit.append(gates.Y(0).matrix(backend=backend))
            BO_1qubit.append(gates.Z(0).matrix(backend=backend))
            BO_1qubit.append((1/np.sqrt(2)) *np.array([[1,1j],[1j,1]]))
            BO_1qubit.append((1/np.sqrt(2)) *np.array([[1,1],[-1,1]]))
            BO_1qubit.append((1/np.sqrt(2)) *np.array([[(1+1j),0],[0,(1-1j)]]))
            BO_1qubit.append((1/np.sqrt(2)) *np.array([[1,-1j],[1j,-1]]))
            BO_1qubit.append(gates.H(0).matrix(backend=backend))
            BO_1qubit.append((1/np.sqrt(2)) *np.array([[0, (1-1j)],[(1+1j), 0]]))
            BO_1qubit.append(identity)
            BO_1qubit.append(identity)
            BO_1qubit.append(identity)
        
            # Intermediate operations: "interm_ops"
            interm_ops = []
            interm_ops.append(CNOT)
            interm_ops.append(np.kron(xgate, identity) @ CNOT @ np.kron(xgate, identity))
            interm_ops.append(np.matrix(np.block([[identity, np.zeros((2,2))],[np.zeros((2,2)), sgate]])))
            interm_ops.append(np.matrix(np.block([[identity, np.zeros((2,2))],[np.zeros((2,2)), hadamard]])))
            interm_ops.append(np.kron(ry_piby4, identity) @ CNOT @ np.kron(ry_piby4, identity))
            interm_ops.append(CNOT @ np.kron(hadamard, identity))
            interm_ops.append(SWAP)
            interm_ops.append(np.matrix([[1,0,0,0],[0,0,1j,0],[0,1j,0,0],[0,0,0,1]]))
            interm_ops.append(SWAP @ np.kron(hadamard, identity))
            
            # Two qubit basis operations
            # "Easy"
            BO_2qubits_easy = []
            for ii in range(0,13):
                for jj in range(0,13):
                    temp_matrix = np.kron(BO_1qubit[ii], BO_1qubit[jj])
                    BO_2qubits_easy.append(temp_matrix)
                    
            # "Hard"
            BO_2qubits_hard = []
            for ii in range(0,9):
                if ii == 6:
                    temp_matrix1 = np.kron(identity, sgate @ hadamard  ) @ interm_ops[ii] @ np.kron(identity, hadamard @ sdggate)
                    temp_matrix2 = np.kron(identity, hadamard @ sdggate) @ interm_ops[ii] @ np.kron(identity, sgate @ hadamard  )
                    temp_matrix3 = np.kron(identity, identity          ) @ interm_ops[ii] @ np.kron(identity, identity          )
            
                    BO_2qubits_hard.append(temp_matrix1)
                    BO_2qubits_hard.append(temp_matrix2)
                    BO_2qubits_hard.append(temp_matrix3)
                    
                elif ii == 7:
                    temp_matrix1 = np.kron(sgate @ hadamard   , sgate @ hadamard  ) @ interm_ops[ii] @ np.kron(hadamard @ sdggate, hadamard @ sdggate)
                    temp_matrix2 = np.kron(sgate @ hadamard   , hadamard @ sdggate) @ interm_ops[ii] @ np.kron(hadamard @ sdggate, sgate @ hadamard  )
                    temp_matrix3 = np.kron(sgate @ hadamard   , identity          ) @ interm_ops[ii] @ np.kron(hadamard @ sdggate, identity          )
                    temp_matrix4 = np.kron(identity @ hadamard, sgate @ hadamard  ) @ interm_ops[ii] @ np.kron(identity @ sdggate, hadamard @ sdggate)
                    temp_matrix5 = np.kron(identity @ hadamard, hadamard @ sdggate) @ interm_ops[ii] @ np.kron(identity @ sdggate, sgate @ hadamard  )
                    temp_matrix6 = np.kron(identity           , identity          ) @ interm_ops[ii] @ np.kron(identity          , identity          )
                    
                    BO_2qubits_hard.append(temp_matrix1)
                    BO_2qubits_hard.append(temp_matrix2)
                    BO_2qubits_hard.append(temp_matrix3)
                    BO_2qubits_hard.append(temp_matrix4)
                    BO_2qubits_hard.append(temp_matrix5)
                    BO_2qubits_hard.append(temp_matrix6)
                else:
                    temp_matrix1 = np.kron(sgate @ hadamard  , sgate @ hadamard  ) @ interm_ops[ii] @ np.kron(hadamard @ sdggate, hadamard @ sdggate)
                    temp_matrix2 = np.kron(sgate @ hadamard  , hadamard @ sdggate) @ interm_ops[ii] @ np.kron(hadamard @ sdggate, sgate @ hadamard  )
                    temp_matrix3 = np.kron(sgate @ hadamard  , identity          ) @ interm_ops[ii] @ np.kron(hadamard @ sdggate, identity          )
                    
                    temp_matrix4 = np.kron(hadamard @ sdggate, sgate @ hadamard  ) @ interm_ops[ii] @ np.kron(sgate @ hadamard  , hadamard @ sdggate)
                    temp_matrix5 = np.kron(hadamard @ sdggate, hadamard @ sdggate) @ interm_ops[ii] @ np.kron(sgate @ hadamard  , sgate @ hadamard  )
                    temp_matrix6 = np.kron(hadamard @ sdggate, identity          ) @ interm_ops[ii] @ np.kron(sgate @ hadamard  , identity          )
                    
                    temp_matrix7 = np.kron(identity        , sgate @ hadamard  ) @ interm_ops[ii] @ np.kron(identity        , hadamard @ sdggate)
                    temp_matrix8 = np.kron(identity        , hadamard @ sdggate) @ interm_ops[ii] @ np.kron(identity        , sgate @ hadamard  )
                    temp_matrix9 = np.kron(identity        , identity          ) @ interm_ops[ii] @ np.kron(identity        , identity          )
                    
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
            
            BasisOps_169 = BO_1qubit
            gatenames = 'BasisOp00', 'BasisOp01', 'BasisOp02', 'BasisOp03', 'BasisOp04', 'BasisOp05', 'BasisOp06', 'BasisOp07', 'BasisOp08', 'BasisOp09', 'BasisOp10', 'BasisOp11', 'BasisOp12'
            
            idx_basis_ops = 0
            Bjk_tilde_2qb = np.zeros((241,16,16))
            
            for idx_gatenames_1 in range(0,13):
                for idx_gatenames_2 in range(0,13):
                        
                    # 169 basis operations
                    for k in range(0,16):
        
                        # Initial state
                        circ = prepare_states(k, nqubits)

                        # 72 Basis operations
                        if  idx_gatenames_1 < 10 and idx_gatenames_2 < 10: ### NEW SCENARIO 1
                            circ.add(gates.Unitary(BasisOps_169[idx_gatenames_1], 0, trainable=False, name='%s' %(gatenames[idx_gatenames_1])))
                            circ.add(gates.Unitary(BasisOps_169[idx_gatenames_2], 1, trainable=False, name='%s' %(gatenames[idx_gatenames_2])))
                        
                        ### PARTIALLY REPREPARE QUBIT_1
                        elif idx_gatenames_1 == 10 and idx_gatenames_2 < 10: 
                            circ.add(gates.ResetChannel(0, [1,0])) # ResetChannel(qubit, [prob0, prob1])
                            circ.add(gates.H(0))
                            circ.add(gates.Unitary(BasisOps_169[idx_gatenames_2], 1, trainable=False, name='%s' %(gatenames[idx_gatenames_2])))
                            
                        elif idx_gatenames_1 == 11 and idx_gatenames_2 < 10: 
                            circ.add(gates.ResetChannel(0, [1,0])) # ResetChannel(qubit, [prob0, prob1])
                            circ.add(gates.H(0))
                            circ.add(gates.S(0))
                            circ.add(gates.Unitary(BasisOps_169[idx_gatenames_2], 1, trainable=False, name='%s' %(gatenames[idx_gatenames_2]))) 
                            
                        elif idx_gatenames_1 == 12 and idx_gatenames_2 < 10: 
                            circ.add(gates.ResetChannel(0, [1,0])) # ResetChannel(qubit, [prob0, prob1])
                            circ.add(gates.Unitary(BasisOps_169[idx_gatenames_2], 1, trainable=False, name='%s' %(gatenames[idx_gatenames_2])))
            
                        ### PARTIALLY REPREPARE QUBIT_2
                        elif idx_gatenames_1 < 10 and idx_gatenames_2 == 10: 
                            circ.add(gates.Unitary(BasisOps_169[idx_gatenames_1], 0, trainable=False, name='%s' %(gatenames[idx_gatenames_1]))) 
                            circ.add(gates.ResetChannel(1, [1,0])) # ResetChannel(qubit, [prob0, prob1])
                            circ.add(gates.H(1))
            
                        elif idx_gatenames_1 < 10 and idx_gatenames_2 == 11: 
                            circ.add(gates.Unitary(BasisOps_169[idx_gatenames_1], 0, trainable=False, name='%s' %(gatenames[idx_gatenames_1]))) 
                            circ.add(gates.ResetChannel(1, [1,0])) # ResetChannel(qubit, [prob0, prob1])
                            circ.add(gates.H(1))
                            circ.add(gates.S(1))
            
                        elif idx_gatenames_1 < 10 and idx_gatenames_2 == 12: 
                            circ.add(gates.Unitary(BasisOps_169[idx_gatenames_1], 0, trainable=False, name='%s' %(gatenames[idx_gatenames_1])))
                            circ.add(gates.ResetChannel(1, [1,0])) # ResetChannel(qubit, [prob0, prob1])
            
                        ### FULLY REPREPARE QUBIT_1 AND QUBIT_2
                        elif idx_gatenames_1 == 10 and idx_gatenames_2 == 10: 
                            # qubit_1 = reinitialised |+> state
                            # qubit_2 = reinitialised |+> state
                            circ.add(gates.ResetChannel(0, [1,0])) # ResetChannel(qubit, [prob0, prob1])
                            circ.add(gates.ResetChannel(1, [1,0])) # ResetChannel(qubit, [prob0, prob1])
                            circ.add(gates.H(0))
                            circ.add(gates.H(1))
            
                        elif idx_gatenames_1 == 10 and idx_gatenames_2 == 11: 
                            # qubit_1 = reinitialised |+> state
                            # qubit_2 = reinitialised |y+> state
                            circ.add(gates.ResetChannel(0, [1,0])) # ResetChannel(qubit, [prob0, prob1])
                            circ.add(gates.ResetChannel(1, [1,0])) # ResetChannel(qubit, [prob0, prob1])
                            circ.add(gates.H(0))
                            circ.add(gates.H(1))
                            circ.add(gates.S(1))
            
                        elif idx_gatenames_1 == 10 and idx_gatenames_2 == 12: 
                            # qubit_1 = reinitialised |+> state
                            # qubit_2 = reinitialised |0> state
                            circ.add(gates.ResetChannel(0, [1,0])) # ResetChannel(qubit, [prob0, prob1])
                            circ.add(gates.ResetChannel(1, [1,0])) # ResetChannel(qubit, [prob0, prob1])
                            circ.add(gates.H(0))
            
                        #==================================================================================
                        elif idx_gatenames_1 == 11 and idx_gatenames_2 == 10: 
                            # qubit_1 = reinitialised |y+> state
                            # qubit_2 = reinitialised |+> state
                            circ.add(gates.ResetChannel(0, [1,0])) # ResetChannel(qubit, [prob0, prob1])
                            circ.add(gates.ResetChannel(1, [1,0])) # ResetChannel(qubit, [prob0, prob1])
                            circ.add(gates.H(0))
                            circ.add(gates.S(0))
                            circ.add(gates.H(1))
                        
                        elif idx_gatenames_1 == 11 and idx_gatenames_2 == 11: 
                            # qubit_1 = reinitialised |y+> state
                            # qubit_2 = reinitialised |y+> state
                            circ.add(gates.ResetChannel(0, [1,0])) # ResetChannel(qubit, [prob0, prob1])
                            circ.add(gates.ResetChannel(1, [1,0])) # ResetChannel(qubit, [prob0, prob1])
                            circ.add(gates.H(0))
                            circ.add(gates.S(0))
                            circ.add(gates.H(1))
                            circ.add(gates.S(1))
                        
                        elif idx_gatenames_1 == 11 and idx_gatenames_2 == 12: 
                            # qubit_1 = reinitialised |y+> state
                            # qubit_2 = reinitialised |0> state
                            circ.add(gates.ResetChannel(0, [1,0])) # ResetChannel(qubit, [prob0, prob1])
                            circ.add(gates.ResetChannel(1, [1,0])) # ResetChannel(qubit, [prob0, prob1])
                            circ.add(gates.H(0))
                            circ.add(gates.S(0))
                        
                        #==================================================================================
                        elif idx_gatenames_1 == 12 and idx_gatenames_2 == 10: 
                            # qubit_1 = reinitialised |0> state
                            # qubit_2 = reinitialised |+> state
                            circ.add(gates.ResetChannel(0, [1,0])) # ResetChannel(qubit, [prob0, prob1])
                            circ.add(gates.ResetChannel(1, [1,0])) # ResetChannel(qubit, [prob0, prob1])
                            circ.add(gates.H(1))
                            
                        elif idx_gatenames_1 == 12 and idx_gatenames_2 == 11: 
                            # qubit_1 = reinitialised |0> state
                            # qubit_2 = reinitialised |y+> state
                            circ.add(gates.ResetChannel(0, [1,0])) # ResetChannel(qubit, [prob0, prob1])
                            circ.add(gates.ResetChannel(1, [1,0])) # ResetChannel(qubit, [prob0, prob1])
                            circ.add(gates.H(1))
                            circ.add(gates.S(1))
                            
                        elif idx_gatenames_1 == 12 and idx_gatenames_2 == 12: 
                            # qubit_1 = reinitialised |0> state
                            # qubit_2 = reinitialised |0> state
                            circ.add(gates.ResetChannel(0, [1,0])) # ResetChannel(qubit, [prob0, prob1])
                            circ.add(gates.ResetChannel(1, [1,0])) # ResetChannel(qubit, [prob0, prob1])
            
                        column_of_js = np.zeros((4**nqubits,1))
                        #==================================================================================
        
                        # Measurements
                        for j in range(0,16):
                            new_circ = measurement_basis(j, circ, nqubits)
                            expectation_val = GST_execute_circuit(new_circ, k, j, nshots, nqubits)
                            column_of_js[j,:] = expectation_val
        
                        Bjk_tilde_2qb[idx_basis_ops, :, k] = column_of_js.reshape(16,)
                    
                    idx_basis_ops += 1
        
            idx_basis_ops = 169
            for idx_basis_ops_72 in range(169,241):
        
                for k in range(0,16):
        
                    # Initial states
                    circ = prepare_states(k, nqubits)
        
                    # 72 Basis operations
                    circ.add(gates.Unitary(BasisOps_241[idx_basis_ops_72], 0,1, trainable=False, name='BasisOp%s' %(idx_basis_ops_72)))
        
                    # Measurements
                    for j in range(0,16):
                        new_circ = measurement_basis(j, circ, nqubits)
                        expectation_val = GST_execute_circuit(new_circ, k, j, nshots, nqubits)
                        column_of_js[j,:] = expectation_val
                    
                    Bjk_tilde_2qb[idx_basis_ops, :, k] = column_of_js.reshape(16,)
            
                idx_basis_ops += 1
                
            # Get noisy Pauli Transfer Matrix 
            T = np.array([[1,1,1,1],[0,0,1,0],[0,0,0,1],[1,-1,0,0]])
            Bjk_hat_2qb = np.zeros((241,4**nqubits,4**nqubits))
            for idx_BO in range(0,241):
                Bjk_hat_2qb[idx_BO, :, :] = np.kron(T,T) @ np.linalg.inv(gjk) @ Bjk_tilde_2qb[idx_BO, :, :] @ np.linalg.inv(np.kron(T,T));
            
            # Reshape
            Bjk_hat_2qb_reshaped = np.zeros((256,241))
            for idx_basis_ops in range(0,241):
                temp = np.reshape(Bjk_hat_2qb[idx_basis_ops,:,:], [256,1], order='F') # 'F' so that it stacks columns. 
                Bjk_hat_2qb_reshaped[:,idx_basis_ops] = temp[:,0].flatten()
        
            return Bjk_hat_2qb, Bjk_hat_2qb_reshaped, BasisOps_241

        else:
            raise ValueError(f"nqubits given as {nqubits}. nqubits needs to be either 1 or 2.")
    else:
        raise ValueError(f"nqubits given as {nqubits}. nqubits needs to be either 1 or 2.")

