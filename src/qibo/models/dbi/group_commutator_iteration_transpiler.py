from qibo import *
from qibo.models.dbi import *
from qibo.hamiltonians import SymbolicHamiltonian
from qibo import symbols
from qibo.models.dbi.double_bracket import *

from copy import deepcopy
from enum import Enum, auto
from functools import partial

import hyperopt
import numpy as np

from qibo.config import raise_error
from qibo.hamiltonians import Hamiltonian

class DoubleBracketGeneratorType(Enum):
    """Define the evolution generator of a variant of the double-bracket iterations."""
    
    canonical = auto()
    """Use canonical commutator."""
    
    custom = auto()
    """Use some input diagonal matrix"""
    
class DoubleBracketRotationType(Enum):    
    #The dbr types below need a diagonal input matrix $\hat D_k$   :
    
    single_commutator = auto()
    """Use single commutator."""
    
    group_commutator = auto()
    """Use group commutator approximation"""
    
    group_commutator_reduced = auto()
    """Use group commutator approximation with a reduction using symmetry
    
    """  
    
    group_commutator_imperfect = auto()
    """Use group commutator approximation"""
        
    group_commutator_reduced_imperfect = auto()
    """Use group commutator approximation: 
    symmetry of the Hamiltonian implies that with perfect reversion of the input evolution the first order needs less queries.
    We extrapolate that symmetry to the imperfect inversion.
    Note that while may not be performing a similarity operation on the generator of the double bracket iteration, 
    the unfolded operation applied to a state vector will still be unitary:
    
    """

 class EvolutionOracleType(Enum):   
     numpy = auto()
     TrotterSuzuki = auto()

    
class GroupCommutatorIterationWithEvolutionOracles(DoubleBracketIteration):
    """
    Class which will be later merged into the @super but for now develops new tricks in a private repository.
    """

    def __init__(
        self,
        hamiltonian: Hamiltonian,
        mode: DoubleBracketGeneratorType = DoubleBracketGeneratorType.canonical,
        mode_DBR: DoubleBracketRotationType = DoubleBracketRotationType.single_commutator,
        mode_GCI_inversion: EvolutionOracleInputHamiltonianReversalType = None,
    ):
        super().__init__( hamiltonian, mode )
        
        self.mode_GCI_inversion = mode_GCI_inversion
        self.mode_DBR = mode_DBR  
    def __call__(
        self, step: float, mode_DBR: DoubleBracketRotationType = None, d: np.array = None
    ):
        
        if mode_DBR is None:
            mode_DBR= self.mode_DBR
        if d is None:
            if self.mode is DoubleBracketGeneratorType.canonical:
                d = self.diagonal_h_matrix
            else:
                raise_error(ValueError, f"Cannot use group_commutator without specifying matrix {d}")
                
        if mode_DBR is DoubleBracketRotationType.group_commutator_reduced:          
  
            double_bracket_rotation = self.group_commutator_reduced(step, d)
            double_bracket_rotation_dagger = double_bracket_rotation.T.conj()
        else:
            super().__call__(step, d )      
            
        self.h.matrix = double_bracket_rotation_dagger @ self.h.matrix @ double_bracket_rotation

        
    def group_commutator_reduced(self, step, d = None):
        if d is None:
            if mode is DoubleBracketRotationType.canonical:
                d = self.diagonal_h_matrix
            else:
                raise_error(ValueError, f"Cannot use group_commutator without specifying matrix {d}")
        return  (
         self.backend.calculate_matrix_exp(-step, d)
                @ self.h.exp(step)
                @ self.backend.calculate_matrix_exp(step, d)
        )

    class DBI_step:
        def __init__(
        self,
        s_step: double,
        d: Hamiltonian,
        mode: DoubleBracketGeneratorType = DoubleBracketGeneratorType.canonical,
        mode_DBR: DoubleBracketRotationType = DoubleBracketRotationType.single_commutator,
        mode_GCI_inversion: EvolutionOracleInputHamiltonianReversalType = None,
    ):      
            self.s_step = s_step
            self.d = d
            self.mode = mode #@TODO: this should allow to request gradient search or other operator optimization
            self.mode_GCI_inversion = mode_GCI_inversion
            self.mode_DBR = mode_DBR 
        @property
        def circuit_sequence(self):
            if mode_DBR = DoubleBracketRotationType.group_commutator_reduced
                return [EvolutionOracleDiagonalInput(s_step, d), EvolutionOracleInputHamiltonian(s_step), EvolutionOracleDiagonalInput(s_step,d) ]
    class EvolutionOracle:
        def __init__(h: TrotterHamiltonian = None):
           self.h = h
           def get_circuit(self, t_duration)
             freturn self.h.circuit(t_duration)
    class EvolutionOracleDiagonalInput(EvolutionOracle):
       def __init__():
           self.name = 'DiagonalInput'

    class EvolutionOracleInputHamiltonian(EvolutionOracle):
       def __init__():
           self.name = 'Input Hamiltonian'

    def unfold_DBI_circuit_symbolic(self, nmb_DBI_steps = 1, s_list, d_list = None):
        if d is None:
            if mode is DoubleBracketRotationType.canonical:
                d = self.diagonal_h_matrix
            else:
                print( DoubleBracketRotationType.canonical)
                raise_error(ValueError, f"Cannot use group_commutator without specifying matrix {d}")
        if nmb_DBI_steps == 1:
            return [ DBI_step( s_list[0], d_list[0], mode_DBR= DoubleBracketRotationType.group_commutator_reduced).circuit_sequence ]
        else:
            circuit_sequence_so_far = self.unfold_DBI_circuit_symbolic( nmb_DBI_steps = nmb_DBI_steps - 1, s_list,d_list )
            shift_frame_Hk_H0 = reverse(circuit_sequence_so_far) + [EvolutionOracleInputHamiltonian(s_step[nmb_DBI_steps]) ]  + circuit_sequence_so_far 
            return circuit_sequence_so_far.append(EvolutionOracleDiagonalInput(s_step, d[nmb_DBI_steps])) 
                    + shift_frame_Hk_H0.append(EvolutionOracleDiagonalInput(-s_step, d[nmb_DBI_steps]))



    def group_commutator_reduced_unfold(self, step, d = None):
        if d is None:
            if mode is DoubleBracketRotationType.canonical:
                d = self.diagonal_h_matrix
            else:
                print( DoubleBracketRotationType.canonical)
                raise_error(ValueError, f"Cannot use group_commutator without specifying matrix {d}")
        return  (
         self.backend.calculate_matrix_exp(-step, d)
            @ self.unfold_DBI_circuit( nmb_dbi_steps = k ).T.conj()
                @ self.evolution_oracle_reverse_h(step, self.h0)
            @ self.unfold_DBI_circuit( nmb_dbi_steps = k )
                @ self.backend.calculate_matrix_exp(step, d)       
        )

    def evolution_oracle_reverse_h(self, step, h):
        if self.mode_GCI_inversion is None:
            raise_error(ValueError, f"You need to specify what is {self.mode_GCI_inversion} when running a GCI step with the imperfect group commutator")

        if self.mode_GCI_inversion is EvolutionOracleInputHamiltonianReversalType.flip_odd_sites:
            R = self.evolution_oracle_reversal_conjugation_by_flips(step)
        elif self.mode_GCI_inversion is EvolutionOracleInputHamiltonianReversalType.product_Hadamards:
            R = self.evolution_oracle_reversal_conjugation_by_hadamards(step)
        elif self.mode_GCI_inversion is EvolutionOracleInputHamiltonianReversalType.flip_odd_sites_and_NN_Ising_correction:
            R = self.evolution_oracle_reversal_conjugation_by_flips(step)
            Z = self.evolution_oracle_reversal_conjugation_by_unitary_diagonal_correction(R, step, h)
            return R.matrix @ h.exp(step) @ R.matrix @ Z.matrix
        else:
            raise_error(TypeError, "Not implemented")
        return R.matrix @ h.exp(step) @ R.matrix

    def evolution_oracle_reversal_conjugation_by_flips(self,step):
        
        R = symbols.I(0)        
        for qubit_nmb in range(self.h.nqubits):
            if np.mod(qubit_nmb,2) == 0:
                R *= symbols.X(qubit_nmb)

        return SymbolicHamiltonian(R, nqubits = self.h.nqubits)
    
    def evolution_oracle_reversal_conjugation_by_hadamards(self,step):

        R = symbols.I(0)        
        for qubit_nmb in range(self.h.nqubits):
            if np.mod(qubit_nmb,2) == 0:
                R *= ( symbols.X(qubit_nmb)+symbols.Z(qubit_nmb))/np.sqrt(2)

        return SymbolicHamiltonian(R, nqubits = self.nqubits)

    def evolution_oracle_reversal_conjugation_by_unitary_diagonal_correction(self,R, step, h, how = 'default'):
        """This function takes an input reversal unitary and searches for the correction of diagonal terms variationally"""
            @TODO use hyperopt to find d
            if how is 'hyperopt':
                def loss( d, s):
                    np.linalg.norm( R.matrix @ h.exp(s) @ R.matrix @ d @ h.exp(s) - h.exp(0) )
                raise_error(TypeError, "Not implemented")
            else:
                d0 = np.inv( R.matrix @ h.exp(s) @ R.matrix @ h.exp(s) )
                d1 = self.backend.cast(np.diag(np.diag(self.backend.to_numpy( d0 ))))
                from import scipy.linalg import polar
                return polar(d1)[0]
    def evolution_oracle_input_hamiltonian(self,step, diagonal_correction):
            raise_error(TypeError, "Not implemented")

            return None    
    #Questions

