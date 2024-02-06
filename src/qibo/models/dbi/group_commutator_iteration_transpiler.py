from qibo import *
from qibo.models.dbi import *
from qibo.hamiltonians import SymbolicHamiltonian
from qibo import symbols
from qibo.models.dbi.double_bracket import *
from qibo.models.dbi.double_bracket_evolution_oracles import *

from copy import deepcopy
from enum import Enum, auto
from functools import partial

import hyperopt
import numpy as np

from qibo.config import raise_error
from qibo.hamiltonians import Hamiltonian

class DoubleBracketRotationType(Enum):    
    #The dbr types below need a diagonal input matrix $\hat D_k$   :
    
    single_commutator = auto()
    """Use single commutator."""
    
    group_commutator = auto()
    """Use group commutator approximation"""
    
    group_commutator_reduced = auto()
    """Use group commutator approximation with a reduction using symmetry
    
    """  
   
   ## Reserving for later development
    group_commutator_imperfect = auto()
    """Use group commutator approximation"""
        
    group_commutator_reduced_imperfect = auto()
    """Use group commutator approximation: 
    symmetry of the Hamiltonian implies that with perfect reversion of the input evolution the first order needs less queries.
    We extrapolate that symmetry to the imperfect reversal.
    Note that while may not be performing a similarity operation on the generator of the double bracket iteration, 
    the unfolded operation applied to a state vector will still be unitary:
    
    """

    
class GroupCommutatorIterationWithEvolutionOracles(DoubleBracketIteration):
    """
    Class which will be later merged into the @super somehow    """

    def __init__(
        self,
        input_hamiltonian_evolution_oracle: EvolutionOracle,
        mode_double_bracket_rotation: DoubleBracketRotationType = DoubleBracketRotationType.group_commutator,
        mode_evolution_oracle: EvolutionOracleType = EvolutionOracleType.numerical,
        mode_diagonal_association: DoubleBracketDiagonalAssociationType = DoubleBracketDiagonalAssociationType.dephasing
    ):
        if mode_double_bracket_rotation is DoubleBracketRotationType.single_commutator:
            mode_double_bracket_rotation_old = DoubleBracketGeneratorType.single_commutator
        else:
            mode_double_bracket_rotation_old = DoubleBracketGeneratorType.group_commutator
        super().__init__( input_hamiltonian_evolution_oracle.h, mode_double_bracket_rotation_old )
        
        self.input_hamiltonian_evolution_oracle = input_hamiltonian_evolution_oracle 

        self.mode_diagonal_association = mode_diagonal_association
        self.mode_double_bracket_rotation = mode_double_bracket_rotation  

        self.gci_unitary = []
        self.gci_unitary_dagger = []
        self.iterated_hamiltonian_query_list = [ [ self.input_hamiltonian_evolution_oracle] ] 

    def __call__(
        self, 
        step_duration: float = None, 
        diagonal_association: EvolutionOracle = None,
        mode_double_bracket_rotation: DoubleBracketRotationType = None,
        ):
        
        if mode_double_bracket_rotation is None:
            mode_double_bracket_rotation = self.mode_double_bracket_rotation

        if diagonal_association is None:
            if self.mode_diagonal_association is DoubleBracketDiagonalAssociationType.dephasing:
                raise_error(NotImplementedError, "diagonal_h_matrix is np.array but need to cast to SymbolicHamiltonian")
                diagonal_association = EvolutionOracle( self.diagonal_h_matrix, 'Dephasing',
                        mode_evolution_oracle = self.input_hamiltonian_evolution_oracle.mode_evolution_oracle)
            else:
                raise_error(ValueError, f"Cannot use group_commutator without specifying matrix {d}. Did you want to set to canonical mode?")
        else:
            self.mode_diagonal_association = DoubleBracketDiagonalAssociationType.prescribed

        if self.mode_double_bracket_rotation is DoubleBracketRotationType.single_commutator:
            raise_error(NotImplementedError, "Keeping track of single commutator DBRs not implemented")
            double_bracket_rotation_step = self.single_commutator_query_list(step, diagonal_association)
        else:
        #This will run the appropriate group commutator step
            double_bracket_rotation_step = self.group_commutator_query_list(step_duration, diagonal_association)        

        if self.mode_evolution_oracle is EvolutionOracleType.numerical:  
            #then dbr step output  is a matrix
            double_bracket_rotation_matrix = 0 * self.h.matrix
            double_bracket_rotation_dagger_matrix = 0 * self.h.matrix
            for m in double_bracket_rotation_matrix:
                double_bracket_rotation_matrix = double_bracket_rotation_matrix * m
            for m in double_bracket_rotation_dagger_matrix:
                double_bracket_rotation_dagger_matrix = double_bracket_rotation_dagger_matrix * m
            self.h.matrix = double_bracket_rotation_dagger_matrix @ self.h.matrix @ double_bracket_rotation_matrix

        elif self.mode_evolution_oracle is EvolutionOracleType.hamiltonian_simulation:
            #then dbr step output is a query list
            self.gci_unitary.append(double_bracket_rotation_step[forwards])
            self.gci_unitary_dagger.append(double_bracket_rotation_step[backwards]) 

            self.iterated_hamiltonian_evolution_oracle =
                                        FrameShiftedEvolutionOracle(
                                                self.iterated_hamiltonian_evolution_oracle,
                                                name = str(step_duration),
                                                double_bracket_rotation_step[backwards],
                                                double_bracket_rotation_step[forwards])
                    

        elif self.mode_evolution_oracle is EvolutionOracleType.text_strings):  
            raise_error(NotImplementedError)
        else:
            super().__call__(step, d )      
            
    def group_commutator_query_list(self,
            s_step: float,
            diagonal_association_evolution_oracle: EvolutionOracle = None,
            iterated_hamiltonian_evolution_oracle: EvolutionOracle = None):
        
        if iterated_hamiltonian_evolution_oracle is None:
            iterated_hamiltonian_evolution_oracle = self.iterated_hamiltonian_evolution_oracle

        if self.mode_double_bracket_rotation is DoubleBracketRotationType.group_commutator:
            return {'forwards': [
                    iterated_hamiltonian_evolution_oracle.circuit(-s_step),
                    diagonal_association_evolution_oracle.circuit(s_step),
                    iterated_hamiltonian_evolution_oracle.circuit(s_step),
                    diagonal_association_evolution_oracle.circuit(-s_step) 
                    ]   ,
                    'backwards': [ #in general an evolution oracle might have imperfect time reversal    
                    diagonal_association_evolution_oracle.circuit(s_step),
                    iterated_hamiltonian_evolution_oracle.circuit(-s_step),
                    diagonal_association_evolution_oracle.circuit(-s_step),
                    iterated_hamiltonian_evolution_oracle.circuit(s_step)
                    ] }
        elif self.mode_double_bracket_rotation is DoubleBracketRotationType.group_commutator_reduced:
            return {'forwards': [
                    diagonal_association_evolution_oracle.circuit(s_step),
                    iterated_hamiltonian_evolution_oracle.circuit(s_step),
                    diagonal_association_evolution_oracle.circuit(-s_step) 
                    ]   ,
                    'backwards': [
                    diagonal_association_evolution_oracle.circuit(s_step),
                    iterated_hamiltonian_evolution_oracle.circuit(-s_step),
                    diagonal_association_evolution_oracle.circuit(-s_step) 
                    ] }
        else:
            if self.mode_double_bracket_rotation is DoubleBracketRotationType.single_commutator:
                raise_error(ValueError, "You are in the group commutator query list but your dbr mode is a perfect bracket and not an approximation by means of a group commutator!")
            else:
                raise_error(ValueError, "You are in the group commutator query list but your dbr mode is not recognized")



