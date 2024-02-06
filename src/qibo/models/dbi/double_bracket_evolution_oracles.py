from copy import deepcopy
from enum import Enum, auto
from functools import partial

import hyperopt
import numpy as np

from qibo.config import raise_error
from qibo.hamiltonians import AbstractHamiltonian, SymbolicHamiltonian




class EvolutionOracleType(Enum):  
    text_strings = auto()
    """If you only want to get a sequence of names of the oracle"""

    numerical = auto()
    """If you will work with exp(is_k J_k) as a numerical matrix"""

    hamiltonian_simulation = auto()
    """If you will use SymbolicHamiltonian"""


class EvolutionOracle:
    def __init__( self, 
            h_generator: AbstractHamiltonian, 
            name,
            mode_evolution_oracle: EvolutionOracleType = EvolutionOracleType.text_strings ):
       if mode_evolution_oracle is EvolutionOracleType.hamiltonian_simulation and type(h_generator) is not SymbolicHamiltonian:
            raise_error(TypeError, "If the evolution oracle mode will be to make Trotter-Suzuki decompositions then you must use the SymbolicHamiltonian generator")
       if h_generator is None and name is None:
           raise_error(NotImplementedError, "You have to specify either a matrix and then work in the numerical mode, or SymbolicHamiltonian and work in hamiltonian_simulation mode or at least a name and work with text_strings to list DBI query lists")

       self.h = h_generator
       self.name = name
       self.mode_evolution_oracle = mode_evolution_oracle

    def __call__(self, t_duration: float = None):
        """ Returns either the name or the circuit """
        if t_duration is None:
            return self.name
        else:
            return self.circuit( t_duration = t_duration )
        
    def circuit(self, t_duration: float = None):

        if self.mode_evolution_oracle is EvolutionOracleType.text_strings:
            return self.name + str(t_duration)
        elif self.mode_evolution_oracle is EvolutionOracleType.numerical:
            return self.h.exp(t_duration)
        elif self.mode_evolution_oracle is EvolutionOracleType.hamiltonian_simulation:
            return self.h.circuit(t_duration)
        else:
            raise_error(ValueError,
                    f"You are using an EvolutionOracle type which is not yet supported.")
    @property
    def print(self):
        return self.name

class DoubleBracketDiagonalAssociationType(Enum):
    """Define the evolution generator of a variant of the double-bracket iterations."""
    
    dephasing = auto()
    """Use dephasing for a canonical bracket."""
    
    prescribed = auto()
    """Use some input diagonal matrix for each step: general diagonalization DBI"""

    fixed = auto()
    """Use same input diagonal matrix in each step: BHMM DBI""" 

    optimization = auto()
    """Perform optimization to find best diagonal operator""" 


class DiagonalAssociationDephasing(EvolutionOracle):

    def __init__(
    self,
    mode_evolution_oracle: EvolutionOracleType = EvolutionOracleType.text_strings
):
        super().__init__(mode_diagonal_association = DoubleBracketDiagonalAssociationType.dephasing, mode_evolution_oracle = mode_evolution_oracle)





    def __call__(self, J_input: EvolutionOracle, k_step_number: list = None, t_duration = None ):
        if mode_evolution_oracle is EvolutionOracleType.text_strings:
            if t_duration is None:
                #iterate over all Z ops
                return '\Delta(' + J_input.name + ')'
            else:
                return 'exp( i'+ str(t_duration) +'\Delta(' + J_input.name + ')'
        elif mode_evolution_oracle is EvolutionOracleType.numerical:
            if t_duration is None:
                return J_input.h.diag()
            else:
                return J_input.diag().exp(t_duration) 
        if mode_evolution_oracle is EvolutionOracleType.hamiltonian_simulation:
            if t_duration is None:
                #iterate over all Z ops
                return sum(Z @ J_input @ Z)
            else:
                return sum( Z @ J_input.circuit(t_duration) @Z)

class DiagonalAssociationFromList(EvolutionOracle):

    def __init__(
            self,
            d_k_list: list = None,
            mode_evolution_oracle: EvolutionOracleType = EvolutionOracleType.text_strings
        ):
        super().__init__(mode_diagonal_association = DoubleBracketDiagonalAssociationType.prescribed, mode_evolution_oracle = mode_evolution_oracle)
        self.d_k_list = d_k_list

    def __call__(self, k_step_number: list = None, t_duration = None ):
        
        if mode_evolution_oracle is EvolutionOracleType.text_strings:
            if t_duration is None:
                return 'D_' + str(k_step_number)
            else:
                return 'exp( i'+ str(t_duration) +'D_k'
        elif mode_evolution_oracle is EvolutionOracleType.numerical:
            if t_duration is None:
                return d_k_list[k_step_number]
            else:
                return  d_k_list[k_step_number].exp(t_duration)
        if mode_evolution_oracle is EvolutionOracleType.hamiltonian_simulation:
            if t_duration is None:
                raise_error(ValueError, f"In the hamiltonian_simulation mode you need to work with evolution operators so please specify a time.")
            else:
                return d_k_list[k_step_number.circuit(t_duration)]

class DiagonalAssociationFromOptimization(EvolutionOracle):

    def __init__(
    self,
    loss_function: None,
    mode: DoubleBracketDiagonalAssociationType = DoubleBracketDiagonalAssociationType.dephasing,
    mode_evolution_oracle: EvolutionOracleType = EvolutionOracleType.text_strings
):
        self.loss = loss_function

    def __call__(self, h: AbstractHamiltonian, k_step_number: list = None, t_duration = None ):
        if self.mode_evolution_oracle is EvolutionOracleType.text_strings:
            if t_duration is None:
                return 'Optimize $\mu$ D_' + str(k_step_number)
            else:
                return 'Optimize  $\mu$ exp( i'+ str(t_duration) +'D_k'
        elif self.mode_evolution_oracle is EvolutionOracleType.numerical:
            if t_duration is None:
                raise_error(TypeError, "Not implemented")
                return 0 
            else:
                raise_error(TypeError, "Not implemented")
                return 0
        if self.mode_evolution_oracle is EvolutionOracleType.hamiltonian_simulation:
            if t_duration is None:
                raise_error(ValueError, f"In the hamiltonian_simulation mode you need to work with evolution operators so please specify a time.")
            else:
                raise_error(TypeError, "Not implemented")
                return None



