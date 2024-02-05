from copy import deepcopy
from enum import Enum, auto
from functools import partial

import hyperopt
import numpy as np

from qibo.config import raise_error
from qibo.hamiltonians import Hamiltonian

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

class DiagonalAssociation:

        def __init__( name = None,
        mode_diagonal_association: DoubleBracketDiagonalAssociationType = DoubleBracketDiagonalAssociationType.dephasing,
        mode_evolution_oracle: EvolutionOracleType = EvolutionOracleType.text_strings
    ):
        self.name = Name
        self.mode_diagonal_association = mode_diagonal_association  
        self.mode_evolution_oracle = mode_evolution_oracle
    @property
    def name(self):
        return self.name

class EvolutionHamiltonian:
    def __init__(name: String = None, mode_evolution_oracle: EvolutionOracleType):
        self.name = name
    def queryEvolution(self, t_duration):
        return 0
    @property
    def name(self):
        return self.name

class DiagonalAssociationDephasing(DiagonalAssociation):

        def __init__(
        self,
        mode_evolution_oracle: EvolutionOracleType = EvolutionOracleType.text_strings
    ):
            super().__init__(mode_diagonal_association = DoubleBracketDiagonalAssociationType.dephasing, mode_evolution_oracle = mode_evolution_oracle)

        def __call__(self, J_input: EvolutionHamiltonian, k_step_number: list = None, t_duration = None ):
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
        if mode_evolution_oracle is EvolutionOracleType.TrotterSuzuki:
            if t_duration is None:
                #iterate over all Z ops
                return sum Z @ J_input @ Z
            else:
                return sum Z @ J_input.circuit(t_duration) @Z

class DiagonalAssociationFromList(DiagonalAssociation):

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
        if mode_evolution_oracle is EvolutionOracleType.TrotterSuzuki:
            if t_duration is None:
                raise_error(ValueError, f"In the TrotterSuzuki mode you need to work with evolution operators so please specify a time.")
            else:
                return d_k_list[k_step_number.circuit(t_duration)

class DiagonalAssociationFromOptimization(DiagonalAssociation):

    def __init__(
    self,
    loss_function: None,
    mode: DoubleBracketDiagonalAssociationType = DoubleBracketDiagonalAssociationType.canonical,
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
        if self.mode_evolution_oracle is EvolutionOracleType.TrotterSuzuki:
            if t_duration is None:
                raise_error(ValueError, f"In the TrotterSuzuki mode you need to work with evolution operators so please specify a time.")
            else:
                raise_error(TypeError, "Not implemented")
                return sum Z @ J_input.circuit(t_duration) @Z



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

class EvolutionOracleType(Enum):  
    text_strings = auto()
    """If you only want to get a sequence of names of the oracle"""

    numerical = auto()
    """If you will work with exp(is_k J_k) as a numerical matrix"""

    TrotterSuzuki = auto()
    """If you will use SymbolicHamiltonian"""

class EvolutionOracle:
    def __init__( J: AbstractHamiltonian = None, 
            name = None,
            mode_evolution_oracle: EvolutionOracleType = EvolutionOracleType.text_strings ):
       self.h = h
       self.mode_evolution_oracle = mode_evolution_oracle
       self.name = name

    def __call__(self, t_duration: double = None, d: np.array = None):
        """ Returns either the name or the circuit """
        if t is None:
            return self.name
        else:
            return self.circuit( t_duration = t_duration )
        
    def circuit(self, t_duration: double = None):

        if self.mode_evolution_oracle is EvolutionOracleType.text_strings:
            return self.name + str(t_duration)
        elif self.mode_evolution_oracle is EvolutionOracleType.SymbolicHamiltonian:
            return self.h.circuit(t_duration)
        else:
            raise_error(ValueError,
                    f"You are using an EvolutionOracle type which is not yet supported.")
    @property
    def name(self, ):
        return None

    def circuit_sequence(self, k_step_number: int = None):
       EvolutionOracleDiagonalInput =  EvolutionOracle( 
               name = "DiagonalInput",
               mode_evolution_oracle = self.mode_evolution_oracle)
       EvolutionOracleInputHamiltonian = EvolutionOracle( name = "InputHamiltonian" )

        if mode_dbr = DoubleBracketRotationType.group_commutator_reduced
            return [
                    EvolutionOracleDiagonalInput(s_step, d),
                    EvolutionOracleInputHamiltonian(s_step),
                    EvolutionOracleDiagonalInput(s_step,d) ]



class doubleBracketStep:
    def __init__(
    self,
    s_step: double = None,
    d_Z: DiagonalAssociation = None,
    mode_dbr: DoubleBracketRotationType = DoubleBracketRotationType.single_commutator,
    mode_evolutiom_reversal: EvolutionOracleInputHamiltonianReversalType = None,
    mode_evolution_oracle: EvolutionOracleType = EvolutionOracleType.text_strings
):      
        self.s_step = s_step
        self.diagonal_association = d_Z
        self.mode_dbr = mode_dbr #@TODO: this should allow to request gradient search or other operator optimization
        self.mode_gci_reversal = mode_GCI_reversal
        self.mode_dbr = mode_dbr 
        self.mode_evolution_oracle = mode_evolution_oracle

    def loss(self, step: float, look_ahead: int = 1):
        """
        Compute loss function distance between `look_ahead` steps.

        Args:
            step: iteration step.
            look_ahead: number of iteration steps to compute the loss function;
        """
        # copy initial hamiltonian
        h_copy = deepcopy(self.h)

        for _ in range(look_ahead):
            self.__call__(mode=self.mode, step=step)

        # off_diagonal_norm's value after the steps
        loss = self.off_diagonal_norm

        # set back the initial configuration
        self.h = h_copy

        return loss

    def energy_fluctuation(self, state):
        """
        Evaluate energy fluctuation

        .. math::
            \\Xi_{k}(\\mu) = \\sqrt{\\langle\\mu|\\hat{H}^2|\\mu\\rangle - \\langle\\mu|\\hat{H}|\\mu\\rangle^2} \\,

        for a given state :math:`|\\mu\\rangle`.

        Args:
            state (np.ndarray): quantum state to be used to compute the energy fluctuation with H.
        """
        return self.h.energy_fluctuation(state)

    @property
    def backend(self):
        """Get Hamiltonian's backend."""
        return self.h0.backend

    def hyperopt_step(
        self,
        step_min: float = 1e-5,
        step_max: float = 1,
        max_evals: int = 1000,
        space: callable = None,
        optimizer: callable = None,
        look_ahead: int = 1,
        verbose: bool = False,
    ):
        """
        Optimize iteration step.

        Args:
            step_min: lower bound of the search grid;
            step_max: upper bound of the search grid;
            max_evals: maximum number of iterations done by the hyperoptimizer;
            space: see hyperopt.hp possibilities;
            optimizer: see hyperopt algorithms;
            look_ahead: number of iteration steps to compute the loss function;
            verbose: level of verbosity.

        Returns:
            (float): optimized best iteration step.
        """
        if space is None:
            space = hyperopt.hp.uniform
        if optimizer is None:
            optimizer = hyperopt.tpe

        space = space("step", step_min, step_max)
        best = hyperopt.fmin(
            fn=partial(self.loss, look_ahead=look_ahead),
            space=space,
            algo=optimizer.suggest,
            max_evals=max_evals,
            verbose=verbose,
        )

        return best["step"]


class DoubleBracketIterationNumpy(DoubleBracketIteration):
    """
    Class implementing the Double Bracket iteration algorithm.
    For more details, see https://arxiv.org/pdf/2206.11772.pdf

    Args:
        hamiltonian (Hamiltonian): Starting Hamiltonian;
        mode (DoubleBracketDiagonalAssociationType): type of generator of the evolution.

    Example:
        .. testcode::

            import numpy as np
            from qibo.models.dbi.double_bracket import DoubleBracketIteration, DoubleBracketDiagonalAssociationType
            from qibo.hamiltonians import Hamiltonian
            from qibo.quantum_info import random_hermitian

            nqubits = 4
            h0 = random_hermitian(2**nqubits)
            dbf = DoubleBracketIteration(Hamiltonian(nqubits=nqubits, matrix=h0))

            # diagonalized matrix
            dbf.h
    """

    def __init__(
        self,
        hamiltonian: Hamiltonian,
        mode: DoubleBracketDiagonalAssociationType = DoubleBracketDiagonalAssociationType.canonical,
    ):
        self.h = hamiltonian
        self.h0 = deepcopy(self.h)
        self.mode = mode

    def __call__(
        self, step: float, mode: DoubleBracketDiagonalAssociationType = None, d: np.array = None
    ):
        if mode is None:
            mode = self.mode

        if mode is DoubleBracketDiagonalAssociationType.canonical:
            operator = self.backend.calculate_matrix_exp(
                1.0j * step,
                self.commutator(self.diagonal_h_matrix, self.h.matrix),
            )
        elif mode is DoubleBracketDiagonalAssociationType.single_commutator:
            if d is None:
                raise_error(ValueError, f"Cannot use group_commutator with matrix {d}")
            operator = self.backend.calculate_matrix_exp(
                1.0j * step,
                self.commutator(d, self.h.matrix),
            )
        elif mode is DoubleBracketDiagonalAssociationType.group_commutator:
            if d is None:
                raise_error(ValueError, f"Cannot use group_commutator with matrix {d}")
            operator = (
                self.h.exp(-step)
                @ self.backend.calculate_matrix_exp(-step, d)
                @ self.h.exp(step)
                @ self.backend.calculate_matrix_exp(step, d)
            )
        operator_dagger = self.backend.cast(
            np.matrix(self.backend.to_numpy(operator)).getH()
        )
        self.h.matrix = operator @ self.h.matrix @ operator_dagger

    @staticmethod
    def commutator(a, b):
        """Compute commutator between two arrays."""
        return a @ b - b @ a

    @property
    def diagonal_h_matrix(self):
        """Diagonal H matrix."""
        return self.backend.cast(np.diag(np.diag(self.backend.to_numpy(self.h.matrix))))

    @property
    def off_diag_h(self):
        return self.h.matrix - self.diagonal_h_matrix

    @property
    def off_diagonal_norm(self):
        """Norm of off-diagonal part of H matrix."""
        off_diag_h_dag = self.backend.cast(
            np.matrix(self.backend.to_numpy(self.off_diag_h)).getH()
        )
        return np.real(
            np.trace(self.backend.to_numpy(off_diag_h_dag @ self.off_diag_h))
        )


