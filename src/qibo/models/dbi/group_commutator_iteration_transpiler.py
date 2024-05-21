from enum import Enum, auto

import hyperopt
import numpy as np

from qibo import *
from qibo import symbols
from qibo.config import raise_error
from qibo.hamiltonians import Hamiltonian, SymbolicHamiltonian
from qibo.models.dbi import *
from qibo.models.dbi.double_bracket import *
from qibo.models.dbi.double_bracket_evolution_oracles import *


class DoubleBracketRotationType(Enum):
    # The dbr types below need a diagonal input matrix $\hat D_k$   :

    single_commutator = auto()
    """Use single commutator."""

    group_commutator = auto()
    """Use group commutator approximation"""

    group_commutator_reordered = auto()
    """Use group commutator approximation with reordering of the operators"""

    group_commutator_reduced = auto()
    """Use group commutator approximation with a reduction using symmetry

    """
    ## Reserving for later development
    exact_GWW = auto()
    r""" $e^{-s [\Delta(H),H]}$"""
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
    Class which will be later merged into the @super somehow"""

    def __init__(
        self,
        input_hamiltonian_evolution_oracle: EvolutionOracle,
        mode_double_bracket_rotation: DoubleBracketRotationType = DoubleBracketRotationType.group_commutator,
        mode_evolution_oracle: EvolutionOracleType = EvolutionOracleType.numerical,
    ):
        if mode_double_bracket_rotation is DoubleBracketRotationType.single_commutator:
            mode_double_bracket_rotation_old = (
                DoubleBracketGeneratorType.single_commutator
            )
        else:
            mode_double_bracket_rotation_old = (
                DoubleBracketGeneratorType.group_commutator
            )
        super().__init__(
            input_hamiltonian_evolution_oracle.h.dense, mode_double_bracket_rotation_old
        )

        self.input_hamiltonian_evolution_oracle = input_hamiltonian_evolution_oracle

        self.mode_double_bracket_rotation = mode_double_bracket_rotation

        self.gci_unitary = []
        self.gci_unitary_dagger = []
        self.iterated_hamiltonian_evolution_oracle = deepcopy(
            self.input_hamiltonian_evolution_oracle
        )

    def __call__(
        self,
        step_duration: float,
        diagonal_association: EvolutionOracle,
        mode_dbr: DoubleBracketRotationType = None,
    ):

        # Set rotation type
        if mode_dbr is None:
            mode_dbr = self.mode_double_bracket_rotation

        if mode_dbr is DoubleBracketRotationType.single_commutator:
            raise_error(
                ValueError,
                "single_commutator DBR mode doesn't make sense with EvolutionOracle",
            )

        # This will run the appropriate group commutator step
        double_bracket_rotation_step = self.group_commutator(
            step_duration, diagonal_association, mode_dbr=mode_dbr
        )

        before_circuit = double_bracket_rotation_step["backwards"]
        after_circuit = double_bracket_rotation_step["forwards"]

        self.iterated_hamiltonian_evolution_oracle = FrameShiftedEvolutionOracle(
            deepcopy(self.iterated_hamiltonian_evolution_oracle),
            str(step_duration),
            before_circuit,
            after_circuit,
        )

        if (
            self.input_hamiltonian_evolution_oracle.mode_evolution_oracle
            is EvolutionOracleType.numerical
        ):
            self.h.matrix = before_circuit @ self.h.matrix @ after_circuit

        elif (
            self.input_hamiltonian_evolution_oracle.mode_evolution_oracle
            is EvolutionOracleType.hamiltonian_simulation
        ):

            self.h.matrix = (
                before_circuit.unitary() @ self.h.matrix @ after_circuit.unitary()
            )

        elif (
            self.input_hamiltonian_evolution_oracle.mode_evolution_oracle
            is EvolutionOracleType.text_strings
        ):
            raise_error(NotImplementedError)
        else:
            super().__call__(step_duration, diagonal_association.h.dense.matrix)

    def eval_gcr_unitary(
        self,
        step_duration: float,
        eo1: EvolutionOracle,
        eo2: EvolutionOracle = None,
        mode_dbr: DoubleBracketRotationType = None,
    ):
        u = self.group_commutator(step_duration, eo1, eo2, mode_dbr=mode_dbr)[
            "forwards"
        ]
        if eo1.mode_evolution_oracle is EvolutionOracleType.hamiltonian_simulation:
            return u.unitary()
        elif eo1.mode_evolution_oracle is EvolutionOracleType.numerical:
            return u

    def group_commutator(
        self,
        t_step: float,
        eo1: EvolutionOracle,
        eo2: EvolutionOracle = None,
        mode_dbr: DoubleBracketRotationType = None,
    ):
        s_step = np.sqrt(t_step)

        if eo2 is None:
            eo2 = self.iterated_hamiltonian_evolution_oracle

        assert eo1.mode_evolution_oracle.value is eo2.mode_evolution_oracle.value

        if mode_dbr is None:
            gc_type = self.mode_double_bracket_rotation
        else:
            gc_type = mode_dbr

        if gc_type is DoubleBracketRotationType.single_commutator:
            raise_error(
                ValueError,
                "You are trying to get the group commutator query list but your dbr mode is single_commutator and not an approximation by means of a product formula!",
            )

        if gc_type is DoubleBracketRotationType.group_commutator:
            query_list_forward = [
                eo2.circuit(-s_step),
                eo1.circuit(s_step),
                eo2.circuit(s_step),
                eo1.circuit(-s_step),
            ]
            query_list_backward = [
                eo1.circuit(s_step),
                eo2.circuit(-s_step),
                eo1.circuit(-s_step),
                eo2.circuit(s_step),
            ]
        elif gc_type is DoubleBracketRotationType.group_commutator_reordered:
            query_list_forward = [
                eo1.circuit(s_step),
                eo2.circuit(-s_step),
                eo1.circuit(-s_step),
                eo2.circuit(s_step),
            ]
            query_list_backward = [
                eo2.circuit(-s_step),
                eo1.circuit(s_step),
                eo2.circuit(s_step),
                eo1.circuit(-s_step),
            ]
        elif gc_type is DoubleBracketRotationType.group_commutator_reduced:
            query_list_forward = [
                eo1.circuit(s_step),
                eo2.circuit(s_step),
                eo1.circuit(-s_step),
            ]
            query_list_backward = [
                eo1.circuit(s_step),
                eo2.circuit(-s_step),
                eo1.circuit(-s_step),
            ]

        else:
            raise_error(
                ValueError,
                "You are in the group commutator query list but your dbr mode is not recognized",
            )

        eo_mode = eo1.mode_evolution_oracle
        from functools import reduce

        if eo_mode is EvolutionOracleType.text_strings:
            return {
                "forwards": reduce(str.__add__, query_list_forward),
                "backwards": reduce(str.__add__, query_list_backward),
            }
        elif eo_mode is EvolutionOracleType.hamiltonian_simulation:
            return {
                "forwards": reduce(Circuit.__add__, query_list_forward[::-1]),
                "backwards": reduce(Circuit.__add__, query_list_backward[::-1]),
            }
        elif eo_mode is EvolutionOracleType.numerical:
            return {
                "forwards": reduce(np.ndarray.__matmul__, query_list_forward),
                "backwards": reduce(np.ndarray.__matmul__, query_list_backward),
            }
        else:
            raise_error(ValueError, "Your EvolutionOracleType is not recognized")
