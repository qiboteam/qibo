from enum import Enum, auto

import hyperopt
import numpy as np

from qibo import Circuit
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
    def __init__(
        self,
        h_generator: AbstractHamiltonian,
        name,
        mode_evolution_oracle: EvolutionOracleType = EvolutionOracleType.text_strings,
    ):
        if (
            mode_evolution_oracle is EvolutionOracleType.hamiltonian_simulation
            and type(h_generator) is not SymbolicHamiltonian
        ):
            raise_error(
                TypeError,
                "If the evolution oracle mode will be to make Trotter-Suzuki decompositions then you must use the SymbolicHamiltonian generator",
            )
        if h_generator is None and name is None:
            raise_error(
                NotImplementedError,
                "You have to specify either a matrix and then work in the numerical mode, or SymbolicHamiltonian and work in hamiltonian_simulation mode or at least a name and work with text_strings to list DBI query lists",
            )

        self.h = h_generator
        self.name = name
        self.mode_evolution_oracle = mode_evolution_oracle
        self.mode_find_number_of_trottersuzuki_steps = True
        self.eps_trottersuzuki = 0.1
        self.please_be_verbose = False

    def __call__(self, t_duration: float = None):
        """Returns either the name or the circuit"""
        if t_duration is None:
            return self.name
        else:
            return self.circuit(t_duration=t_duration)

    def circuit(self, t_duration: float = None):

        if self.mode_evolution_oracle is EvolutionOracleType.text_strings:
            return self.name + str(t_duration)
        elif self.mode_evolution_oracle is EvolutionOracleType.numerical:
            return self.h.exp(t_duration)
        elif self.mode_evolution_oracle is EvolutionOracleType.hamiltonian_simulation:

            if self.please_be_verbose:
                print(
                    "Calling circuit in Hamiltonian simulation mode for time t="
                    + str(t_duration)
                    + " and next running discretization adjustment to reach precision eps = "
                    + str(self.eps_trottersuzuki)
                )
            return self.discretized_evolution_circuit_binary_search(
                t_duration, eps=self.eps_trottersuzuki
            )
        else:
            raise_error(
                ValueError,
                f"You are using an EvolutionOracle type which is not yet supported.",
            )

    def discretized_evolution_circuit(self, t_duration, eps=None):

        nmb_trottersuzuki_steps = 3
        if eps is None:
            eps = self.eps_trottersuzuki
        target_unitary = self.h.exp(t_duration)

        from copy import deepcopy

        proposed_circuit_unitary = np.linalg.matrix_power(
            deepcopy(self.h.circuit(t_duration / nmb_trottersuzuki_steps)).unitary(),
            nmb_trottersuzuki_steps,
        )
        norm_difference = np.linalg.norm(target_unitary - proposed_circuit_unitary)

        if self.please_be_verbose:
            print(nmb_trottersuzuki_steps, norm_difference)
        while norm_difference > eps:
            nmb_trottersuzuki_steps = nmb_trottersuzuki_steps * 2
            proposed_circuit_unitary = np.linalg.matrix_power(
                deepcopy(self.h)
                .circuit(t_duration / nmb_trottersuzuki_steps)
                .unitary(),
                nmb_trottersuzuki_steps,
            )
            norm_difference = np.linalg.norm(target_unitary - proposed_circuit_unitary)
            if self.please_be_verbose:
                print(nmb_trottersuzuki_steps, norm_difference)
        from functools import reduce

        circuit_1_step = deepcopy(self.h.circuit(t_duration / nmb_trottersuzuki_steps))
        combined_circuit = reduce(
            Circuit.__add__, [circuit_1_step] * nmb_trottersuzuki_steps
        )
        assert np.linalg.norm(combined_circuit.unitary() - target_unitary) < eps
        return combined_circuit

    def discretized_evolution_circuit_binary_search(self, t_duration, eps=None):
        nmb_trottersuzuki_steps = 3  # this is the smallest size
        nmb_trottersuzki_steps_right = 50  # this is the largest size for binary search
        if eps is None:
            eps = self.eps_trottersuzuki
        target_unitary = self.h.exp(t_duration)

        from copy import deepcopy

        def check_accuracy(n_steps):
            proposed_circuit_unitary = np.linalg.matrix_power(
                deepcopy(self.h).circuit(t_duration / n_steps).unitary(),
                n_steps,
            )
            norm_difference = np.linalg.norm(target_unitary - proposed_circuit_unitary)
            if self.please_be_verbose:
                print(n_steps, norm_difference)
            return norm_difference < eps

        while nmb_trottersuzuki_steps < nmb_trottersuzki_steps_right:
            mid = (nmb_trottersuzuki_steps + nmb_trottersuzki_steps_right) // 2
            if check_accuracy(mid):
                nmb_trottersuzki_steps_right = mid
            else:
                nmb_trottersuzuki_steps = mid + 1

        from functools import reduce

        circuit_1_step = deepcopy(self.h.circuit(t_duration / nmb_trottersuzuki_steps))
        combined_circuit = reduce(
            Circuit.__add__, [circuit_1_step] * nmb_trottersuzuki_steps
        )
        assert np.linalg.norm(combined_circuit.unitary() - target_unitary) < eps
        return combined_circuit


class FrameShiftedEvolutionOracle(EvolutionOracle):
    def __init__(
        self,
        base_evolution_oracle: EvolutionOracle,
        name,
        before_circuit,
        after_circuit,
    ):

        assert isinstance(before_circuit, type(after_circuit))

        #       if base_evolution_oracle.mode_evolution_oracle is EvolutionOracleType.hamiltonian_simulation :
        #            assert type(before_circuit) is Circuit, str(type(before_circuit))

        self.h = base_evolution_oracle.h
        self.base_evolution_oracle = base_evolution_oracle
        self.name = name + "(" + base_evolution_oracle.name + ")"
        self.mode_evolution_oracle = base_evolution_oracle.mode_evolution_oracle
        self.before_circuit = before_circuit
        self.after_circuit = after_circuit

    def circuit(self, t_duration: float = None):

        if self.mode_evolution_oracle is EvolutionOracleType.text_strings:
            return self.name + "(" + str(t_duration) + ")"
        elif self.mode_evolution_oracle is EvolutionOracleType.numerical:
            return self.before_circuit @ self.h.exp(t_duration) @ self.after_circuit
        elif self.mode_evolution_oracle is EvolutionOracleType.hamiltonian_simulation:
            return (
                self.before_circuit
                + self.base_evolution_oracle.circuit(t_duration)
                + self.after_circuit
            )
        else:
            raise_error(
                ValueError,
                f"You are using an EvolutionOracle type which is not yet supported.",
            )


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


class DiagonalAssociationDephasingChannel(EvolutionOracle):

    def __init__(
        self,
        mode_evolution_oracle: EvolutionOracleType = EvolutionOracleType.text_strings,
    ):
        super().__init__(
            mode_diagonal_association=DoubleBracketDiagonalAssociationType.dephasing,
            mode_evolution_oracle=mode_evolution_oracle,
        )

    def __call__(
        self, J_input: EvolutionOracle, k_step_number: list = None, t_duration=None
    ):
        if mode_evolution_oracle is EvolutionOracleType.text_strings:
            if t_duration is None:
                # iterate over all Z ops
                return r"\Delta(" + J_input.name + ")"
            else:
                return "exp( i" + str(t_duration) + r"\Delta(" + J_input.name + ")"
        elif mode_evolution_oracle is EvolutionOracleType.numerical:
            if t_duration is None:
                return J_input.h.diag()
            else:
                return J_input.diag().exp(t_duration)
        if mode_evolution_oracle is EvolutionOracleType.hamiltonian_simulation:
            if t_duration is None:
                # iterate over all Z ops
                return sum(Z @ J_input @ Z)
            else:
                return sum(Z @ J_input.circuit(t_duration) @ Z)


class DiagonalAssociationFromList(EvolutionOracle):

    def __init__(
        self,
        d_k_list: list = None,
        mode_evolution_oracle: EvolutionOracleType = EvolutionOracleType.text_strings,
    ):
        super().__init__(
            mode_diagonal_association=DoubleBracketDiagonalAssociationType.prescribed,
            mode_evolution_oracle=mode_evolution_oracle,
        )
        self.d_k_list = d_k_list

    def __call__(self, k_step_number: list = None, t_duration=None):

        if mode_evolution_oracle is EvolutionOracleType.text_strings:
            if t_duration is None:
                return "D_" + str(k_step_number)
            else:
                return "exp( i" + str(t_duration) + "D_k"
        elif mode_evolution_oracle is EvolutionOracleType.numerical:
            if t_duration is None:
                return d_k_list[k_step_number]
            else:
                return d_k_list[k_step_number].exp(t_duration)
        if mode_evolution_oracle is EvolutionOracleType.hamiltonian_simulation:
            if t_duration is None:
                raise_error(
                    ValueError,
                    f"In the hamiltonian_simulation mode you need to work with evolution operators so please specify a time.",
                )
            else:
                return d_k_list[k_step_number.circuit(t_duration)]


class DiagonalAssociationFromOptimization(EvolutionOracle):

    def __init__(
        self,
        loss_function: None,
        mode: DoubleBracketDiagonalAssociationType = DoubleBracketDiagonalAssociationType.dephasing,
        mode_evolution_oracle: EvolutionOracleType = EvolutionOracleType.text_strings,
    ):
        self.loss = loss_function

    def __call__(
        self, h: AbstractHamiltonian, k_step_number: list = None, t_duration=None
    ):
        if self.mode_evolution_oracle is EvolutionOracleType.text_strings:
            if t_duration is None:
                return r"Optimize $\mu$ D_" + str(k_step_number)
            else:
                return r"Optimize  $\mu$ exp( i" + str(t_duration) + "D_k"
        elif self.mode_evolution_oracle is EvolutionOracleType.numerical:
            if t_duration is None:
                raise_error(TypeError, "Not implemented")
                return 0
            else:
                raise_error(TypeError, "Not implemented")
                return 0
        if self.mode_evolution_oracle is EvolutionOracleType.hamiltonian_simulation:
            if t_duration is None:
                raise_error(
                    ValueError,
                    f"In the hamiltonian_simulation mode you need to work with evolution operators so please specify a time.",
                )
            else:
                raise_error(TypeError, "Not implemented")
                return None