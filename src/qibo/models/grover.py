import numpy as np

from qibo import gates
from qibo.config import log, raise_error
from qibo.models.circuit import Circuit


class Grover:
    """Model that performs Grover's algorithm.

    For Grover's original search algorithm: `arXiv:quant-ph/9605043 <https://arxiv.org/abs/quant-ph/9605043>`_
    For the iterative version with unknown solutions:`arXiv:quant-ph/9605034 <https://arxiv.org/abs/quant-ph/9605034>`_
    For the Grover algorithm with any superposition:`arXiv:quant-ph/9712011 <https://arxiv.org/abs/quant-ph/9712011>`_

    Args:
        oracle (:class:`qibo.core.circuit.Circuit`): quantum circuit that flips
            the sign using a Grover ancilla initialized with -X-H-. Grover ancilla
            expected to be last qubit of oracle circuit.
        superposition_circuit (:class:`qibo.core.circuit.Circuit`): quantum circuit that
            takes an initial state to a superposition. Expected to use the first
            set of qubits to store the relevant superposition.
        initial_state_circuit (:class:`qibo.core.circuit.Circuit`): quantum circuit
            that initializes the state. If empty defaults to ``|000..00>``
        superposition_qubits (int): number of qubits that store the relevant superposition.
            Leave empty if superposition does not use ancillas.
        superposition_size (int): how many states are in a superposition.
            Leave empty if its an equal superposition of quantum states.
        number_solutions (int): number of expected solutions. Needed for normal Grover.
            Leave empty for iterative version.
        target_amplitude (float): absolute value of the amplitude of the target state. Only for
            advanced use and known systems.
        check (function): function that returns True if the solution has been
            found. Required of iterative approach.
            First argument should be the bitstring to check.
        check_args (tuple): arguments needed for the check function.
            The found bitstring not included.
        iterative (bool): force the use of the iterative Grover

    Example:
        .. testcode::

            import numpy as np

            from qibo import Circuit, gates
            from qibo.models.grover import Grover

            # Create an oracle. Ex: Oracle that detects state |11111>
            oracle = Circuit(5 + 1)
            oracle.add(gates.X(5).controlled_by(*range(5)))

            # Create superoposition circuit. Ex: Full superposition over 5 qubits.
            superposition = Circuit(5)
            superposition.add([gates.H(i) for i in range(5)])

            # Generate and execute Grover class
            grover = Grover(oracle, superposition_circuit=superposition, number_solutions=1)
            solution, iterations = grover()
    """

    def __init__(
        self,
        oracle,
        superposition_circuit=None,
        initial_state_circuit=None,
        superposition_qubits=None,
        superposition_size=None,
        number_solutions=None,
        target_amplitude=None,
        check=None,
        check_args=(),
        iterative=False,
    ):
        self.oracle = oracle
        self.initial_state_circuit = initial_state_circuit

        if superposition_circuit:
            self.superposition = superposition_circuit
        else:
            if not superposition_qubits:
                raise_error(
                    ValueError,
                    "Cannot create Grover model if the "
                    "superposition circuit or number of "
                    "qubits is not specified.",
                )
            self.superposition = Circuit(superposition_qubits)
            self.superposition.add([gates.H(i) for i in range(superposition_qubits)])

        if superposition_qubits:
            self.sup_qubits = superposition_qubits
        else:
            self.sup_qubits = self.superposition.nqubits

        if superposition_size:
            self.sup_size = superposition_size
        else:
            self.sup_size = int(2**self.sup_qubits)

        assert oracle.nqubits > self.sup_qubits

        self.anc_qubits_sup = self.superposition.nqubits - self.sup_qubits
        self.anc_qubits_ora = self.oracle.nqubits - self.sup_qubits - 1

        self.nqubits = (
            self.sup_qubits + max(self.anc_qubits_sup, self.anc_qubits_ora) + 1
        )

        self.check = check
        self.check_args = check_args
        self.num_sol = number_solutions
        self.targ_a = target_amplitude
        self.iterative = iterative

        self.space_sup = list(range(self.sup_qubits + self.anc_qubits_sup))
        self.space_ora = list(range(self.sup_qubits + self.anc_qubits_ora)) + [
            self.nqubits - 1
        ]

    def initialize(self):
        """Initialize the Grover algorithm with the superposition and Grover ancilla."""
        circuit = Circuit(self.nqubits)
        circuit.add(gates.X(self.nqubits - 1))
        circuit.add(gates.H(self.nqubits - 1))
        if self.initial_state_circuit:
            circuit.add(
                self.initial_state_circuit.invert().on_qubits(
                    *range(self.initial_state_circuit.nqubits)
                )
            )
        circuit.add(self.superposition.on_qubits(*self.space_sup))
        return circuit

    def diffusion(self):
        """Construct the diffusion operator out of the superposition circuit."""
        nqubits = self.superposition.nqubits + 1
        circuit = Circuit(nqubits)
        circuit.add(self.superposition.invert().on_qubits(*range(nqubits - 1)))
        if self.initial_state_circuit:
            circuit.add(
                self.initial_state_circuit.invert().on_qubits(
                    *range(self.initial_state_circuit.nqubits)
                )
            )
        circuit.add([gates.X(i) for i in range(self.sup_qubits)])
        circuit.add(gates.X(nqubits - 1).controlled_by(*range(self.sup_qubits)))
        circuit.add([gates.X(i) for i in range(self.sup_qubits)])
        if self.initial_state_circuit:
            circuit.add(
                self.initial_state_circuit.on_qubits(
                    *range(self.initial_state_circuit.nqubits)
                )
            )
        circuit.add(self.superposition.on_qubits(*range(nqubits - 1)))
        return circuit

    def step(self):
        """Combine oracle and diffusion for a Grover step."""
        circuit = Circuit(self.nqubits)
        circuit.add(self.oracle.on_qubits(*self.space_ora))
        circuit.add(self.diffusion().on_qubits(*(self.space_sup + [self.nqubits - 1])))
        return circuit

    def circuit(self, iterations):
        """Creates circuit that performs Grover's algorithm with a set amount of iterations.

        Args:
            iterations (int): number of times to repeat the Grover step.

        Returns:
            :class:`qibo.core.circuit.Circuit` that performs Grover's algorithm.
        """
        circuit = Circuit(self.nqubits)
        circuit += self.initialize()
        for _ in range(iterations):
            circuit += self.step()
        circuit.add(gates.M(*range(self.sup_qubits)))
        return circuit

    def iterative_grover(self, lamda_value=6 / 5, backend=None):
        """Iterative approach of Grover for when the number of solutions is not known.

        Args:
            lamda_value (real): parameter that controls the evolution of the iterative method.
                                Must be between 1 and 4/3.
            backend (:class:`qibo.backends.abstract.Backend`): Backend to use for circuit execution.

        Returns:
            measured (str): bitstring measured and checked as a valid solution.
            total_iterations (int): number of times the oracle has been called.
        """
        from qibo.backends import _check_backend

        backend = _check_backend(backend)

        k = 1
        lamda = lamda_value
        total_iterations = 0
        while True:
            it = np.random.randint(k + 1)
            if it != 0:
                total_iterations += it
                circuit = self.circuit(it)
                result = backend.execute_circuit(circuit, nshots=1)
                measured = result.frequencies(binary=True).most_common(1)[0][0]
                if self.check(measured, *self.check_args):
                    return measured, total_iterations
            k = min(lamda * k, np.sqrt(self.sup_size))
            if total_iterations > (9 / 4) * np.sqrt(self.sup_size):
                log.warning("Too many total iterations, output might not be solution.")
                return measured, total_iterations

    def execute(self, nshots=100, freq=False, logs=False, backend=None):
        """Execute Grover's algorithm.

        If the number of solutions is given, calculates iterations,
        otherwise it uses an iterative approach.

        Args:
            nshots (int): number of shots in order to get the frequencies.
            freq (bool): print the full frequencies after the exact Grover algorithm.
            backend (:class:`qibo.backends.abstract.Backend`): Backend to use for circuit execution.

        Returns:
            solution (str): bitstring (or list of bitstrings) measured as solution of the search.
            iterations (int): number of oracle calls done to reach a solution.
        """
        from qibo.backends import _check_backend

        backend = _check_backend(backend)

        if (self.num_sol or self.targ_a) and not self.iterative:
            if self.targ_a:
                it = int(np.pi * (1 / self.targ_a) / 4)
            else:
                it = int(np.pi * np.sqrt(self.sup_size / self.num_sol) / 4)
            circuit = self.circuit(it)
            result = backend.execute_circuit(circuit, nshots=nshots)
            result = result.frequencies(binary=True)
            if freq:
                if logs:
                    log.info("Result of sampling Grover's algorihm")
                    log.info(result)
                self.frequencies = result
            if logs:
                log.info(
                    f"Most common states found using Grover's algorithm with {it} iterations:"
                )
            if self.targ_a:
                most_common = result.most_common(1)
            else:
                most_common = result.most_common(self.num_sol)
            self.solution = []
            self.iterations = it
            for i in most_common:
                if logs:
                    log.info(i[0])
                self.solution.append(i[0])
                if logs:
                    if self.check:
                        if self.check(i[0], *self.check_args):
                            log.info("Solution checked and successful.")
                        else:
                            log.info(
                                "Not a solution of the problem. Something went wrong."
                            )
        else:
            if not self.check:
                raise_error(ValueError, "Check function needed for iterative approach.")
            measured, total_iterations = self.iterative_grover(backend=backend)
            if logs:
                log.info("Solution found in an iterative process.")
                log.info(f"Solution: {measured}")
                log.info(f"Total Grover iterations taken: {total_iterations}")
            self.solution = measured
            self.iterations = total_iterations
        return self.solution, self.iterations

    def __call__(self, nshots=100, freq=False, logs=False, backend=None):
        """Equivalent to :meth:`qibo.models.grover.Grover.execute`."""
        return self.execute(nshots=nshots, freq=freq, logs=logs, backend=backend)
