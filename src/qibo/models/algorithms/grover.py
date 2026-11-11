import math

import numpy as np

from qibo import gates
from qibo.config import log, raise_error
from qibo.models.circuit import Circuit


class Grover:
    """Model that performs Grover's search algorithm.

    Given a Boolean function :math:`\\chi` over :math:`N` states, which is :math:`1` for the
    :math:`t` states that are solutions and :math:`0` otherwise, the algorithm finds a solution
    with :math:`O(\\sqrt{N / t})` calls to the oracle :math:`\\mathbf{S}_{\\chi}`, which changes
    the sign of the amplitudes of the solutions [1, 2]. Let :math:`\\mathcal{A}` be the circuit
    that prepares the initial superposition
    :math:`\\ket{\\psi} = \\mathcal{A}\\ket{0}` from the all-zero state :math:`\\ket{0}`.
    In the original algorithm it is the Walsh-Hadamard transform, so that
    :math:`\\ket{\\psi}` is the equal superposition of all the states. The algorithm applies
    :math:`j` times the Grover iteration

    .. math::
        \\mathbf{G} = \\mathcal{A} \\, \\mathbf{S}_{0} \\, \\mathcal{A}^{-1} \\, \\mathbf{S}_{\\chi}

    to :math:`\\ket{\\psi}`, where :math:`\\mathbf{S}_{0}` changes the sign of the amplitude of
    :math:`\\ket{0}`, and then it measures the state. For the Walsh-Hadamard transform,
    :math:`\\mathbf{G}` is minus the iteration of Refs. [1, 2], and this global sign has no effect
    on the measurements. After :math:`j` iterations, the probability of measuring a solution is
    :math:`\\sin^{2}((2 j + 1) \\theta)`, with :math:`\\sin^{2}(\\theta) = t / N` [2]. The number
    of iterations is chosen as follows:

    * if ``number_solutions`` :math:`t` is given, it is :math:`\\lfloor \\pi / (4 \\theta) \\rfloor`
      [2], for which the probability of failure is at most :math:`t / N`;
    * if ``target_amplitude`` :math:`a` is given, it is :math:`\\lfloor \\pi / (4 a) \\rfloor`
      [3], where :math:`a` is the absolute value of the amplitude of the target state in
      :math:`\\ket{\\psi}`. This allows using any superposition;
    * otherwise, :math:`t` is unknown, and the iterative algorithm of Ref. [2], Sec. 4, is
      used. See :meth:`iterative_grover`.

    The oracle changes the sign of the amplitudes of the solutions by phase kickback. It flips
    the Grover auxiliary qubit, which is the last qubit of the oracle circuit and is prepared
    in the state :math:`\\ket{-} = H X \\ket{0}` by the algorithm, when the other qubits are
    in a solution.

    Example:
        .. testcode::

            from qibo import Circuit, gates
            from qibo.models import Grover

            # oracle that detects the state |11111>, with the Grover auxiliary qubit being the
            # last qubit
            oracle = Circuit(5 + 1)
            oracle.add(gates.X(5).controlled_by(*range(5)))

            # superposition circuit: equal superposition over 5 qubits
            superposition = Circuit(5)
            superposition.add(gates.H(qubit) for qubit in range(5))

            # create and execute the Grover model
            grover = Grover(oracle, superposition_circuit=superposition, number_solutions=1)
            solution, iterations = grover()
            print(solution, iterations)

        .. testoutput::

            ['11111'] 4

    Args:
        oracle (:class:`qibo.models.circuit.Circuit`): circuit that implements
            :math:`\\mathbf{S}_{\\chi}` by phase kickback. The last qubit is the Grover auxiliary
            qubit, which is prepared in the state :math:`\\ket{-}`, and the first qubits store
            the superposition.
        superposition_circuit (:class:`qibo.models.circuit.Circuit`, optional): circuit
            :math:`\\mathcal{A}` that takes the initial state to the superposition. It is expected
            to use the first qubits to store the relevant superposition. If ``None``, Hadamard
            gates are applied to ``superposition_qubits`` qubits. Defaults to ``None``.
        initial_state_circuit (:class:`qibo.models.circuit.Circuit`, optional): circuit that
            prepares the initial state, which is applied before ``superposition_circuit``. If
            ``None``, the initial state is the all-zero state :math:`\\ket{0 \\cdots 0}`.
            Defaults to ``None``.
        superposition_qubits (int, optional): number of qubits that store the relevant
            superposition, which are the ones that are measured. If ``None``, it is the number of
            qubits of ``superposition_circuit``, which then must not use auxiliary qubits. It is
            required if ``superposition_circuit`` is not given. Defaults to ``None``.
        superposition_size (int, optional): number :math:`N` of states in the superposition. If
            ``None``, it is :math:`2^{n}`, with :math:`n` being the number of qubits that store
            the superposition, *i.e.* an equal superposition of all their states.
            Defaults to ``None``.
        number_solutions (int, optional): number :math:`t` of solutions. It is needed unless the
            iterative algorithm is used. Defaults to ``None``.
        target_amplitude (float, optional): absolute value :math:`a` of the amplitude of the
            target state in the superposition. Only for advanced use and known systems.
            Defaults to ``None``.
        check (Callable, optional): function that returns ``True`` if the bitstring is a
            solution. Its first argument is the bitstring to check, and it is required by the
            iterative algorithm. Defaults to ``None``.
        check_args (tuple, optional): additional arguments of ``check``, without the bitstring.
            Defaults to ``()``.
        iterative (bool, optional): If ``True``, uses the iterative algorithm even if
            ``number_solutions`` or ``target_amplitude`` is given. Defaults to ``False``.

    References:
        1. L. K. Grover, *A fast quantum mechanical algorithm for database search*,
        `Proc. 28th Annu. ACM Symp. Theory Comput. (STOC '96), 212 (1996)
        <https://arxiv.org/abs/quant-ph/9605043>`_.

        2. M. Boyer, G. Brassard, P. Høyer, and A. Tapp, *Tight bounds on quantum searching*,
        `Fortschr. Phys. 46, 493 (1998) <https://arxiv.org/abs/quant-ph/9605034>`_.

        3. L. K. Grover, *Quantum computers can search rapidly by using almost any
        transformation*, `Phys. Rev. Lett. 80, 4329 (1998)
        <https://doi.org/10.1103/PhysRevLett.80.4329>`_.
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
            self.superposition.add(
                [gates.H(qubit) for qubit in range(superposition_qubits)]
            )

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
        """Creates the circuit that prepares the initial state of the algorithm.

        The circuit acts on all the qubits. It prepares the state :math:`\\ket{-}` in the Grover
        auxiliary qubit, which is the last one, with a :class:`qibo.gates.X` gate followed by a
        :class:`qibo.gates.H` gate. Then, it applies ``initial_state_circuit``, if any, and
        ``superposition_circuit``.

        Returns:
            :class:`qibo.models.circuit.Circuit`: circuit that prepares the initial state.
        """
        circuit = Circuit(self.nqubits)
        circuit.add(gates.X(self.nqubits - 1))
        circuit.add(gates.H(self.nqubits - 1))
        if self.initial_state_circuit:
            circuit.add(
                self.initial_state_circuit.on_qubits(
                    *range(self.initial_state_circuit.nqubits)
                )
            )
        circuit.add(self.superposition.on_qubits(*self.space_sup))
        return circuit

    def diffusion(self):
        """Creates the circuit of the diffusion operator.

        The operator is :math:`\\mathcal{A} \\, \\mathbf{S}_{0} \\, \\mathcal{A}^{-1}`, where
        :math:`\\mathcal{A}` is the circuit that prepares the superposition from the all-zero
        state, *i.e.* ``initial_state_circuit`` followed by ``superposition_circuit``, and
        :math:`\\mathbf{S}_{0}` changes the sign of the amplitude of the state in which the
        qubits that store the superposition are all in the state :math:`\\ket{0}`. It is
        implemented with a multi-controlled :class:`qibo.gates.X` gate that acts on the Grover
        auxiliary qubit, between layers of :class:`qibo.gates.X` gates. If the superposition
        circuit does not use auxiliary qubits, the operator is
        :math:`\\mathbf{I} - 2 \\ket{\\psi}\\!\\bra{\\psi}`, which for the equal superposition is
        minus the inversion about the average, *i.e.* the diffusion transform of Ref. [1].

        Returns:
            :class:`qibo.models.circuit.Circuit`: circuit of the diffusion operator, which acts
            on the qubits of the superposition circuit and the Grover auxiliary qubit.
        """
        nqubits = self.superposition.nqubits + 1
        circuit = Circuit(nqubits)
        circuit.add(self.superposition.invert().on_qubits(*range(nqubits - 1)))
        if self.initial_state_circuit:
            circuit.add(
                self.initial_state_circuit.invert().on_qubits(
                    *range(self.initial_state_circuit.nqubits)
                )
            )
        circuit.add([gates.X(qubit) for qubit in range(self.sup_qubits)])
        circuit.add(gates.X(nqubits - 1).controlled_by(*range(self.sup_qubits)))
        circuit.add([gates.X(qubit) for qubit in range(self.sup_qubits)])
        if self.initial_state_circuit:
            circuit.add(
                self.initial_state_circuit.on_qubits(
                    *range(self.initial_state_circuit.nqubits)
                )
            )
        circuit.add(self.superposition.on_qubits(*range(nqubits - 1)))
        return circuit

    def step(self):
        """Creates the circuit of a Grover iteration.

        It applies the oracle, which changes the sign of the amplitudes of the solutions, followed
        by the diffusion operator of :meth:`diffusion`. Hence, the circuit implements the operator
        :math:`\\mathbf{G} = \\mathcal{A} \\, \\mathbf{S}_{0} \\, \\mathcal{A}^{-1} \\,
        \\mathbf{S}_{\\chi}`, and it calls the oracle once.

        Returns:
            :class:`qibo.models.circuit.Circuit`: circuit that acts on all the qubits.
        """
        circuit = Circuit(self.nqubits)
        circuit.add(self.oracle.on_qubits(*self.space_ora))
        circuit.add(self.diffusion().on_qubits(*(self.space_sup + [self.nqubits - 1])))
        return circuit

    def circuit(self, iterations):
        """Creates the circuit that performs Grover's algorithm with a given number of iterations.

        The circuit is composed of :meth:`initialize`, ``iterations`` times :meth:`step` and a
        measurement of the qubits that store the superposition, which are the first ones.

        Args:
            iterations (int): number of times to repeat the Grover iteration, *i.e.* number of
                calls to the oracle.

        Returns:
            :class:`qibo.models.circuit.Circuit`: circuit that performs Grover's algorithm.
        """
        circuit = Circuit(self.nqubits)
        circuit += self.initialize()
        for _ in range(iterations):
            circuit += self.step()
        circuit.add(gates.M(*range(self.sup_qubits)))
        return circuit

    def iterative_grover(self, lamda_value=6 / 5, backend=None):
        """Finds a solution with the iterative algorithm, for an unknown number of solutions.

        It is the algorithm of Ref. [2], Sec. 4, whose expected number of Grover iterations is in
        :math:`O(\\sqrt{N / t})`, where :math:`t` is the unknown number of solutions. Starting
        with :math:`m = 1`, in each round the algorithm:

        1. draws an integer :math:`j` uniformly at random among the non-negative integers
           smaller than :math:`m`;
        2. applies :math:`j` Grover iterations to the initial state, which are none if
           :math:`j = 0`, measures the qubits that store the superposition once, and returns the
           bitstring if ``check`` is ``True`` for it;
        3. sets :math:`m \\leftarrow \\min(\\lambda \\, m, \\sqrt{N})`, where :math:`\\lambda` is
           ``lamda_value``.

        If the total number of iterations exceeds :math:`(9 / 4) \\sqrt{N}`, which is the bound of
        Ref. [2] on the expected number of iterations for :math:`t = 1`, a warning is logged and
        the last measured bitstring is returned, even if it is not a solution. The function
        ``check`` is required.

        Args:
            lamda_value (float, optional): parameter :math:`\\lambda` that controls the growth of
                :math:`m`. It must be strictly between :math:`1` and :math:`4 / 3`. Defaults to
                :math:`6 / 5`.
            backend (:class:`qibo.backends.abstract.Backend`, optional): backend to be used in
                the execution of the circuits. If ``None``, it uses the current backend.
                Defaults to ``None``.

        Returns:
            tuple: the bitstring that was measured and checked as a solution, and the total number
            of Grover iterations, *i.e.* the number of calls to the oracle.
        """
        from qibo.backends import _check_backend

        backend = _check_backend(backend)

        m = 1
        lamda = lamda_value
        total_iterations = 0
        while True:
            it = np.random.randint(math.ceil(m))
            total_iterations += it
            circuit = self.circuit(it)
            result = backend.execute_circuit(circuit, nshots=1)
            measured = result.frequencies(binary=True).most_common(1)[0][0]
            if self.check(measured, *self.check_args):
                return measured, total_iterations
            m = min(lamda * m, math.sqrt(self.sup_size))
            if total_iterations > (9 / 4) * math.sqrt(self.sup_size):
                log.warning("Too many total iterations, output might not be solution.")
                return measured, total_iterations

    def execute(self, nshots=100, freq=False, logs=False, backend=None):
        """Executes Grover's algorithm.

        If ``number_solutions`` or ``target_amplitude`` is given and ``iterative`` is ``False``,
        the number of iterations is computed as described in the documentation of the class. The
        circuit of :meth:`circuit` is sampled ``nshots`` times, and the solutions are the
        ``number_solutions`` most frequent bitstrings, or only the most frequent one if
        ``target_amplitude`` is given. Otherwise, it uses the iterative algorithm of
        :meth:`iterative_grover`, which requires ``check``, and a ``ValueError`` is raised if it
        is not given. The solution and the number of iterations are also stored in the attributes
        ``solution`` and ``iterations``.

        Args:
            nshots (int, optional): number of shots used to sample the circuit when the number of
                iterations is computed. Defaults to :math:`100`.
            freq (bool, optional): If ``True``, stores the frequencies of the measured bitstrings
                in the attribute ``frequencies``, and logs them if ``logs`` is ``True``. It only
                applies when the number of iterations is computed. Defaults to ``False``.
            logs (bool, optional): If ``True``, logs the solutions found, and whether they pass
                ``check``, if given. Defaults to ``False``.
            backend (:class:`qibo.backends.abstract.Backend`, optional): backend to be used in
                the execution of the circuits. If ``None``, it uses the current backend.
                Defaults to ``None``.

        Returns:
            tuple: the solution, which is a list of bitstrings, or a single bitstring if the
            iterative algorithm is used, and the number of Grover iterations, *i.e.* the number of
            calls to the oracle.
        """
        from qibo.backends import _check_backend

        backend = _check_backend(backend)

        if (self.num_sol or self.targ_a) and not self.iterative:
            if self.targ_a:
                it = int(np.pi * (1 / self.targ_a) / 4)
            else:
                theta = math.asin(math.sqrt(self.num_sol / self.sup_size))
                it = int(math.pi / (4 * theta))
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
            for elem in most_common:
                if logs:
                    log.info(elem[0])
                self.solution.append(elem[0])
                if logs and self.check:
                    if self.check(elem[0], *self.check_args):
                        log.info("Solution checked and successful.")
                    else:
                        log.info("Not a solution of the problem. Something went wrong.")
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
        """Equivalent to :meth:`qibo.models.Grover.execute`."""
        return self.execute(nshots=nshots, freq=freq, logs=logs, backend=backend)
