import random

import numpy as np

from qibo import gates, hamiltonians
from qibo.backends import GlobalBackend, matrices
from qibo.config import raise_error
from qibo.hamiltonians import Hamiltonian
from qibo.hamiltonians.abstract import AbstractHamiltonian
from qibo.models import Circuit
from qibo.models.error_mitigation import (
    CDR,
    apply_readout_mitigation,
    calibration_matrix,
)


def execute_circuit(
    backend,
    c,
    obs,
    nshots,
    initial_state=None,
    cdr_params=None,
    calibration=None,
    deterministic=False,
):
    """
    Probabilistic circuit execution with possibilities for error mitigation

    Args:
        backend (:class:`qibo.backends.abstract.Backend`): backend to execute circuit on
        c (:class:`qibo.models.circuit.Circuit`): custom quantum circuit.
        obs (:class:`qibo.hamiltonians.Hamiltonian`): target observable.
            if you want to execute on hardware, a symbolic hamiltonian must be
            provided as follows (example with Pauli Z and ``nqubits=1``):
            ``SymbolicHamiltonian(np.prod([ Z(i) for i in range(1) ]))``.
        parameter_index (int): the index which identifies the target parameter
            in the ``circuit.get_parameters()`` list.
        initial_state (ndarray, optional): initial state on which the circuit
            acts. Default is ``None``.
        scale_factor (float, optional): parameter scale factor. Default is ``1``.
        nshots (int, optional): number of shots if derivative is evaluated on
            hardware. If ``None``, the simulation mode is executed.
            Default is ``None``.
    """
    if deterministic:
        state = c().state()
        res = obs.expectation(state)
        return res

    # retrieve state
    state = backend.execute_circuit(
        circuit=c, nshots=nshots, initial_state=initial_state
    )

    # apply readout mitigation
    if calibration is not None:
        state = apply_readout_mitigation(state, calibration)

    # get expectation value
    result = state.expectation_from_samples(obs)

    # apply CDR correction
    if cdr_params is not None:
        a, b = cdr_params
        result = a * result + b

    return result


def create_hamiltonian(qubit=0, nqubits=1, backend=None):
    """Precomputes Hamiltonian.

    Args:
        nqubits (int): number of qubits.
        z_qubit (int): qubit where the Z measurement is applied, must be z_qubit < nqubits
        backend (:class:`qibo.backends.abstract.Backend`): Backend object to use for execution.
            If ``None`` the currently active global backend is used.
            Default is ``None``.

    Returns:
        An Hamiltonian object.
    """
    eye = matrices.I
    if qubit == 0:
        h = hamiltonians.Z(1).matrix
        for _ in range(nqubits - 1):
            h = np.kron(h, eye)

    elif qubit == nqubits - 1:
        h = eye
        for _ in range(nqubits - 2):
            h = np.kron(eye, h)
        h = np.kron(h, hamiltonians.Z(1).matrix)
    else:
        h = eye
        for _ in range(nqubits - 1):
            if _ + 1 == qubit:
                h = np.kron(matrices.Z, h)
            else:
                h = np.kron(eye, h)
    return Hamiltonian(nqubits, h, backend=backend)


def error_mitigation(circuit, nqubits, hamiltonian, backend, noise_model, nshots):
    """Fit CDR regression model to noisy states"""

    calibration = calibration_matrix(nqubits=nqubits, backend=backend, nshots=nshots)

    montecarlo = random.randrange(len(hamiltonian))

    _, _, optimal_params, _ = CDR(
        circuit=circuit,
        observable=hamiltonian[montecarlo],
        backend=backend,
        nshots=nshots,
        noise_model=noise_model,
        full_output=True,
        readout={"calibration_matrix": calibration},
    )

    return optimal_params


def calculate_circuit_gradients(
    circuit,
    ham,
    initparams,
    nparams,
    shift_rule,
    cdr_params,
    nshots,
    deterministic,
    var_gates=None,
):
    """
    Full gradient calculation over all circuit parameters, using specific gradient calculation method
    Args:
        circuit: Circuit object whose parameters are trainable
        ham: Hamiltonian applied to final state
        initparams: initial parameter values given to circuit. This is mostly useful for using
                    Parameter object lambda functions
        nparams: number of trainable parameters
        shift-rule: gradient calculation method ("psr", "spsr", or "fdiff")
        cdr_params: error mitigation parameters
        nshots: number of shots for circuit execution
        deterministic: flag to calculate final state deterministically
        var_gates: for a gate generator (H + theta V}, this gate implements V alone. Useful for SPSR.
    Returns: np.array of the observable's gradients with respect to the variational parameters
    """

    obs_gradients = np.zeros(nparams, dtype=np.float64)
    if deterministic:
        nshots = None

    # parameter shift
    if shift_rule == "psr":
        if isinstance(initparams, np.ndarray):
            for ipar in range(nparams):
                obs_gradients[ipar] = parameter_shift(
                    circuit,
                    ham,
                    ipar,
                    initial_state=None,
                    scale_factor=1,
                    nshots=nshots,
                    cdr_params=cdr_params,
                )
        else:
            count = 0
            for ipar, Param in enumerate(initparams):
                scaling = []
                for nparam in range(Param.nparams):
                    scaling.append(Param.get_scaling_factor(nparam))

                obs_gradients[count : count + len(scaling)] = parameter_shift(
                    circuit,
                    ham,
                    ipar,
                    initial_state=None,
                    scale_factor=np.array(scaling),
                    nshots=nshots,
                    cdr_params=cdr_params,
                )
                count += len(scaling)

    # stochastic parameter shift
    """
    elif shift_rule == "spsr":
        if isinstance(initparams, np.ndarray):
            count = 0
            for gate in circuit.queue:
                if not isinstance(gate, gates.ParametrizedGate):
                    continue
                # -1 for s
                for ipar in range(gate.nparams - 1):
                    obs_gradients[ipar] = stochastic_parameter_shift(
                        circuit,
                        ham,
                        count + ipar,
                        gate_index_start=count,
                        variable_gate=var_gates[count + ipar],
                        scale_factor=1.0,
                        initial_state=None,
                        nshots=None,
                    )

                count += gate.nparams

        else:
            count = 0
            for gate in circuit.queue:
                if not isinstance(gate, gates.ParametrizedGate):
                    continue
                # -1 for s
                for ipar in range(gate.nparams - 1):
                    Param = initparams(count + ipar)
                    scaling = []
                    for nparam in range(Param.nparams):
                        scaling.append(Param.get_scaling_factor(nparam))

                    obs_gradients[ipar] = stochastic_parameter_shift(
                        circuit,
                        ham,
                        count + ipar,
                        gate_index_start=count,
                        variable_gate=var_gates[count + ipar],
                        scale_factor=scaling,
                        initial_state=None,
                        nshots=None,
                    )

                count += gate.nparams


    # finite differences (central difference)
    else:
        for ipar in range(optimizer.nparams):
            obs_gradients[ipar] = finite_differences(
                optimizer._circuit,
                ham,
                ipar,
                initial_state=None,
                scale_factor=1,
                nshots=None,
            )"""

    return obs_gradients


def calculate_circuit_gradients(
    circuit,
    ham,
    initparams,
    nparams,
    shift_rule,
    cdr_params,
    nshots,
    deterministic,
    var_gates=None,
):
    """
    Full gradient calculation over all circuit parameters, using specific gradient calculation method
    Args:
        circuit: Circuit object whose parameters are trainable
        ham: Hamiltonian applied to final state
        initparams: initial parameter values given to circuit. This is mostly useful for using
                    Parameter object lambda functions
        nparams: number of trainable parameters
        shift-rule: gradient calculation method ("psr", "spsr", or "fdiff")
        cdr_params: error mitigation parameters
        nshots: number of shots for circuit execution
        deterministic: flag to calculate final state deterministically
        var_gates: for a gate generator (H + theta V}, this gate implements V alone. Useful for SPSR.
    Returns: np.array of the observable's gradients with respect to the variational parameters
    """

    obs_gradients = np.zeros(nparams, dtype=np.float64)
    if deterministic:
        nshots = None

    # parameter shift
    if shift_rule == "psr":
        if isinstance(initparams, np.ndarray):
            for ipar in range(nparams):
                obs_gradients[ipar] = parameter_shift(
                    circuit,
                    ham,
                    ipar,
                    initial_state=None,
                    scale_factor=1,
                    nshots=nshots,
                    cdr_params=cdr_params,
                )
        else:
            count = 0
            for ipar, Param in enumerate(initparams):
                scaling = []
                for nparam in range(Param.nparams):
                    scaling.append(Param.get_scaling_factor(nparam))

                obs_gradients[count : count + len(scaling)] = parameter_shift(
                    circuit,
                    ham,
                    ipar,
                    initial_state=None,
                    scale_factor=np.array(scaling),
                    nshots=nshots,
                    cdr_params=cdr_params,
                )
                count += len(scaling)

    # stochastic parameter shift
    """
    elif shift_rule == "spsr":
        if isinstance(initparams, np.ndarray):
            count = 0
            for gate in circuit.queue:
                if not isinstance(gate, gates.ParametrizedGate):
                    continue
                # -1 for s
                for ipar in range(gate.nparams - 1):
                    obs_gradients[ipar] = stochastic_parameter_shift(
                        circuit,
                        ham,
                        count + ipar,
                        gate_index_start=count,
                        variable_gate=var_gates[count + ipar],
                        scale_factor=1.0,
                        initial_state=None,
                        nshots=None,
                    )

                count += gate.nparams

        else:
            count = 0
            for gate in circuit.queue:
                if not isinstance(gate, gates.ParametrizedGate):
                    continue
                # -1 for s
                for ipar in range(gate.nparams - 1):
                    Param = initparams(count + ipar)
                    scaling = []
                    for nparam in range(Param.nparams):
                        scaling.append(Param.get_scaling_factor(nparam))

                    obs_gradients[ipar] = stochastic_parameter_shift(
                        circuit,
                        ham,
                        count + ipar,
                        gate_index_start=count,
                        variable_gate=var_gates[count + ipar],
                        scale_factor=scaling,
                        initial_state=None,
                        nshots=None,
                    )

                count += gate.nparams


    # finite differences (central difference)
    else:
        for ipar in range(optimizer.nparams):
            obs_gradients[ipar] = finite_differences(
                optimizer._circuit,
                ham,
                ipar,
                initial_state=None,
                scale_factor=1,
                nshots=None,
            )"""

    return obs_gradients


def parameter_shift(
    circuit,
    hamiltonian,
    parameter_index,
    initial_state=None,
    scale_factor=1,
    nshots=None,
    cdr_params=None,
):
    """In this method the parameter shift rule (PSR) is implemented.
    Given a circuit U and an observable H, the PSR allows to calculate the derivative
    of the expected value of H on the final state with respect to a variational
    parameter of the circuit.
    There is also the possibility of setting a scale factor. It is useful when a
    circuit's parameter is obtained by combination of a variational
    parameter and an external object, such as a training variable in a Quantum
    Machine Learning problem. For example, performing a re-uploading strategy
    to embed some data into a circuit, we apply to the quantum state rotations
    whose angles are in the form: theta' = theta * x, where theta is a variational
    parameter and x an input variable. The PSR allows to calculate the derivative
    with respect of theta' but, if we want to optimize a system with respect its
    variational parameters we need to "free" this procedure from the x depencency.
    If the `scale_factor` is not provided, it is set equal to one and doesn't
    affect the calculation.
    If the PSR is needed to be executed on a real quantum device, it is important
    to set `nshots` to some integer value. This enables the execution on the
    hardware by calling the proper methods.

    Args:
        circuit (:class:`qibo.models.circuit.Circuit`): custom quantum circuit.
        hamiltonian (:class:`qibo.hamiltonians.Hamiltonian`): target observable.
            if you want to execute on hardware, a symbolic hamiltonian must be
            provided as follows (example with Pauli Z and ``nqubits=1``):
            ``SymbolicHamiltonian(np.prod([ Z(i) for i in range(1) ]))``.
        parameter_index (int): the index which identifies the target parameter
            in the ``circuit.get_parameters()`` list.
        initial_state (ndarray, optional): initial state on which the circuit
            acts. Default is ``None``.
        scale_factor (float, optional): parameter scale factor. Default is ``1``.
        nshots (int, optional): number of shots if derivative is evaluated on
            hardware. If ``None``, the simulation mode is executed.
            Default is ``None``.

    Returns:
        (float): Value of the derivative of the expectation value of the hamiltonian
            with respect to the target variational parameter.

    Example:

        .. testcode::

            import qibo
            import numpy as np
            from qibo import Circuit, gates, hamiltonians
            from qibo.derivative import parameter_shift

            # defining an observable
            def hamiltonian(nqubits = 1):
                m0 = (1/nqubits)*hamiltonians.Z(nqubits).matrix
                ham = hamiltonians.Hamiltonian(nqubits, m0)

                return ham

            # defining a dummy circuit
            def circuit(nqubits = 1):
                c = Circuit(nqubits = 1)
                c.add(gates.RY(q = 0, theta = 0))
                c.add(gates.RX(q = 0, theta = 0))
                c.add(gates.M(0))

                return c

            # initializing the circuit
            c = circuit(nqubits = 1)

            # some parameters
            test_params = np.random.randn(2)
            c.set_parameters(test_params)

            test_hamiltonian = hamiltonian()

            # running the psr with respect to the two parameters
            grad_0 = parameter_shift(circuit=c, hamiltonian=test_hamiltonian, parameter_index=0)
            grad_1 = parameter_shift(circuit=c, hamiltonian=test_hamiltonian, parameter_index=1)

    """

    # some raise_error
    if parameter_index > len(circuit.get_parameters()):
        raise_error(ValueError, """This index is out of bounds.""")

    if not isinstance(hamiltonian, AbstractHamiltonian):
        raise_error(
            TypeError,
            "hamiltonian must be a qibo.hamiltonians.Hamiltonian or qibo.hamiltonians.SymbolicHamiltonian object",
        )

    # inheriting hamiltonian's backend
    backend = hamiltonian.backend

    # getting the gate's type
    gate = circuit.associate_gates_with_parameters()[parameter_index]

    # getting the generator_eigenvalue
    if isinstance(gate, gates.CU1):
        generator_eigenval = 0.5
    else:
        generator_eigenval = gate.generator_eigenvalue()

    # defining the shift according to the psr
    s = np.pi / (4 * generator_eigenval)

    # saving original parameters and making a copy
    original = np.asarray(circuit.get_parameters()).copy()
    shifted = original.copy()

    # forward shift
    shifted[parameter_index] += s
    circuit.set_parameters(shifted)

    if nshots is None:
        # forward evaluation
        forward = hamiltonian.expectation(
            backend.execute_circuit(
                circuit=circuit, initial_state=initial_state
            ).state()
        )

        # backward shift and evaluation
        shifted[parameter_index] -= 2 * s
        circuit.set_parameters(shifted)

        backward = hamiltonian.expectation(
            backend.execute_circuit(
                circuit=circuit, initial_state=initial_state
            ).state()
        )

    # same but using expectation from samples
    else:
        forward = execute_circuit(
            backend, circuit, hamiltonian, nshots, initial_state, cdr_params
        )

        shifted[parameter_index] -= 2 * s
        circuit.set_parameters(shifted)

        backward = execute_circuit(
            backend, circuit, hamiltonian, nshots, initial_state, cdr_params
        )

    circuit.set_parameters(original)

    # float() necessary to not return a 0-dim ndarray
    result = float(generator_eigenval * (forward - backward)) * scale_factor
    return result


def create_hamiltonian(qubit=0, nqubits=1, backend=None):
    """Precomputes Hamiltonian.

    Args:
        nqubits (int): number of qubits.
        z_qubit (int): qubit where the Z measurement is applied, must be z_qubit < nqubits
        backend (:class:`qibo.backends.abstract.Backend`): Backend object to use for execution.
            If ``None`` the currently active global backend is used.
            Default is ``None``.

    Returns:
        An Hamiltonian object.
    """
    eye = matrices.I
    if qubit == 0:
        h = hamiltonians.Z(1).matrix
        for _ in range(nqubits - 1):
            h = np.kron(h, eye)

    elif qubit == nqubits - 1:
        h = eye
        for _ in range(nqubits - 2):
            h = np.kron(eye, h)
        h = np.kron(h, hamiltonians.Z(1).matrix)
    else:
        h = eye
        for _ in range(nqubits - 1):
            if _ + 1 == qubit:
                h = np.kron(matrices.Z, h)
            else:
                h = np.kron(eye, h)
    return Hamiltonian(nqubits, h, backend=backend)


def error_mitigation(circuit, nqubits, hamiltonian, backend, noise_model, nshots):
    """Fit CDR regression model to noisy states"""

    calibration = calibration_matrix(nqubits=nqubits, backend=backend, nshots=nshots)

    montecarlo = random.randrange(len(hamiltonian))

    _, _, optimal_params, _ = CDR(
        circuit=circuit,
        observable=hamiltonian[montecarlo],
        backend=backend,
        nshots=nshots,
        noise_model=noise_model,
        full_output=True,
        readout={"calibration_matrix": calibration},
    )

    return optimal_params


def execute_circuit(
    backend,
    c,
    obs,
    nshots,
    initial_state=None,
    cdr_params=None,
    calibration=None,
    deterministic=False,
):
    """
    Probabilistic circuit execution with possibilities for error mitigation

    Args:
        backend (:class:`qibo.backends.abstract.Backend`): backend to execute circuit on
        c (:class:`qibo.models.circuit.Circuit`): custom quantum circuit.
        obs (:class:`qibo.hamiltonians.Hamiltonian`): target observable.
            if you want to execute on hardware, a symbolic hamiltonian must be
            provided as follows (example with Pauli Z and ``nqubits=1``):
            ``SymbolicHamiltonian(np.prod([ Z(i) for i in range(1) ]))``.
        parameter_index (int): the index which identifies the target parameter
            in the ``circuit.get_parameters()`` list.
        initial_state (ndarray, optional): initial state on which the circuit
            acts. Default is ``None``.
        scale_factor (float, optional): parameter scale factor. Default is ``1``.
        nshots (int, optional): number of shots if derivative is evaluated on
            hardware. If ``None``, the simulation mode is executed.
            Default is ``None``.
    """
    if deterministic:
        state = c().state()
        res = obs.expectation(state)
        return res

    # retrieve state
    state = backend.execute_circuit(
        circuit=c, nshots=nshots, initial_state=initial_state
    )

    # apply readout mitigation
    if calibration is not None:
        state = apply_readout_mitigation(state, calibration)

    # get expectation value
    result = state.expectation_from_samples(obs)

    # apply CDR correction
    if cdr_params is not None:
        a, b = cdr_params
        result = a * result + b

    return result


def finite_differences(
    circuit,
    hamiltonian,
    parameter_index,
    initial_state=None,
    step_size=1e-7,
):
    """
    Calculate derivative of the expectation value of `hamiltonian` on the
    final state obtained by executing `circuit` on `initial_state` with
    respect to the variational parameter identified by `parameter_index`
    in the circuit's parameters list. This method can be used only in
    exact simulation mode.

    Args:
        circuit (:class:`qibo.models.circuit.Circuit`): custom quantum circuit.
        hamiltonian (:class:`qibo.hamiltonians.Hamiltonian`): target observable.
            if you want to execute on hardware, a symbolic hamiltonian must be
            provided as follows (example with Pauli Z and ``nqubits=1``):
            ``SymbolicHamiltonian(np.prod([ Z(i) for i in range(1) ]))``.
        parameter_index (int): the index which identifies the target parameter
            in the ``circuit.get_parameters()`` list.
        initial_state (ndarray, optional): initial state on which the circuit
            acts. Default is ``None``.
        step_size (float): step size used to evaluate the finite difference
            (default 1e-7).

    Returns:
        (float): Value of the derivative of the expectation value of the hamiltonian
            with respect to the target variational parameter.
    """

    if parameter_index > len(circuit.get_parameters()):
        raise_error(ValueError, f"""Index {parameter_index} is out of bounds.""")

    if not isinstance(hamiltonian, AbstractHamiltonian):
        raise_error(
            TypeError,
            "hamiltonian must be a qibo.hamiltonians.Hamiltonian or qibo.hamiltonians.SymbolicHamiltonian object",
        )

    backend = hamiltonian.backend

    # parameters copies
    parameters = np.asarray(circuit.get_parameters()).copy()
    shifted = parameters.copy()

    # shift the parameter_index element
    shifted[parameter_index] += step_size
    circuit.set_parameters(shifted)

    # forward evaluation
    forward = hamiltonian.expectation(
        backend.execute_circuit(circuit=circuit, initial_state=initial_state).state()
    )

    # backward shift and evaluation
    shifted[parameter_index] -= 2 * step_size
    circuit.set_parameters(shifted)

    backward = hamiltonian.expectation(
        backend.execute_circuit(circuit=circuit, initial_state=initial_state).state()
    )

    circuit.set_parameters(parameters)

    result = (forward - backward) / (2 * step_size)

    return result


class Node:
    """Parent class to create gate nodes"""

    def __init__(self, gate, trainable_params, ID):
        self.gate = gate
        self.trainable_params = trainable_params  # index of optimisable parameters
        self.next = None
        self.id = ID


class ConvergeNode(Node):
    """Node for two-qubit gates"""

    def __init__(self, gate, trainable_params, ID):
        super().__init__(gate, trainable_params, ID)
        self.next_target = None
        self.waiting = None


class Graph:
    """Creates a graph representation of a circuit"""

    def __init__(self, nqubits, gates, trainable_params, gate_params):
        self.gates = gates
        self.trainable_params = trainable_params
        self.gate_params = gate_params
        self.nqubits = nqubits

    def build_graph(self):
        """
        Builds graph based on the circuit gates and associates parameters to each gate.
        """

        # setup
        start = [None] * self.nqubits
        ends = [None] * self.nqubits
        depth = [0] * self.nqubits
        nodes = list()

        count = 0
        # run through each gate in circuit queue
        for i, gate in enumerate(self.gates):
            n = len(gate.init_args) - 1

            # store parameters for ParametrizedGate
            if isinstance(gate, gates.ParametrizedGate):
                trainp = self.trainable_params[count]
                gateid = count
                count += 1
            else:
                trainp = None
                gateid = None

            # two-qubit gates
            if n == 1:
                node = ConvergeNode(gate, trainp, gateid)
                control = gate.init_args[0]
                target = gate.init_args[1]

                # control qubit
                # start of graph
                if start[control] is None:
                    start[control] = i
                    ends[control] = i
                # link to existing graph node
                else:
                    nodes[ends[control]].next = i
                    ends[control] = i

                # target qubit
                # start of graph
                if start[target] is None:
                    start[target] = i
                    ends[target] = i
                # link to existing graph node
                else:
                    nodes[ends[target]].next = i
                    ends[target] = i

                depth[control] += 1
                depth[target] += 1

            # one-qubit gate
            else:
                node = Node(gate, trainp, gateid)
                qubit = gate.target_qubits[0]

                # start of graph
                if start[qubit] is None:
                    start[qubit] = i
                    ends[qubit] = i
                # link to existing graph node
                else:
                    nodes[ends[qubit]].next = i
                    ends[qubit] = i

                depth[qubit] += 1

            # add node to list
            nodes.append(node)

        self.start = start
        self.end = ends
        self.nodes = nodes
        self.depth = max(depth)

    def _determine_basis(self, gate):
        gname = gate.name

        if gname == "rx":
            return gates.X
        elif gname == "ry":
            return gates.Y
        else:
            return gates.Z

    def update_parameters(self, params):
        self.gate_params = params

    def run_layer(self, layer):
        """Runs through one layer of the circuit parameters
        Args:
            layer: int, layer number N
        Returns:
            c: circuit up to nth layer
            trainable_qubits: qubits on which we find trainable gates
            affected_params: which trainable parameters are linked to the trainable gates
        """

        # empty circuit
        c = Circuit(self.nqubits, density_matrix=True)

        current = self.start[:]

        trainable_qubits = []
        affected_params = []
        gate_order = []

        # run through layer up to N
        for iter in range(layer + 1):
            # run through all qubits
            for q in range(self.nqubits):
                node = self.nodes[current[q]]

                # wait for both qubits to reach two-qubit node
                if isinstance(node, ConvergeNode):
                    # first arrived
                    if node.waiting is None:
                        node.waiting = q
                    # second arrived
                    elif node.waiting != q:
                        c.add(node.gate)
                        gate_order.append(node.id)
                        control = node.gate.init_args[0]
                        target = node.gate.init_args[1]
                        current[control] = node.next
                        current[target] = node.next_target
                        node.waiting = None

                # replace last layer by M gate
                elif iter == layer and isinstance(node.gate, gates.ParametrizedGate):
                    c.add(gates.M(q, basis=self._determine_basis(node.gate)))
                    trainable_qubits.append(q)
                    affected_params.append(node.trainable_params)

                # simple one-qubit node
                else:
                    c.add(node.gate)
                    gate_order.append(node.id)
                    if node.next:
                        current[q] = node.next

        # set parameters
        numgates = c.trainable_gates.nparams
        # Update the list_to_update based on the placement order
        updated_list = [self.gate_params[i] for i in gate_order if i is not None]
        c.set_parameters(updated_list[:numgates])

        return c, trainable_qubits, affected_params


def build_graph(circuit, nparams, nqubits, initparams):
    """
    Builds Graph needed for Natural Gradient
    Args:
        circuit (:class:`qibo.models.circuit.Circuit`): custom quantum circuit.
        nparams (int): number of trainable parameters
        nqubits (int): number of qubits in circuit
        initparams (list or np.ndarray): initial circuit parameters

    Returns:
        (:class:`qibo.derivative.Graph`) initialised graph representation of circuit
    """
    # trainable and gate parameters

    if isinstance(initparams, list):
        trainable_params = []
        count = 0
        for Param in initparams:
            indices = Param.get_indices(count)
            count += len(indices)
            trainable_params.append(indices)

    else:
        trainable_params = [[i] for i in range(nparams)]

    # build graph from circuit gates
    graph = Graph(nqubits, circuit.queue, trainable_params, circuit.get_parameters())
    graph.build_graph()

    return graph


def generate_fubini(
    graph,
    nparams,
    nqubits,
    initparams,
    noise_model=None,
    mitigation=False,
    deterministic=False,
):
    """
    Generates the Fubini-Study metric tensor

    Args:
        graph (:class:`qibo.derivative.Graph`): graph representation of circuit
        nparams (int): number of trainable parameters
        nqubits (int): number of qubits in circuit
        initparams (list or np.ndarray): initial circuit parameters
        noise_model (:class:`qibo.noise.NoiseModel`): noise model to apply to circuit
        mitigation (bool): flag to set error mitigation
        deterministic (bool): flag to calculate final state deterministically

    Returns
        (np.ndarray) fubini-study matrix
    """
    fubini = np.zeros((nparams, nparams))
    backend = GlobalBackend()

    scale_factors = []

    if isinstance(initparams, list):
        count = 0
        for Param in initparams:
            indices = Param.get_indices(count)
            count += len(indices)
            for idx in range(len(indices)):
                scale_factors.append(Param.get_scaling_factor(idx))
    else:
        scale_factors = [1] * nparams

    calibration = None
    if mitigation:
        calibration = calibration_matrix(
            nqubits, backend=backend, noise_model=noise_model, nshots=1024
        )
    # run through layers

    for i in range(graph.depth):
        c, qubits, affected_param = graph.run_layer(i)

        if noise_model:
            c = noise_model.apply(c)
        if len(qubits) == 0:
            continue

        # run through parametrized gate
        for qubit, params in zip(qubits, affected_param):
            result = run_subcircuit_measure(
                c, qubit, nqubits, backend, calibration, deterministic
            )

            for p in params:
                # update Fubini-Study matrix
                for t in params:
                    if t == p:  # CHANGE
                        ps = scale_factors[p]
                        ts = scale_factors[t]
                        val = ps * ts * (result - result**2)
                        fubini[p, t] = val if val > 1e-3 else 1e-3

    return fubini


def run_subcircuit_measure(
    c, qubit, nqubits, backend, calibration=None, deterministic=False
):
    """
    Run variance measurement on specific qubit of subcircuit

    Args:
        c (:class:`qibo.models.circuit.Circuit`): custom quantum circuit.
        qubit (int): circuit qubit at which variance is evaluated
        nqubits (int): total number of circuit qubits
        backend (:class:`qibo.backends.abstract.Backend`): simulation backend used to run circuit
        deterministic (bool): flag to calculate final state deterministically
    Return:
        Probability of a specific qubit to be in state |0> in a measured basis"""

    obs = create_hamiltonian(qubit, nqubits, backend)

    if deterministic:
        result = obs.expectation(backend.execute_circuit(c).state())
    else:
        # execute circuit with readout mitigation
        result = execute_circuit(backend, c, obs, 1024, calibration=calibration)

    # expectation value -> state |0> probability
    result = (1 - result) / 2

    return result
