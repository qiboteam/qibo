import copy

import numpy as np

from qibo import gates, hamiltonians
from qibo.backends import GlobalBackend, matrices
from qibo.config import raise_error
from qibo.hamiltonians import Hamiltonian
from qibo.hamiltonians.abstract import AbstractHamiltonian


def execute_circuit(
    backend,
    c,
    obs,
    nshots=None,
    initial_state=None,
    cdr_params=None,
    calibration=None,
    deterministic=False,
):
    """
    Probabilistic circuit execution with possibilities for error mitigation

    Developed by Michael Tsesmelis (ACSE-mct22)

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
    if deterministic or nshots is None:
        state = c().state()
        if isinstance(obs, list):
            res = []
            for o in obs:
                res.append(o.expectation(state))
        else:
            res = obs.expectation(state)
        return res

    # retrieve state
    state = backend.execute_circuit(
        circuit=c, nshots=nshots, initial_state=initial_state
    )

    # get expectation values
    if isinstance(obs, list):
        result = []
        for o in obs:
            exp = state.expectation_from_samples(o)

            # apply CDR correction
            if cdr_params is not None:
                a, b = cdr_params
                exp = a * exp + b

            result.append(exp)

    else:
        result = state.expectation_from_samples(obs)

        # apply CDR correction
        if cdr_params is not None:
            a, b = cdr_params
            result = a * result + b

    return np.real(result)


def create_hamiltonian(qubit=0, nqubits=1, backend=None):
    """Precomputes Hamiltonian.

    Enhanced by Michael Tsesmelis (ACSE-mct22)

    Args:
        nqubits (int): number of qubits.
        z_qubit (int): qubit where the Z measurement is applied, must be z_qubit < nqubits
        backend (:class:`qibo.backends.abstract.Backend`): Backend object to use for execution.
            If ``None`` the currently active global backend is used.
            Default is ``None``.

    Returns:
        (:class:`qibo.hamiltonians.Hamiltonian`): hamiltonian object.
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


def parameter_shift(
    circuit,
    hamiltonian,
    parameter_index,
    initial_state=None,
    scale_factor=1,
    nshots=None,
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
        forward = backend.execute_circuit(
            circuit=circuit, initial_state=initial_state, nshots=nshots
        ).expectation_from_samples(hamiltonian)

        shifted[parameter_index] -= 2 * s
        circuit.set_parameters(shifted)

        backward = backend.execute_circuit(
            circuit=circuit, initial_state=initial_state, nshots=nshots
        ).expectation_from_samples(hamiltonian)

    circuit.set_parameters(original)

    # float() necessary to not return a 0-dim ndarray
    result = float(generator_eigenval * (forward - backward) * scale_factor)

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


def parameter_shift(
    circuit,
    hamiltonian,
    parameter_index,
    initial_state=None,
    scale_factor=1,
    nshots=None,
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
        forward = backend.execute_circuit(
            circuit=circuit, initial_state=initial_state, nshots=nshots
        ).expectation_from_samples(hamiltonian)

        shifted[parameter_index] -= 2 * s
        circuit.set_parameters(shifted)

        backward = backend.execute_circuit(
            circuit=circuit, initial_state=initial_state, nshots=nshots
        ).expectation_from_samples(hamiltonian)

    circuit.set_parameters(original)

    # float() necessary to not return a 0-dim ndarray
    result = float(generator_eigenval * (forward - backward) * scale_factor)

    return result


class VAR:
    def __init__(self, gate):
        self.parameters = gate.parameters
        self.generator = gate.generator
        self.kwargs = gate.init_kwargs
        del self.kwargs["trainable"]
        self.original_gate = gate

        self.variables = dict()
        for key, val in self.kwargs.items():
            full = self.generator(**self.kwargs)
            self.kwargs[key] = 0
            variable = self.generator(**self.kwargs)
            self.kwargs[key] = val
            self.variables[key] = (full - variable) / self.kwargs[key]

    def return_gate(self, item, angle=0.0):
        def generator(angle):
            return angle * self.variables[item]

        g = self.original_gate
        return gates.OneQubitGate(
            g.target_qubits[0],
            "V",
            exponentiated=g.exponentiated,
            generator=generator,
            angle=angle,
        )


def generate_new_stochastic_params(params, s):
    """Generate the three-gate parameters needed for the stochastic parameter-shift rule.

    Args:
        params: first-gate parameters, already known
        s: random initialiser between 0 and 1
    """
    idx = len(params)
    new_params = np.zeros(2 * idx + 1)

    new_params[:idx] = params
    new_params[idx - 1] = s
    new_params[idx + 1 :] = params
    new_params[-1] = 1 - s

    return new_params


def stochastic_parameter_shift(
    circuit,
    hamiltonian,
    parameter_index,
    initial_state=None,
    scale_factor=1,
    nshots=None,
    stochastic_executions=10,
):
    """In this method the stochastic parameter shift rule (SPSR) is implemented.
    Given a circuit U and an observable H, the SPSR allows to calculate the derivative
    of the expected value of H on the final state with respect to a variational
    parameter of the circuit. The SPSR can calculate gradient approximations on
    a larger family of gates than the standard PSR.
    There is also the possibility of setting a scale factor. It is useful when a
    circuit's parameter is obtained by combination of a variational
    parameter and an external object, such as a training variable in a Quantum
    Machine Learning problem. For example, performing a re-uploading strategy
    to embed some data into a circuit, we apply to the quantum state rotations
    whose angles are in the form: theta' = theta * x, where theta is a variational
    parameter and x an input variable. The SPSR allows to calculate the derivative
    with respect to theta, but if we want to optimize a system with respect its
    variational parameters we need to "free" this procedure from the x depencency.
    If the SPSR is needed to be executed on a real quantum device, it is important
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
        gate_index_start (int): the index which identifies the first parameter of
            the gate associated with the target parameter.
        variable_gate (list of :class:`qibo.gate.abstract.Gate`): to each parameter
            of a complex gate is associated a dependent variable gate, which needs to
            be isolated as the second gate in the SPSR methodology.
        initial_state (ndarray, optional): initial state on which the circuit
            acts. Default is ``None``.
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
            from qibo import hamiltonians, gates
            from qibo.models import Circuit
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
            grad_0 = stochastic_parameter_shift(circuit=c, hamiltonian=test_hamiltonian, parameter_index=0)
            grad_1 = stochastic_parameter_shift(circuit=c, hamiltonian=test_hamiltonian, parameter_index=1)
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
    association = circuit.associate_gates_with_parameters()
    gate = association[parameter_index]
    gate_start = parameter_index
    i = 1
    while association[parameter_index - i] == gate:
        gate_start -= 1
        i += 1

    # defining the shift according to the psr
    shift = np.pi / 4

    # saving original parameters and making a copy
    original = np.asarray(circuit.get_parameters()).copy()
    shifted = original.copy()

    # new circuit
    ancilla_gate = copy.deepcopy(gate)
    param_name = "phi"  # TODO accept any name
    variable_gate = VAR(gate).return_gate(item=param_name, angle=shift)
    circuit.add(variable_gate, position=parameter_index)
    circuit.add(ancilla_gate, position=parameter_index)

    # forward shift
    new_param_count = variable_gate.nparams + ancilla_gate.nparams
    shifted = np.insert(shifted, gate_start + gate.nparams, [0] * new_param_count)

    grads = np.zeros(stochastic_executions)

    # stochastic sampling
    for i, s in enumerate(np.random.uniform(size=stochastic_executions)):
        new_params = generate_new_stochastic_params(gate.parameters, s)
        new_params[ancilla_gate.nparams] += shift

        shifted[parameter_index : parameter_index + len(new_params)] = new_params
        print("Here", shifted)
        circuit.set_parameters(shifted)

        if nshots is None:
            # forward evaluation
            forward = hamiltonian.expectation(
                backend.execute_circuit(
                    circuit=circuit, initial_state=initial_state
                ).state()
            )

            # backward shift and evaluation
            shifted[parameter_index + ancilla_gate.nparams] -= shift * 2

            circuit.set_parameters(shifted)

            backward = hamiltonian.expectation(
                backend.execute_circuit(
                    circuit=circuit, initial_state=initial_state
                ).state()
            )

        # same but using expectation from samples
        else:
            forward = backend.execute_circuit(
                circuit=circuit, initial_state=initial_state, nshots=nshots
            ).expectation_from_samples(hamiltonian)

            shifted[parameter_index + ancilla_gate.nparams] -= shift * 2

            circuit.set_parameters(shifted)

            backward = backend.execute_circuit(
                circuit=circuit, initial_state=initial_state, nshots=nshots
            ).expectation_from_samples(hamiltonian)

        # float() necessary to not return a 0-dim ndarray
        result = float(np.real(forward) - np.real(backward))

        grads[i] = result

    # cleanup
    circuit.remove(variable_gate)
    circuit.remove(ancilla_gate)
    circuit.set_parameters(original)

    return grads.mean() * scale_factor
