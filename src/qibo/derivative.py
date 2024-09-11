import numpy as np

from qibo.config import raise_error
from qibo.hamiltonians.abstract import AbstractHamiltonian


def parameter_shift(
    circuit,
    hamiltonian,
    parameter_index,
    initial_state=None,
    scale_factor=1,
    nshots=None,
):
    """In this method the parameter shift rule (PSR) is implemented.
    Given a circuit :math:`U` and an observable :math:`H`, the PSR allows to calculate the derivative
    of the expected value of :math:`H` on the final state with respect to a variational
    parameter of the circuit.
    There is also the possibility of setting a scale factor. It is useful when a
    circuit's parameter is obtained by combination of a variational
    parameter and an external object, such as a training variable in a Quantum
    Machine Learning problem. For example, performing a re-uploading strategy
    to embed some data into a circuit, we apply to the quantum state rotations
    whose angles are in the form :math:`\\theta^{\\prime} = x \\, \\theta`,
    where :math:`\\theta` is a variational parameter, and :math:`x` an input variable.
    The PSR allows to calculate the derivative with respect to :math:`\\theta^{\\prime}`.
    However, if we want to optimize a system with respect to its
    variational parameters, we need to "free" this procedure from the :math:`x` depencency.
    If the ``scale_factor`` is not provided, it is set equal to one and doesn't
    affect the calculation.
    If the PSR is needed to be executed on a real quantum device, it is important
    to set ``nshots`` to some integer value. This enables the execution on the
    hardware by calling the proper methods.

    Args:
        circuit (:class:`qibo.models.circuit.Circuit`): custom quantum circuit.
        hamiltonian (:class:`qibo.hamiltonians.Hamiltonian`): target observable.
            if you want to execute on hardware, a symbolic hamiltonian must be
            provided as follows (example with Pauli-:math:`Z` and :math:`n = 1`):
            ``SymbolicHamiltonian(np.prod([ Z(i) for i in range(1) ]))``.
        parameter_index (int): the index which identifies the target parameter
            in the ``circuit.get_parameters()`` list.
        initial_state (ndarray, optional): initial state on which the circuit
            acts. If ``None``, defaults to the zero state :math:`\\ket{\\mathbf{0}}`.
            Defaults to ``None``.
        scale_factor (float, optional): parameter scale factor. Defaults to :math:`1`.
        nshots (int, optional): number of shots if derivative is evaluated on
            hardware. If ``None``, the simulation mode is executed.
            Defaults to ``None``.

    Returns:
        float: Value of the derivative of the expectation value of the hamiltonian
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
    Calculate derivative of the expectation value of ``hamiltonian`` on the
    final state obtained by executing ``circuit`` on ``initial_state`` with
    respect to the variational parameter identified by ``parameter_index``
    in the circuit's parameters list. This method can be used only in
    exact simulation mode.

    Args:
        circuit (:class:`qibo.models.circuit.Circuit`): custom quantum circuit.
        hamiltonian (:class:`qibo.hamiltonians.Hamiltonian`): target observable.
            To execute on hardware, a symbolic hamiltonian must be
            provided as follows (example with Pauli-:math:`Z` and :math:`n = 1`):
            ``SymbolicHamiltonian(np.prod([ Z(i) for i in range(1) ]))``.
        parameter_index (int): the index which identifies the target parameter
            in the :meth:`qibo.models.Circuit.get_parameters` list.
        initial_state (ndarray, optional): initial state on which the circuit
            acts. If ``None``, defaults to the zero state :math:`\\ket{\\mathbf{0}}`.
            Defaults to ``None``.
        step_size (float, optional): step size used to evaluate the finite difference.
            Defaults to :math:`10^{-7}`.

    Returns:
        float: Value of the derivative of the expectation value of the hamiltonian
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
