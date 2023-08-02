import numpy as np

from qibo import gates
from qibo.backends import matrices
from qibo.config import raise_error
from qibo.hamiltonians import Hamiltonian
from qibo.hamiltonians.abstract import AbstractHamiltonian
from qibo.models.error_mitigation import apply_readout_mitigation


def calculate_gradients(
    circuit, shift_rule, nparams, paramInputs, cdr_params, ham, nshots
):
    """
    Full parameter-shift rule's implementation
    Args:
        this_feature: np.array 2**nqubits-long containing the state vector assciated to a data
    Returns: np.array of the observable's gradients with respect to the variational parameters
    """

    obs_gradients = np.zeros(nparams, dtype=np.float64)

    # parameter shift
    if shift_rule == "psr":
        if isinstance(paramInputs, np.ndarray):
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
            for ipar, Param in enumerate(paramInputs):
                for nparam in range(Param.nparams):
                    scaling = Param.get_scaling_factor(nparam)

                    obs_gradients[count] = parameter_shift(
                        circuit,
                        ham,
                        ipar,
                        initial_state=None,
                        scale_factor=scaling,
                        nshots=nshots,
                        cdr_params=cdr_params,
                    )
                    count += 1

    # stochastic parameter shift
    """
    elif optimizer.options["shift_rule"] == "spsr":
        for ipar, Param in enumerate(optimizer.parameters):
            ntrainable_params = Param.nparams
            obs_gradients[ipar : ipar + ntrainable_params] = stochastic_parameter_shift(
                optimizer._circuit,
                ham,
                ipar,
                initial_state=None,
                scale_factor=1,
                nshots=None,
            )

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
    result = float(generator_eigenval * (forward - backward) * scale_factor)
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
        h = matrices.Z
        for _ in range(nqubits - 1):
            h = np.kron(h, eye)

    elif qubit == nqubits - 1:
        h = eye
        for _ in range(nqubits - 2):
            h = np.kron(eye, h)
        h = np.kron(h, matrices.Z)
    else:
        h = eye
        for _ in range(nqubits - 1):
            if _ + 1 == qubit:
                h = np.kron(matrices.Z, h)
            else:
                h = np.kron(eye, h)
    return Hamiltonian(nqubits, h, backend=backend)


def execute_circuit(
    backend,
    c,
    obs,
    nshots,
    initial_state=None,
    cdr_params=None,
    calibration=None,
    precise=False,
):
    """Probabilistic circuit execution with possibilities for error mitigation"""
    if precise:
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
