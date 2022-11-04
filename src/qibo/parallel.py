"""
Resources for parallel circuit evaluation.
"""


def parallel_execution(circuit, states, processes=None, backend=None):
    """Execute circuit for multiple states.

    Example:
        .. code-block:: python

            import qibo
            original_backend = qibo.get_backend()
            qibo.set_backend('qibojit')
            from qibo import models, set_threads
            from qibo.parallel import parallel_execution
            import numpy as np
            # create circuit
            nqubits = 22
            circuit = models.QFT(nqubits)
            # create random states
            states = [ np.random.random(2**nqubits) for i in range(5)]
            # set threads to 1 per process (optional, requires tuning)
            set_threads(1)
            # execute in parallel
            results = parallel_execution(circuit, states, processes=2)
            qibo.set_backend(original_backend)

    Args:
        circuit (qibo.models.Circuit): the input circuit.
        states (list): list of states for the circuit evaluation.
        processes (int): number of processes for parallel evaluation.

    Returns:
        Circuit evaluation for input states.
    """
    if backend is None:  # pragma: no cover
        from qibo.backends import GlobalBackend

        backend = GlobalBackend()

    if states is None or not isinstance(states, list):  # pragma: no cover
        from qibo.config import raise_error

        raise_error(RuntimeError, "states must be a list.")

    def operation(state, circuit):
        return backend.execute_circuit(circuit, state)

    from joblib import Parallel, delayed

    results = Parallel(n_jobs=processes, prefer="threads")(
        delayed(operation)(state, circuit) for state in states
    )

    return results


def parallel_parametrized_execution(
    circuit, parameters, initial_state=None, processes=None, backend=None
):
    """Execute circuit for multiple parameters and fixed initial_state.

    Example:
        .. code-block:: python

            import qibo
            original_backend = qibo.get_backend()
            qibo.set_backend('qibojit')
            from qibo import models, gates, set_threads
            from qibo.parallel import parallel_parametrized_execution
            import numpy as np
            # create circuit
            nqubits = 6
            nlayers = 2
            circuit = models.Circuit(nqubits)
            for l in range(nlayers):
                circuit.add((gates.RY(q, theta=0) for q in range(nqubits)))
                circuit.add((gates.CZ(q, q+1) for q in range(0, nqubits-1, 2)))
                circuit.add((gates.RY(q, theta=0) for q in range(nqubits)))
                circuit.add((gates.CZ(q, q+1) for q in range(1, nqubits-2, 2)))
                circuit.add(gates.CZ(0, nqubits-1))
            circuit.add((gates.RY(q, theta=0) for q in range(nqubits)))
            # create random parameters
            size = len(circuit.get_parameters())
            parameters = [ np.random.uniform(0, 2*np.pi, size) for _ in range(10) ]
            # set threads to 1 per process (optional, requires tuning)
            set_threads(1)
            # execute in parallel
            results = parallel_parametrized_execution(circuit, parameters, processes=2)
            qibo.set_backend(original_backend)

    Args:
        circuit (qibo.models.Circuit): the input circuit.
        parameters (list): list of parameters for the circuit evaluation.
        initial_state (np.array): initial state for the circuit evaluation.
        processes (int): number of processes for parallel evaluation.

    Returns:
        Circuit evaluation for input parameters.
    """
    if backend is None:  # pragma: no cover
        from qibo.backends import GlobalBackend

        backend = GlobalBackend()

    if not isinstance(parameters, list):  # pragma: no cover
        from qibo.config import raise_error

        raise_error(RuntimeError, "parameters must be a list.")

    def operation(params, circuit, state):
        if state is not None:
            state = backend.cast(state, copy=True)
        circuit.set_parameters(params)
        return backend.execute_circuit(circuit, state)

    from joblib import Parallel, delayed

    results = Parallel(n_jobs=processes, prefer="threads")(
        delayed(operation)(param, circuit.copy(deep=True), initial_state)
        for param in parameters
    )

    return results
