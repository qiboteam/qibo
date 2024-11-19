"""
Resources for parallel circuit evaluation.
"""

from typing import Iterable, Optional

from joblib import Parallel, delayed

from qibo.backends import _check_backend
from qibo.config import raise_error


def parallel_execution(circuit, states, processes: Optional[int] = None, backend=None):
    """Execute circuit for multiple states.

    Example:
        .. code-block:: python

            import qibo
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

    Args:
        circuit (:class:`qibo.models.Circuit`): the input circuit.
        states (list): list of states for the circuit evaluation.
        processes (int, optional): number of processes for parallel evaluation.
            If ``None``, defaults to :math:`1`. Defaults to ``None``.

    Returns:
        Circuit evaluation for input states.
    """
    backend = _check_backend(backend)

    if states is None or not isinstance(states, list):  # pragma: no cover
        raise_error(TypeError, "states must be a list.")

    def operation(state, circuit):
        backend.set_threads(backend.nthreads)
        return backend.execute_circuit(circuit, state)

    results = Parallel(n_jobs=processes, prefer="threads")(
        delayed(operation)(state, circuit) for state in states
    )

    return results


def parallel_circuits_execution(
    circuits, states=None, nshots=1000, processes=None, backend=None
):
    """Execute multiple circuits

    Example:
        .. code-block:: python

            import qibo
            qibo.set_backend('qibojit')
            from qibo import models, set_threads
            from qibo.parallel import parallel_circuits_execution
            import numpy as np
            # create different circuits
            circuits = [models.QFT(n) for n in range(5, 16)]
            # set threads to 1 per process (optional, requires tuning)
            set_threads(1)
            # execute in parallel
            results = parallel_circuits_execution(circuits, processes=2)

    Args:
        circuits (list): list of circuits to execute.
        states (optional, list): list of states to use as initial for each circuit.
            Must have the same length as ``circuits``.
            If one state is given it is used on all circuits.
            If not given the default initial state on all circuits.
        nshots (int): Number of shots when performing measurements, same for all circuits.
        processes (int): number of processes for parallel evaluation.

    Returns:
        Circuit evaluation for input states.
    """
    backend = _check_backend(backend)

    if not isinstance(circuits, Iterable):  # pragma: no cover
        raise_error(TypeError, "circuits must be iterable.")

    if (
        isinstance(states, (list, tuple))
        and isinstance(circuits, (list, tuple))
        and len(states) != len(circuits)
    ):
        raise_error(ValueError, "states must have the same length as circuits.")
    elif states is not None and not isinstance(states, Iterable):
        raise_error(TypeError, "states must be iterable.")

    def operation(circuit, state):
        backend.set_threads(backend.nthreads)
        return backend.execute_circuit(circuit, state, nshots)

    if states is None or isinstance(states, backend.tensor_types):
        results = Parallel(n_jobs=processes, prefer="threads")(
            delayed(operation)(circuit, states) for circuit in circuits
        )
    else:
        results = Parallel(n_jobs=processes, prefer="threads")(
            delayed(operation)(circuit, state)
            for circuit, state in zip(circuits, states)
        )

    return results


def parallel_parametrized_execution(
    circuit,
    parameters,
    initial_state=None,
    processes: Optional[int] = None,
    backend=None,
):
    """Execute circuit for multiple parameters and fixed initial_state.

    Example:
        .. code-block:: python

            import numpy as np

            from qibo import Circuit, gates, set_backend, set_threads
            from qibo.parallel import parallel_parametrized_execution

            set_backend('qibojit')

            # create circuit
            nqubits = 6
            nlayers = 2
            circuit = Circuit(nqubits)
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

    Args:
        circuit (:class:`qibo.models.Circuit`): the input circuit.
        parameters (list): list of parameters for the circuit evaluation.
        initial_state (ndarray): initial state for the circuit evaluation.
        processes (int): number of processes for parallel evaluation.
            This corresponds to the number of threads, if a single thread is used
            for each circuit evaluation. If more threads are used for each circuit
            evaluation then some tuning may be required to obtain optimal performance.
            Default is ``None`` which corresponds to a single thread.

    Returns:
        Circuit evaluation for input parameters.
    """
    backend = _check_backend(backend)

    if not isinstance(parameters, list):  # pragma: no cover
        raise_error(TypeError, "parameters must be a list.")

    def operation(params, circuit, state):
        backend.set_threads(backend.nthreads)
        if state is not None:
            state = backend.cast(state, copy=True)
        circuit.set_parameters(params)
        return backend.execute_circuit(circuit, state)

    results = Parallel(n_jobs=processes, prefer="threads")(
        delayed(operation)(param, circuit.copy(deep=True), initial_state)
        for param in parameters
    )

    return results
