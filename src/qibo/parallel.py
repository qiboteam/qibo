"""
Resources for parallel circuit evaluation.
"""


class ParallelResources:  # pragma: no cover
    """Auxiliary singleton class for sharing memory objects in a
    multiprocessing environment when performing a parallel_L-BFGS-B
    minimization procedure.

    This class takes care of duplicating resources for each process
    and calling the respective loss function.
    """
    from sys import platform
    import multiprocessing as mp
    if platform == 'darwin':
        mp.set_start_method('fork')  # enforce on Darwin

    # private objects holding the state
    _instance = None
    # dict with of shared objects
    _objects_per_process = {}
    custom_function = None
    lock = None
    arguments = ()

    def __new__(cls, *args, **kwargs):
        """Creates singleton instance."""
        if cls._instance is None:
            cls._instance = super(ParallelResources, cls).__new__(
                cls, *args, **kwargs)
        return cls._instance

    def run(self, params=None):
        """Computes loss (custom_loss) for a specific set of parameters.
        This class performs the lock mechanism to duplicate objects
        for each process.
        """
        # lock to avoid race conditions
        self.lock.acquire()
        # get process name
        pname = self.mp.current_process().name
        # check if there are objects already stored
        args = self._objects_per_process.get(pname, None)
        if args is None:
            args = []
            for obj in self.arguments:
                try:
                    # copy object if copy method is available
                    copy = obj.copy(deep=True)
                except AttributeError:
                    # if copy is not implemented just use the original object
                    copy = obj
                except Exception as e:
                    # use print otherwise the message will not appear
                    print('Exception in ParallelResources', str(e))
                args.append(copy)
            args = tuple(args)
            self._objects_per_process[pname] = args
        # unlock
        self.lock.release()
        # finally compute the loss function
        return self.custom_function(params, *args)

    def reset(self):
        """Cleanup memory."""
        self._objects_per_process = {}
        self.custom_loss = None
        self.lock = None
        self.args = ()


def _executor(params):
    """Executes singleton call."""
    return ParallelResources().run(params)


def parallel_execution(circuit, states=None, processes=None):
    """Execute circuit for multiple states.

    Args:
        circuit (qibo.models.Circuit): the input circuit.
        states (list): list of states for the circuit evaluation.
        processes (int): number of processes for parallel evaluation.

    Returns:
        Circuit evaluation for input states.
    """
    if states is None or not isinstance(states, list):  # pragma: no cover
        raise_error(RuntimeError, "states must not be empty.")

    _check_parallel_configuration(processes)

    def operation(state, circuit):
        return circuit(state)

    ParallelResources().arguments = (circuit,)
    ParallelResources().custom_function = operation

    import multiprocessing as mp
    ParallelResources().lock = mp.Lock()
    with mp.Pool(processes=processes) as pool:
        results = pool.map(_executor, states)

    ParallelResources().reset()
    return results


def parallel_reuploading_execution(circuit, parameters=None, initial_state=None, processes=None):
    """Execute circuit for multiple parameters and fixed initial_state.

    Args:
        circuit (qibo.models.Circuit): the input circuit.
        parameters (list): list of parameters for the circuit evaluation.
        initial_state (np.array): initial state for the circuit evaluation.
        processes (int): number of processes for parallel evaluation.

    Returns:
        Circuit evaluation for input parameters.
    """
    if parameters is None or not isinstance(parameters, list):  # pragma: no cover
        raise_error(RuntimeError, "parameters must not be empty.")

    _check_parallel_configuration(processes)

    def operation(params, circuit, state):
        circuit.set_parameters(params)
        return circuit(state)

    ParallelResources().arguments = (circuit, initial_state)
    ParallelResources().custom_function = operation

    import multiprocessing as mp
    ParallelResources().lock = mp.Lock()
    with mp.Pool(processes=processes) as pool:
        results = pool.map(_executor, parameters)

    ParallelResources().reset()
    return results


def _check_parallel_configuration(processes):
    """Check if configuration is suitable for efficient parallel execution."""
    import psutil
    from qibo.config import raise_error, get_device, get_threads, log
    if "GPU" in get_device():  # pragma: no cover
        raise_error(RuntimeError, "Parallel L-BFGS-B cannot be used with GPU.")
    if ((processes is not None and processes * get_threads() > psutil.cpu_count()) or
            (processes is None and get_threads() != 1)):  # pragma: no cover
        log.warning('Please consider using a lower number of threads per process,'
                    ' or reduce the number of processes for better performance')
