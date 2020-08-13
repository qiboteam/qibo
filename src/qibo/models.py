from qibo.config import BACKEND_NAME, raise_error
if BACKEND_NAME != "tensorflow": # pragma: no cover
    # case not tested because backend is preset to TensorFlow
    raise_error(NotImplementedError, "Only Tensorflow backend is implemented.")
from qibo.tensorflow.circuit import TensorflowCircuit as SimpleCircuit
from qibo.tensorflow.distcircuit import TensorflowDistributedCircuit as DistributedCircuit
from qibo.evolution import StateEvolution, AdiabaticEvolution
from typing import Dict, Optional


class Circuit(DistributedCircuit):
    """Factory class for circuits.

    Creates both normal and distributed circuits.
    """

    def __new__(cls, nqubits: int,
                accelerators: Optional[Dict[str, int]] = None,
                memory_device: str = "/CPU:0"):
        if accelerators is None:
            return SimpleCircuit(nqubits)
        else:
            return DistributedCircuit(nqubits, accelerators, memory_device)

    @classmethod
    def from_qasm(cls, qasm_code: str,
                  accelerators: Optional[Dict[str, int]] = None,
                  memory_device: str = "/CPU:0"):
      if accelerators is None:
          return SimpleCircuit.from_qasm(qasm_code)
      else:
          return DistributedCircuit.from_qasm(qasm_code,
                                              accelerators=accelerators,
                                              memory_device=memory_device)


def QFT(nqubits: int, with_swaps: bool = True,
        accelerators: Optional[Dict[str, int]] = None,
        memory_device: str = "/CPU:0") -> Circuit:
    """Creates a circuit that implements the Quantum Fourier Transform.

    Args:
        nqubits (int): Number of qubits in the circuit.
        with_swaps (bool): Use SWAP gates at the end of the circuit so that the
            qubit order in the final state is the same as the initial state.
        accelerators (dict): Accelerator device dictionary in order to use a
            distributed circuit
            If ``None`` a simple (non-distributed) circuit will be used.
        memory_device (str): Device to use for memory in case a distributed circuit
            is used. Ignored for non-distributed circuits.

    Returns:
        A qibo.models.Circuit that implements the Quantum Fourier Transform.

    Example:
        ::

            import numpy as np
            from qibo.models import QFT
            nqubits = 6
            c = QFT(nqubits)
            # Random normalized initial state vector
            init_state = np.random.random(2 ** nqubits) + 1j * np.random.random(2 ** nqubits)
            init_state = init_state / np.sqrt((np.abs(init_state)**2).sum())
            # Execute the circuit
            final_state = c(init_state)
    """
    if accelerators is not None:
        if not with_swaps:
            raise_error(NotImplementedError, "Distributed QFT is only implemented "
                                             "with SWAPs.")
        return _DistributedQFT(nqubits, accelerators, memory_device)

    import numpy as np
    from qibo import gates

    circuit = Circuit(nqubits)
    for i1 in range(nqubits):
        circuit.add(gates.H(i1))
        for i2 in range(i1 + 1, nqubits):
            theta = np.pi / 2 ** (i2 - i1)
            circuit.add(gates.CZPow(i2, i1, theta))

    if with_swaps:
        for i in range(nqubits // 2):
            circuit.add(gates.SWAP(i, nqubits - i - 1))

    return circuit


def _DistributedQFT(nqubits: int,
                    accelerators: Optional[Dict[str, int]] = None,
                    memory_device: str = "/CPU:0") -> DistributedCircuit:
    """QFT with the order of gates optimized for reduced multi-device communication."""
    import numpy as np
    from qibo import gates

    circuit = Circuit(nqubits, accelerators, memory_device)
    icrit = nqubits // 2 + nqubits % 2
    if accelerators is not None:
        circuit.global_qubits = range(circuit.nlocal, nqubits)
        if icrit < circuit.nglobal:
            raise_error(NotImplementedError, "Cannot implement QFT for {} qubits "
                                             "using {} global qubits."
                                             "".format(nqubits, circuit.nglobal))

    for i1 in range(nqubits):
        if i1 < icrit:
            i1eff = i1
        else:
            i1eff = nqubits - i1 - 1
            circuit.add(gates.SWAP(i1, i1eff))

        circuit.add(gates.H(i1eff))
        for i2 in range(i1 + 1, nqubits):
            theta = np.pi / 2 ** (i2 - i1)
            circuit.add(gates.CZPow(i2, i1eff, theta))

    return circuit


class VQE(object):
    """This class implements the variational quantum eigensolver algorithm.

    Args:
        circuit (:class:`qibo.base.circuit.BaseCircuit`): Circuit that
            implements the variaional ansatz.
        hamiltonian (:class:`qibo.hamiltonians.Hamiltonian`): Hamiltonian object.

    Example:
        ::

            import numpy as np
            from qibo import gates, models, hamiltonians
            # create circuit ansatz for two qubits
            circuit = models.Circuit(2)
            circuit.add(gates.RY(q, theta=0))
            # create XXZ Hamiltonian for two qubits
            hamiltonian = hamiltonians.XXZ(2)
            # create VQE model for the circuit and Hamiltonian
            vqe = models.VQE(circuit, hamiltonian)
            # optimize using random initial variational parameters
            initial_parameters = np.random.uniform(0, 2, 1)
            vqe.minimize(initial_parameters)
    """
    from qibo import optimizers

    def __init__(self, circuit, hamiltonian):
        """Initialize circuit ansatz and hamiltonian."""
        self.circuit = circuit
        self.hamiltonian = hamiltonian

    def minimize(self, initial_state, method='Powell', options=None, compile=True):
        """Search for parameters which minimizes the hamiltonian expectation.

        Args:
            initial_state (array): a initial guess for the parameters of the
                variational circuit.
            method (str): the desired minimization method.
                One of ``"cma"`` (genetic optimizer), ``"sgd"`` (gradient descent) or
                any of the methods supported by `scipy.optimize.minimize <https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html>`_.
            options (dict): a dictionary with options for the different optimizers.

        Return:
            The final expectation value.
            The corresponding best parameters.
        """
        def loss(params):
            self.circuit.set_parameters(params)
            final_state = self.circuit()
            return self.hamiltonian.expectation(final_state)

        if compile:
            if not self.circuit.using_tfgates:
                raise_error(RuntimeError, "Cannot compile VQE that uses custom operators. "
                                          "Set the compile flag to False.")
            from qibo import K
            loss = K.function(loss)

        if method == 'sgd':
            # check if gates are using the MatmulEinsum backend
            from qibo.tensorflow.gates import TensorflowGate
            for gate in self.circuit.queue:
                if not isinstance(gate, TensorflowGate):
                    raise_error(RuntimeError, 'SGD VQE requires native Tensorflow '
                                              'gates because gradients are not '
                                              'supported in the custom kernels.')

            result, parameters = self.optimizers.optimize(loss, initial_state,
                                                          "sgd", options,
                                                          compile)
        else:
            result, parameters = self.optimizers.optimize(
                lambda p: loss(p).numpy(), initial_state, method, options)

        self.circuit.set_parameters(parameters)
        return result, parameters
