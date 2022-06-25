# -*- coding: utf-8 -*-
# @authors: S. Efthymiou
import sys
import math
import joblib
import collections
from qibo.config import raise_error
from qibo import gates, states
from qibo.models.circuit import Circuit
from qibo.models.distutils import DistributedQueues
from typing import Dict, List, Optional, Set, Tuple, Union


class DistributedCircuit(Circuit):
    """Distributed implementation of :class:`qibo.models.circuit.Circuit`.

    Uses multiple `accelerator` devices (GPUs) for applying gates to the state vector.
    The full state vector is saved in the given `memory device` (usually the CPU)
    during the simulation. A gate is applied by splitting the state to pieces
    and copying each piece to an accelerator device that is used to perform the
    matrix multiplication. An `accelerator` device can be used more than once
    resulting to logical devices that are more than the physical accelerators in
    the system.

    Distributed circuits currently do not support native tensorflow gates,
    compilation and callbacks.

    Example:
        .. code-block:: python

            from qibo.models import Circuit
            # The system has two GPUs and we would like to use each GPU twice
            # resulting to four total logical accelerators
            accelerators = {'/GPU:0': 2, '/GPU:1': 2}
            # Define a circuit on 32 qubits to be run in the above GPUs keeping
            # the full state vector in the CPU memory.
            c = Circuit(32, accelerators)

    Args:
        nqubits (int): Total number of qubits in the circuit.
        accelerators (dict): Dictionary that maps device names to the number of
            times each device will be used.
            The total number of logical devices must be a power of 2.
    """

    def __new__(cls, nqubits: int, accelerators: Dict[str, int]):
        return object().__new__(cls)

    def __init__(self, nqubits: int, accelerators: Dict[str, int]):
        super().__init__(nqubits, accelerators)
        self.ndevices = sum(accelerators.values())
        self.nglobal = float(math.log2(self.ndevices))

        if not (self.nglobal.is_integer() and self.nglobal > 0):
            raise_error(ValueError, "Number of calculation devices should be a power "
                                    "of 2 but is {}.".format(self.ndevices))
        self.nglobal = int(self.nglobal)
        self.nlocal = self.nqubits - self.nglobal
        
        self.queues = DistributedQueues(self, gates)

    def _set_nqubits(self, gate):
        AbstractCircuit._set_nqubits(self, gate)

    def on_qubits(self, *q):
        if self.queues.queues:
            raise_error(RuntimeError, "Cannot use distributed circuit as a "
                                      "subroutine after it was executed.")
        return super().on_qubits(*q)

    def copy(self, deep: bool = True):
        if not deep:
            raise_error(ValueError, "Non-deep copy is not allowed for distributed "
                                    "circuits because they modify gate objects.")
        return super().copy(deep)

    def fuse(self):
        raise_error(NotImplementedError, "Fusion is not implemented for "
                                         "distributed circuits.")

    def with_noise(self, noise_map, measurement_noise=None):
        raise_error(NotImplementedError, "Distributed circuit does not support "
                                         "density matrices yet.")

    def add(self, gate):
        if isinstance(gate, collections.abc.Iterable):
            for g in gate:
                self.add(g)
        
        else:
            if isinstance(gate, gates.KrausChannel):
                raise_error(NotImplementedError, "Distributed circuits do not "
                                                "support channels.")
            elif (self.nqubits - len(gate.target_qubits) < self.nglobal and
                not isinstance(gate, (gates.M, gates.VariationalLayer))):
                # Check if there is sufficient number of local qubits
                raise_error(ValueError, "Insufficient qubits to use for global in "
                                        "distributed circuit.")
            return super().add(gate)

    def compile(self):
        """"""
        raise_error(RuntimeError, "Cannot compile circuit that uses custom operators.")

    def execute(self, initial_state=None, nshots=None):
        """Equivalent to :meth:`qibo.models.circuit.Circuit.execute`.

        Returns:
            A :class:`qibo.core.states.DistributedState` object corresponding
            to the final state of execution. Note that this state contains the
            full state vector scattered to pieces and does not create a
            single tensor unless the user explicitly calls the ``tensor``
            property. This avoids creating multiple copies of large states in
            CPU memory.
        """
        from qibo.backends import GlobalBackend
        return GlobalBackend().execute_distributed_circuit(self, initial_state, nshots)
