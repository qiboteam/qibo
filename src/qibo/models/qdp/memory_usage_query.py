import numpy as np
import scipy

from qibo import gates
from qibo.transpiler.unitary_decompositions import two_qubit_decomposition
from qibo.models.qdp.quantum_dynamic_programming import (
    QDPSequentialInstruction,
    QDPMeasurementEmulation,
    QDPMeasurementReset
    )


class DensityMatrixExponentiation(QDPSequentialInstruction):
    """
    Subclass of AbstractQuantumDynamicProgramming for density matrix exponentiation,
    where we attempt to instruct the work qubit to do an X rotation, using SWAP gate.

    Args:
        theta (float): Overall rotation angle.
        N (int): Number of steps.
        num_work_qubits (int): Number of work qubits.
        num_instruction_qubits (int): Number of instruction qubits.
        number_muq_per_call (int): Number of memory units per call.

    Example:
        import numpy as np
        from qibo.models.qdp.dynamic_programming import DensityMatrixExponentiation
        my_protocol = DensityMatrixExponentiation(theta=np.pi,N=3,num_work_qubits=1,num_instruction_qubits=3,number_muq_per_call=1)
        my_protocol.memory_call_circuit(num_instruction_qubits_per_query=3)
        print('DME, q0 is target qubit, q1,q2 and q3 are instruction qubit')
        print(my_protocol.c.draw())
        my_protocol.c.execute(nshots=1000).frequencies()
    """

    def __init__(
        self, theta, N, num_work_qubits, num_instruction_qubits, number_muq_per_call
    ):
        super().__init__(
            num_work_qubits, num_instruction_qubits, number_muq_per_call, circuit=None
        )
        self.theta = theta  # overall rotation angle
        self.N = N  # number of steps
        self.delta = theta / N  # small rotation angle
        self.id_current_work_reg = self.list_id_work_reg[0]

    def memory_usage_query_circuit(self):
        """Defines the memory usage query circuit."""
        delta_swap = scipy.linalg.expm(
            -1j
            * gates.SWAP(
                self.id_current_work_reg, self.id_current_instruction_reg
            ).matrix()
            * self.delta
        )
        for decomposed_gate in two_qubit_decomposition(
            self.id_current_work_reg,
            self.id_current_instruction_reg,
            unitary=delta_swap,
        ):
            self.c.add(decomposed_gate)

    def instruction_qubits_initialization(self):
        """Initializes the instruction qubits."""
        for instruction_qubit in self.list_id_current_instruction_reg:
            self.c.add(gates.X(instruction_qubit))


class DME_reset(QDPMeasurementReset):
    """
    Warning: Functional, but without a way to actually do reset.
    DME using reset method.
    """

    def __init__(
        self, theta, N, num_work_qubits, num_instruction_qubits, number_muq_per_call
    ):
        super().__init__(
            num_work_qubits, num_instruction_qubits, number_muq_per_call, circuit=None
        )
        self.theta = theta  # overall rotation angle
        self.N = N  # number of steps
        self.delta = theta / N  # small rotation angle
        self.id_current_work_reg = self.list_id_work_reg[0]

    def current_register_reset(self):
        """
        Resets a single register.

        Args:
            register (int): The register index.
            _c = self.c.copy()
        """
        # todo: find a way to do reset
        # result = _c.execute(nshots=1).samples(binary=True)[0][0]
        result = 1
        if result == 1:
            self.c.add(gates.X(self.id_current_instruction_reg))
        elif result == 0:
            pass
        else:
            print("Warning: qubit wasn't reset")

    def memory_usage_query_circuit(self):
        """Defines the memory usage query circuit."""
        delta_SWAP = scipy.linalg.expm(
            -1j
            * gates.SWAP(
                self.id_current_work_reg, self.id_current_instruction_reg
            ).matrix()
            * self.delta
        )
        for decomposed_gate in two_qubit_decomposition(
            self.id_current_work_reg,
            self.id_current_instruction_reg,
            unitary=delta_SWAP,
        ):
            self.c.add(decomposed_gate)

    def instruction_qubits_initialization(self):
        """Initializes the instruction qubits."""
        for instruction_qubit in self.list_id_current_instruction_reg:
            self.c.add(gates.X(instruction_qubit))
