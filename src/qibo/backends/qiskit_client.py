from qiskit import QuantumCircuit
from qiskit_ibm_runtime import QiskitRuntimeService, Sampler, Session

from qibo.backends import NumpyBackend
from qibo.result import MeasurementOutcomes


class QiskitClientBackend(NumpyBackend):
    """Backend for the remote execution of Qiskit circuits on the IBM servers.

    Args:
        platform (str): The IBM platform.
        token (str): User authentication token.
        runcard (dict): A dictionary containing the settings for the execution:
        - backend (str): One of the backends supported by the platform.
    """

    def __init__(self, token, platform="ibm_cloud", runcard=None):
        super().__init__()
        self.service = QiskitRuntimeService(channel=platform, token=token)
        if not runcard:
            runcard = {"backend": "ibmq_qasm_simulator"}
        self.backend = runcard["backend"]

    def execute_circuit(self, circuit, nshots=1000):
        measurements = circuit.measurements
        circuit = QuantumCircuit.from_qasm_str(circuit.to_qasm())
        with Session(self.service, backend=self.backend) as session:
            sampler = Sampler(session=session)
            job = sampler.run(circuit, shots=nshots)
            samples = job.result()
        return MeasurementOutcomes(measurements, samples=samples, nshots=nshots)
