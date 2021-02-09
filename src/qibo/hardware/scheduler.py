import numpy as np
from concurrent.futures import ThreadPoolExecutor, Future
from qibo.config import raise_error
from qibo.hardware import pulses
from qibo.hardware.qpu import IcarusQ
from qibo.hardware.circuit import PulseSequence


class TaskScheduler:
    """Scheduler class for organizing FPGA calibration and pulse sequence execution."""
    # Temporary calibration result placeholder when actual calibration is not available
    calibration_placeholder = [{
        "id": 0,
        "qubit_frequency": 3.0473825e9,
        "qubit_amplitude": 0.75 / 2,
        "T1": 5.89e-6,
        "T2": 1.27e-6,
        "T2_Spinecho": 3.5e-6,
        "pi-pulse": 24.78e-9,
        "drive_channel": 3,
        "readout_channel": (0, 1),
        "iq_state": {
            "0": [0.016901687416102748, -0.006633150376482062],
            "1": [0.009458352995780546, -0.008570922209494462]
        },
        "gates": {
            "rx": [pulses.BasicPulse(3, 0, 24.78e-9, 0.375, 3.0473825e9 - IcarusQ.sampling_rate, 0, pulses.Rectangular())],
            "ry": [pulses.BasicPulse(3, 0, 24.78e-9, 0.375, 3.0473825e9 - IcarusQ.sampling_rate, 90, pulses.Rectangular())],
        }
    }]

    def __init__(self, qpu=None):
        self.qpu = qpu
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._pi_trig = None # NIY
        sampling_rate = IcarusQ.sampling_rate
        self._qubit_config = None

    def fetch_config(self):
        """Fetches the qubit configuration data

        Returns:
            List of dicts representing qubit metadata or false if data is not ready yet
        """
        if self._qubit_config is None:
            raise_error(RuntimeError, "Cannot fetch qubit configuration "
                                      "because calibration is not complete.")
        return self._qubit_config

    def poll_config(self):
        """Blocking command to wait until qubit calibration is complete."""
        raise_error(NotImplementedError)

    def config_ready(self):
        """Checks if qubit calibration is complete.

        Returns:
            Boolean flag representing status of qubit calibration complete
        """
        return self._qubit_config is not None

    def execute_pulse_sequence(self, pulse_sequence, nshots):
        """Submits a pulse sequence to the queue for execution.

        Args:
            pulse_sequence: Pulse sequence object.
            shots: Number of trials.

        Returns:
            concurrent.futures.Future object representing task status
        """
        if not isinstance(pulse_sequence, PulseSequence):
            raise_error(TypeError, "Pulse sequence {} has invalid type."
                                   "".format(pulse_sequence))
        if not isinstance(nshots, int) or nshots < 1:
            raise_error(ValueError, "Invalid number of shots {}.".format(nshots))
        future = self._executor.submit(self._execute_pulse_sequence,
                                       pulse_sequence=pulse_sequence,
                                       nshots=nshots)
        return future

    def _execute_pulse_sequence(self, pulse_sequence, nshots):
        wfm = pulse_sequence.compile()
        self.qpu.upload(wfm)
        self.qpu.start()
        # NIY
        #self._pi_trig.trigger(shots, delay=50e6)
        # OPC?
        self.qpu.stop()
        res = self.qpu.download()
        return res
