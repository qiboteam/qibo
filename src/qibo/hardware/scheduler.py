import numpy as np
from concurrent.futures import ThreadPoolExecutor, Future
from qibo.config import raise_error
from qibo.hardware import fpga, pulses, static
from qibo.hardware.circuit import PulseSequence


class TaskScheduler:
    """Scheduler class for organizing FPGA calibration and pulse sequence execution."""

    def __init__(self, address, username, password):
        self._fpga = IcarusQ(address, username, password)
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._pi_trig = None # NIY
        sampling_rate = static.sampling_rate
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

    def execute_pulse_sequence(self, pulse_sequence, shots):
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
        if not isinstance(shots, int) or shots < 1:
            raise_error(ValueError, "Invalid number of shots {}.".format(nshots))
        future = self._executor.submit(self._execute_pulse_sequence, args=(pulse_sequence, shots))
        return future

    def _execute_pulse_sequence(self, pulse_sequence, shots):
        wfm = pulse_sequence.compile()
        self._fpga.upload(wfm)
        self._fpga.start()
        # NIY
        #self._pi_trig.trigger(shots, delay=50e6)
        # OPC?
        self._fpga.stop()
        res = self._fpga.download()
        return res
