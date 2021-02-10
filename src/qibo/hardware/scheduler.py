import numpy as np
from concurrent.futures import ThreadPoolExecutor, Future
from qibo.config import raise_error
from qibo.hardware import pulses, experiment
from qibo.hardware.circuit import PulseSequence


class TaskScheduler:
    """Scheduler class for organizing FPGA calibration and pulse sequence execution."""

    def __init__(self):
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._pi_trig = None # NIY
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

    @staticmethod
    def _execute_pulse_sequence(pulse_sequence, nshots):
        wfm = pulse_sequence.compile()
        experiment.upload(wfm)
        experiment.start()
        # NIY
        #self._pi_trig.trigger(shots, delay=50e6)
        # OPC?
        experiment.stop()
        res = experiment.download()
        return res
