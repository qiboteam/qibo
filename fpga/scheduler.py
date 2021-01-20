from concurrent.futures import ThreadPoolExecutor, Future
from typing import Union
from qibo.config import raise_error
import numpy as np
import pulse_abstraction as pa
from fpga_control import IcarusQ
from login import address, username, password
from static_config import sampling_rate

class TaskScheudler:
    """Scheduler class for organizing FPGA calibration and pulse sequence execution

    """
    def __init__(self):
        self._ready = True
        self._fpga = IcarusQ(address, username, password)
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._pi_trig = None # NIY
        self._qubit_config = [{ # placeholder, default none
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
                "rx": [pa.BasicPulse(3, 0, 24.78e-9, 0.375, 3.0473825e9 - sampling_rate, 0, pa.Rectangular())],
                "ry": [pa.BasicPulse(3, 0, 24.78e-9, 0.375, 3.0473825e9 - sampling_rate, 90, pa.Rectangular())],
            }
        }]

    def fetch_config(self) -> Union[list, bool]:
        """Fetches the qubit configuration data

        Returns:
            List of dicts representing qubit metadata or false if data is not ready yet
        """
        if not self.config_ready():
            return False
        else:
            return self._qubit_config

    def poll_config(self):
        """Blocking command to wait until qubit calibration is complete
        
        """
        raise_error(NotImplementedError)

    def config_ready(self) -> bool:
        """Checks if qubit calibration is complete

        Returns:
            Boolean flag representing status of qubit calibration complete
        """
        return self._ready

    def execute_pulse_sequence(self, pulse_sequence: pa.PulseSequence, shots: int) -> Future:
        """
        Submits a pulse sequence to the queue for execution

        Args:
            pulse_sequence: Pulse sequence object defined in pulse_abstraction
            shots: Number of trials

        Returns:
            concurrent.futures.Future object representing task status
        """
        if not isinstance(pulse_sequence, pa.PulseSequence) or not isinstance(shots, int) or shots < 1:
            raise_error(ValueError, "Pulse sequence or number of shots incorrectly defined")

        future = self._executor.submit(self._execute_pulse_sequence, args=(pulse_sequence, shots))
        return future

    def _execute_pulse_sequence(self, pulse_sequence: pa.PulseSequence, shots: int) -> np.ndarray:
        wfm = pulse_sequence.compile()
        self._fpga.upload(wfm)
        self._fpga.start()
        # NIY
        #self._pi_trig.trigger(shots, delay=50e6)
        # OPC?
        self._fpga.stop()
        res = self._fpga.download()
        return res
