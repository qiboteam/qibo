import numpy as np
from abc import ABC, abstractmethod
from io import BytesIO
from qibo.hardware import connections
from qibo.config import raise_error


class Experiment(ABC):

    def __init__(self):
        self._connection = None
        self.static = None

    @property
    def connection(self):
        if self._connection is None:
            raise_error(RuntimeError, "Cannot establish connection.")
        return self.connection

    @abstractmethod
    def connect(self):
        raise_error(NotImplementedError)

    @abstractmethod
    def start(self):
        raise_error(NotImplementedError)

    @abstractmethod
    def stop(self):
        raise_error(NotImplementedError)

    @abstractmethod
    def upload(self):
        raise_error(NotImplementedError)

    @abstractmethod
    def download(self):
        raise_error(NotImplementedError)


class IcarusQ(Experiment):

    class StaticParameters():
        """Hardware static parameters."""
        num_qubits = 2
        sampling_rate = 2.3e9
        nchannels = 4
        sample_size = 32000
        readout_pulse_type = "IQ"
        readout_pulse_duration = 5e-6
        readout_pulse_amplitude = 0.75
        lo_frequency = 4.51e9
        readout_nyquist_zone = 4
        ADC_sampling_rate = 2e9
        qubit_static_parameters = [
            {
                "id": 0,
                "channel": [2, None, [0, 1]], # XY control, Z line, readout
                "frequency_range": [2.6e9, 2.61e9],
                "resonator_frequency": 4.5241e9,
                "neighbours": [2]
            }, {
                "id": 1,
                "channel": [3, None, [0, 1]],
                "frequency_range": [3.14e9, 3.15e9],
                "resonator_frequency": 4.5241e9,
                "neighbours": [1]
            }
        ]
        dac_mode_for_nyquist = ["NRZ", "MIX", "MIX", "NRZ"] # fifth onwards not calibrated yet
        pulse_file = 'C:/fpga_python/fpga/tmp/wave_ch1.csv'

        # Temporary calibration result placeholder when actual calibration is not available
        from qibo.hardware import pulses
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
                "rx": [pulses.BasicPulse(3, 0, 24.78e-9, 0.375, 3.0473825e9 - sampling_rate, 0, pulses.Rectangular())],
                "ry": [pulses.BasicPulse(3, 0, 24.78e-9, 0.375, 3.0473825e9 - sampling_rate, 90, pulses.Rectangular())],
            }
        }]

    def __init__(self):
        super().__init__()
        self.static = self.StaticParameters()

    def connect(self, address, username, password):
        self._connection = connections.ParamikoSSH(address, username, password)

    def clock(self):
        self.connection.exec_command('clk-control')

    def start(self, adc_delay=0.0, verbose=False):
        stdin, stdout, stderr = self.connection.exec_command(
            'cd /tmp; ./cqtaws 1 {:.06f}'.format(adc_delay * 1e6))  # delay in us
        if verbose:
            for line in stdout:
                print(line.strip('\n'))

    def stop(self):
        self.connection.exec_command('cd /tmp; ./cqtaws 0 0')

    def upload(self, waveform):
        dump = BytesIO()
        with self.connection as sftp:
            for i in range(self.static.nchannels):
                dump.seek(0)
                np.savetxt(dump, waveform[i], fmt='%d', newline=',')
                dump.seek(0)
                sftp.putfo(dump)
        dump.close()

    def download(self):
        waveform = np.zeros((self.static.nchannels, self.static.sample_size))
        dump = BytesIO()
        with self.connection as sftp:
            for i in range(self.static.nchannels):
                dump.seek(0)
                #sftp.get('/tmp/ADC_CH{}.txt'.format(i + 1), local + 'ADC_CH{}.txt'.format(i + 1))
                dump = sftp.getfo(dump)
                dump.seek(0)
                #waveform.append(np.genfromtxt(local + 'ADC_CH{}.txt', delimiter=',')[:-1])
                waveform[i] = np.genfromtxt(dump, delimiter=',')[:-1]

        sftp.close()
        dump.close()

        return waveform
