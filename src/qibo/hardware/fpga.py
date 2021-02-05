import numpy as np
import paramiko
from io import BytesIO
from qibo.hardware import static


class IcarusQ:

    def __init__(self, address, username, password):
        self.nchannels = static.nchannels
        self.ssh = paramiko.SSHClient()
        self.ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        self.ssh.connect(hostname=address, username=username, password=password)

    def clock(self):
        self.ssh.exec_command('clk-control')

    def start(self, adc_delay=0.0, verbose=False):
        stdin, stdout, stderr = self.ssh.exec_command(
            'cd /tmp; ./cqtaws 1 {:.06f}'.format(adc_delay * 1e6))  # delay in us
        if verbose:
            for line in stdout:
                print(line.strip('\n'))

    def stop(self):
        self.ssh.exec_command('cd /tmp; ./cqtaws 0 0')

    def upload(self, waveform):
        sftp = self.ssh.open_sftp()
        dump = BytesIO()
        for i in range(self.nchannels):
            dump.seek(0)
            np.savetxt(dump, waveform[i], fmt='%d', newline=',')
            dump.seek(0)
            sftp.putfo(dump, '/tmp/wave_ch{}.csv'.format(i + 1))
        sftp.close()
        dump.close()

    def download(self):
        waveform = np.zeros((self.nchannels, static.sample_size))
        sftp = self.ssh.open_sftp()
        dump = BytesIO()
        for i in range(self.nchannels):
            dump.seek(0)
            #sftp.get('/tmp/ADC_CH{}.txt'.format(i + 1), local + 'ADC_CH{}.txt'.format(i + 1))
            sftp.getfo('/tmp/ADC_CH{}.txt'.format(i + 1), dump)
            dump.seek(0)
            #waveform.append(np.genfromtxt(local + 'ADC_CH{}.txt', delimiter=',')[:-1])
            waveform[i] = np.genfromtxt(dump, delimiter=',')[:-1]

        sftp.close()
        dump.close()

        return waveform
