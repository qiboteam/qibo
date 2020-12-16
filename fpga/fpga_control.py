import os
import numpy as np
import paramiko


class IcarusQ:
    def __init__(self, address, username, password, n_channels):
        self.n_channels = n_channels
        self.ssh = paramiko.SSHClient()
        self.ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        self.ssh.connect(hostname=address, username=username, password=password)

    def clock(self):
        self.ssh.exec_command('clk-control')

    def start(self, adc_delay=0.0, verbose=True):
        stdin, stdout, stderr = self.ssh.exec_command(
            'cd /tmp; ./cqtaws 1 {:.06f}'.format(adc_delay * 1e6))  # delay in us
        if verbose:
            for line in stdout:
                print(line.strip('\n'))

    def stop(self):
        self.ssh.exec_command('cd /tmp; ./cqtaws 0 0')

    def upload(self, waveform):
        sftp = self.ssh.open_sftp()
        local = './tmp/'
        if not os.path.exists(local):
            os.makedirs(local)
        for i in range(self.n_channels):
            np.savetxt(local + 'wave_ch{}.csv'.format(i + 1), waveform[i], fmt='%d', newline=',')
            sftp.put(local + 'wave_ch{}.csv'.format(i + 1), '/tmp/wave_ch{}.csv'.format(i + 1))
        sftp.close()

    def download(self):
        waveform = []
        sftp = self.ssh.open_sftp()
        local = './tmp/'
        if not os.path.exists(local):
            os.makedirs(local)
        for i in range(self.n_channels):
            sftp.get('/tmp/ADC_CH{}.txt'.format(i + 1), local + 'ADC_CH{}.txt'.format(i + 1))
            waveform.append(np.genfromtxt(local + 'ADC_CH{}.txt', delimiter=',')[:-1])
        sftp.close()
        return np.array(waveform)
