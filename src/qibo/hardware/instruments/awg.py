"""
UTILITY CLASS FOR INTERFACING WITH THE TEKTRONIX AWG5204
"""

import time
import os
import subprocess
import logging
from typing import List, Union, Optional, Any, Tuple
import visa
from qcodes.instrument_drivers.tektronix.AWG70000A import AWG70000A

logger = logging.getLogger(__name__)

class AWG():
    """
    AWG Class to interface with the AWG5204
    """
    def __init__(self, address: Optional[str] = 'TCPIP0::192.168.0.2::inst0::INSTR',
                 timeout: Optional[int] = 240 * 1000) -> None:
        rm = visa.ResourceManager()
        self.awg = rm.open_resource(address)
        self.awg.timeout = timeout

    def setup(self,
              offset: List[Union[int, float]],
              amplitude: Optional[Tuple[Union[int, float]]] = (0.75, 0.75, 0.75, 0.75),
              resolution: Optional[int] = 14,
              sampling_rate: Optional[Union[int, float]] = 2.3e9,
              *args, **kwargs) -> None:
        '''
        Function for setting up the AWG
        Input Para: Offset list
        Optional Input: Amplitude list, Resolution, sampling rate
        '''

        self._reset()

        channels = range(1, 5)
        for ch in channels:
            self.awg.write("SOURCe{}:VOLTage {}".format(ch, amplitude[ch - 1]))
            self.awg.write("SOURCE{}:VOLTAGE:LEVEL:IMMEDIATE:OFFSET {}".format(ch, offset[ch - 1]))
            self.awg.write("SOURce{}:DAC:RESolution {}".format(ch, resolution))

        self.awg.write("CLOCk:SRATe {}".format(sampling_rate))
        self.awg.query("*OPC?")
        logger.info("AWG setup complete")

    def set_nyquist_mode(self) -> None:
        # 3rd nyquist zone mix mode
        self.awg.write("SOUR3:DMOD MIX")
        self.awg.write("SOUR4:DMOD MIX")

    def set_mixer_mode(self) -> None:
        # Set back to NRZ for mixer
        self.awg.write("SOUR3:DMOD NRZ")
        self.awg.write("SOUR3:DMOD NRZ")

    def _clearLists(self) -> None:
        """
        Deletes the AWG sequence and waveform lists
        """
        self.awg.write('SLISt:SEQuence:DELete ALL')
        self.awg.write('WLISt:WAVeform:DELete ALL')

    def _reset(self) -> None:
        """
        Resets the AWG
        """
        self.awg.write("INSTrument:MODE AWG")
        self._clearLists()
        self.awg.write("CLOC:SOUR EFIX") # Set AWG to external reference, 10 MHz
        self.awg.write("CLOC:OUTP:STAT OFF") # Disable clock output

    def _fetch_AWG_amplitudes(self) -> List[Union[int, float]]:
        """
        Returns a list of each channel amplitude of the AWG
        """
        return [float(self.awg.query("SOURce{}:VOLTage?".format(ch))) for ch in range(1, 5)]

    def upload_sequence(self, output: Any, expected_steps: int) -> None:
        """
        Function for uploading and setting the AWG sequence
        Input para: Output of Sequence.forge() of the main sequence,
                    Expected number of steps the AWG sequence should have
        """

        # Generate the SEQX file of the mains equence
        output_file = AWG70000A.make_SEQX_from_forged_sequence(output,
                                                               self._fetch_AWG_amplitudes(),
                                                               "MainSeq")

        # Write the file directly to the AWG
        with open("//192.168.0.2/Users/OEM/Documents/MainSeq.seqx", "wb+") as w:
            w.write(output_file)

        # Write it into a file and transfer it via Windows Robocopy
        """ with open("./filesend.bat", "w+") as w:
            w.write('net use \\\\192.168.0.2\IPC$ oem  /USER:"OEM" \n')
            w.write('robocopy {}\seq \\\\192.168.0.2\\Users\\OEM\\Documents'.format(os.getcwd()))   
        p = subprocess.Popen("filesend.bat", shell=True)
        p.wait() """

        # Load the sequence
        try:
            pathstr = 'C:' + "\\Users\\OEM\\Documents" + '\\' + "MainSeq.seqx"
            self.awg.write('MMEMory:OPEN:SASSet:SEQuence "{}"'.format(pathstr))
            # the above command is overlapping, but we want a blocking command
            self.awg.query('*OPC?')
        # We expect a timeout if the loading takes too long
        except:
            # Wait for 15 more timeouts
            for counter in range(15):
                try:
                    if self.awg.query('*OPC?'):
                        break
                except:
                    pass
            else:
                raise Exception("AWG took too long to load waveform")
        logger.info("AWG waveform loaded")
        # Check that correct sequence file is loaded
        if int(self.awg.query('SLISt:SEQuence:LENGth? "MainSeq"')) != expected_steps:
            raise Exception("Incorrect sequence file loaded into AWG")

        #Assign the Track to the corresponding Channel
        for ch in range(1, 5):
            self.awg.write('SOURCE{}:CASSet:SEQuence "MainSeq", {}'.format(ch, ch))
        self.awg.query("*OPC?")
        logger.info("AWG waveform set")

    def ready(self) -> None:
        """
        Turns on AWG output channels and readies system for scanning
        """
        for ch in range(1, 5):
            self.awg.write("OUTPut{}:STATe 1".format(ch))
            self.awg.write('SOURce{}:RMODe TRIGgered'.format(ch))
            self.awg.write('SOURce1{}TINPut ATRIGGER'.format(ch))

        # Arm the trigger
        self.awg.write('AWGControl:RUN:IMMediate')
        self.awg.query('*OPC?')
        counter = 0
        for counter in range(120):
            if self.awg.query("SOURCE:SCSTEP?") == '"subsequence_1.1"\n':
                break
            counter = counter + 1
            time.sleep(1)
        else:
            raise Exception("AWG took too long to ready")
        logger.info("AWG ready for scanning")

    def trigger(self) -> None:
        """
        Triggers the AWG
        """
        self.awg.write('TRIGger:IMMediate ATRigger')

    def _is_on_expected_step(self, expected_step: int) -> bool:
        """
        Check if AWG is on the expected step
        """
        return self.awg.query("SOURCE:SCSTEP?") == '"subsequence_{}.1"\n'.format(expected_step)

    def _is_waiting_for_trigger(self) -> bool:
        """
        Check if AWG is waiting for trigger
        """
        return int(self.awg.query("AWGControl:RSTate?")) == 1

    def has_completed_current_step(self, current_step: int) -> bool:
        """
        Checks if the AWG has finished outputting the current step
        """
        expected_step = current_step + 1
        is_on_expected_step = False
        is_stopped = False
        for counter in range(45):
            if self._is_on_expected_step(expected_step):
                is_on_expected_step = True
                break
            time.sleep(0.1)
        else:
            logger.warn("AWG step count timeout")

        for counter in range(30):
            if self._is_waiting_for_trigger():
                is_stopped = True
                break
        else:
            logger.warn("AWG runstate error")

        if is_on_expected_step and not is_stopped:
            return Exception("Misaligned step")
        return True

    def stop(self) -> None:
        """
        Stops and turns off the AWG channels
        """
        self.awg.write('AWGControl:STOP')
        for ch in range(1, 5):
            self.awg.write("OUTPut{}:STATe 0".format(ch))
        logger.info("AWG stopped")
