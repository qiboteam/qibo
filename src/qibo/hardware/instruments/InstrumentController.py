"""
CLASS FILE FOR INSTRUMENT COMMUNICATION AND UTILITY
"""

from typing import Union, Optional, Any, Tuple
import logging
import numpy as np
import broadbean as bb
from .quicsyn import QuicSyn
from .attenuator import MCAttenuator
from .awg import AWG
from .rigol import RG

logger = logging.getLogger(__name__)

class InstrumentController():
    """
    InstrumentController class to interface and provide shortcut to instrument functions
    """
    def __init__(self, expected_waveform_count: Optional[int] = 450) -> None:
        self.awg = AWG()
        self.RFsource_RO = QuicSyn("ASRL3::INSTR")
        self.qubit_attenuator = MCAttenuator("192.168.0.9:90")
        self.readout_attenuator = MCAttenuator("192.168.0.10:100")
        self.RG = RG()
        logger.info("All instruments connected")
        self.expected_waveform_count = expected_waveform_count

    def setup(self,
              awg_parameters: dict, readout_frequency: Union[int, float],
              qubit_attenuation: int, readout_attenuation: int,
              flux_offset: Union[int, float]) -> None:
        """
        Setup the instruments
        """
        self.awg.setup(**awg_parameters)
        self.RFsource_RO.setup(readout_frequency)
        self.qubit_attenuator.set_attenuation(qubit_attenuation)
        self.readout_attenuator.set_attenuation(readout_attenuation)
        self.RG.set_voltage(flux_offset)

    def generate_broadbean_sequence(self, i_readout: Any, q_readout: Any, i_qubit: Any, q_qubit: Any,
                                    steps: int, osci_ttl: Any, switch_ttl: Any, qubit_ttl: Any, delay_time: int, averaging: int,
                                    sampling_rate: Union[int, float] = 2.3e9) -> Any:
        """
        Convert waveforms into Broadbean sequence, upload and load into AWG
        """
        
        # Create delay waveform
        delay_time = int(delay_time / 1.5)
        sample_delay = int(1.5e-6 * sampling_rate)
        wfm_delay = np.zeros((sample_delay), dtype=float)
        # Format delay waveform into broadbean wave element
        delay_element = bb.Element()
        for ch in range(1, 5):
            delay_element.addArray(ch, wfm_delay, sampling_rate, m1=wfm_delay, m2=wfm_delay)

        logger.info("Sequence generation started")
        # Create main sequence
        mainseq = bb.Sequence()
        mainseq.name = "MainSeq"
        mainseq.setSR(sampling_rate)
        for j in range(steps):

            # Create subsequence, each subsequence is a step of the main sequence
            subseq = bb.Sequence()
            subseq.setSR(sampling_rate)

            # Create broadbean wave element to hold qubit and readout signals
            subseq_element = bb.Element()
            # Set signals into wave element
            subseq_element.addArray(1, i_readout, sampling_rate, m1=osci_ttl, m2=switch_ttl)
            subseq_element.addArray(2, q_readout, sampling_rate, m1=osci_ttl, m2=qubit_ttl)
            subseq_element.addArray(3, i_qubit[j], sampling_rate, m1=osci_ttl, m2=switch_ttl)
            subseq_element.addArray(4, q_qubit[j], sampling_rate, m1=osci_ttl, m2=switch_ttl)

            # Set subsequence to play signals first, then place a delay afterwards
            subseq.addElement(1, subseq_element)
            subseq.addElement(2, delay_element)
            subseq.setSequencingNumberOfRepetitions(1, 1)
            # Repeat 1us delay waveform as defined by delay_time
            subseq.setSequencingNumberOfRepetitions(2, delay_time)

            # Set subsequence in step j + 1 (as steps begin from 1 and not 0 in AWG)
            mainseq.addSubSequence(j + 1, subseq)
            # Set subsequence to wait for trigger A
            mainseq.setSequencingTriggerWait(j + 1, 1)
            # Set subsequence to repeat 450 times
            mainseq.setSequencingNumberOfRepetitions(j + 1, averaging + 10)
        # If this is the final step, set a Goto to move to the first step afterwards
        mainseq.setSequencingGoto(j + 1, 1)
        logger.info("Sequence generation complete")
        return mainseq.forge(apply_delays=False, apply_filters=False)

    def generate_pulse_sequence(self, i_readout: Any, q_readout: Any, i_qubit: Any, q_qubit: Any,
                                osci_ttl: Any, switch_ttl: Any, qubit_ttl: Any, delay_time: int, averaging: int,
                                sampling_rate: Union[int, float] = 2.5e9) -> Any:
        """
        Convert waveforms into Broadbean sequence, upload and load into AWG
        """
        
        # Create delay waveform
        delay_time = int(delay_time / 1.5)
        sample_delay = int(1.5e-6 * sampling_rate)
        wfm_delay = np.zeros((sample_delay), dtype=float)
        # Format delay waveform into broadbean wave element
        delay_element = bb.Element()
        for ch in range(1, 5):
            delay_element.addArray(ch, wfm_delay, sampling_rate, m1=wfm_delay, m2=wfm_delay)

        logger.info("Sequence generation started")
        # Create main sequence
        mainseq = bb.Sequence()
        mainseq.name = "MainSeq"
        mainseq.setSR(sampling_rate)
        for j in range(1):

            # Create subsequence, each subsequence is a step of the main sequence
            subseq = bb.Sequence()
            subseq.setSR(sampling_rate)

            # Create broadbean wave element to hold qubit and readout signals
            subseq_element = bb.Element()
            # Set signals into wave element
            subseq_element.addArray(1, i_readout, sampling_rate, m1=osci_ttl, m2=switch_ttl)
            subseq_element.addArray(2, q_readout, sampling_rate, m1=osci_ttl, m2=qubit_ttl)
            subseq_element.addArray(3, i_qubit, sampling_rate, m1=osci_ttl, m2=switch_ttl)
            subseq_element.addArray(4, q_qubit, sampling_rate, m1=osci_ttl, m2=switch_ttl)

            # Set subsequence to play signals first, then place a delay afterwards
            subseq.addElement(1, subseq_element)
            subseq.addElement(2, delay_element)
            subseq.setSequencingNumberOfRepetitions(1, 1)
            # Repeat 1us delay waveform as defined by delay_time
            subseq.setSequencingNumberOfRepetitions(2, delay_time)

            # Set subsequence in step j + 1 (as steps begin from 1 and not 0 in AWG)
            mainseq.addSubSequence(j + 1, subseq)
            # Set subsequence to wait for trigger A
            mainseq.setSequencingTriggerWait(j + 1, 1)
            # Set subsequence to repeat 450 times
            mainseq.setSequencingNumberOfRepetitions(j + 1, averaging)
        # If this is the final step, set a Goto to move to the first step afterwards
        mainseq.setSequencingGoto(j + 1, 1)
        logger.info("Sequence generation complete")
        return mainseq.forge(apply_delays=False, apply_filters=False)

    def generate_gauss_broadbean_sequence(self):
        pass

    def ready_instruments_for_scanning(self, qubit_attenuation: int, readout_attenuation: int,
                                       flux_offset: Union[int, float]) -> float:
        """
        Readies instruments for measurements
        Equivalent to previous Runfile.__init()
        Returns oscilloscope parameters
        """
        logger.info("Setting up instruments for scanning")
        self.RG.set_voltage(flux_offset)
        self.RG.start()
        
        # Fetch the dispersive peak
        #readout_frequency = self._get_dispersive_peak()
        #self.RFsource_RO.set_frequency(readout_frequency - readout_IF)

        self.RFsource_RO.start()
        self.readout_attenuator.set_attenuation(readout_attenuation)
        self.qubit_attenuator.set_attenuation(qubit_attenuation)
        self.awg.ready()
        logger.info("All instruments ready for scanning")

        #return readout_frequency

    
    """ def scan(self, current_step: int) -> Tuple[Any]:
        
        Performs a scan and returns a tuple of I_raw_data and Q_raw data
        
        self.osc.clear_sweeps()
        self.awg.trigger() # Trigger the AWG

        self.awg.has_completed_current_step(current_step)
        self.osc.check_waveforms_captured(self.expected_waveform_count)

        return self.osc.fetch_data() """

    def stop(self):
        """
        Stops and turns off all instrument
        """
        self.awg.stop()
        self.RFsource_RO.stop()
        #self.RG.stop()
        logger.info("All instruments stopped")

    def atexit(self):
        """
        Close instrument connections and fill warning log
        """
        self.awg.awg.close()
        self.RFsource_RO.synt.close()
        self.RG.RG.close()
        logger.info("All instrument connections closed")
