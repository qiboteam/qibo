"""
UTILITY CLASS FOR INTERFACING WITH THE QUICSYN LITE
"""

from typing import Optional, Union
import visa

class QuicSyn():
    def __init__(self, address: Optional[str] = 'ASRL4::INSTR',
                 timeout: Optional[Union[int, float]] = 10000):
        rm = visa.ResourceManager()
        self.synt = rm.open_resource(address, timeout=timeout)

    def setup(self, frequency: Union[int, float]) -> None:
        '''
        Function for setting up the quicsyn RF source
        Input para: Readout frequency (Hz)
        '''
        #Select reference source: INT ('0600'), EXT ('0601')
        #Ref source Query ('07')
        self.synt.write('0601')
        self.set_frequency(frequency)
        #On command: 0F01

    def set_frequency(self, frequency: Union[int, float]) -> None:
        """
        Sets the frequency
        """
        self.synt.write('FREQ {0:f}Hz'.format(frequency))

    def start(self) -> None:
        """
        Starts the QuicSyn
        """
        self.synt.write('0F01')

    def stop(self) -> None:
        """
        Stops the QuicSyn
        """
        self.synt.write('0F00')
