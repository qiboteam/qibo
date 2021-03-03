"""
CLASS TO INTERFACE WITH THE RIGOL DC 5072
"""

from typing import Union
import visa

class RG():
    def __init__(self, address: str = 'TCPIP0::192.168.0.7::INSTR'):
        rm = visa.ResourceManager()
        self.RG = rm.open_resource(address)
        self.RG.write(':SOUR1:FUNC:SHAP DC') #Change waveform to DC

    def set_voltage(self, voltage: Union[int, float]):
        """
        Sets the Rigol offset voltage
        """
        self.stop()
        self.RG.write(':SOUR1:VOLT:LEV:IMM:OFFS {}'.format(voltage)) #Input DC offset
        self.start()

    def start(self):
        """
        Starts the Rigol
        """
        self.RG.write(':OUTP1 ON ') #Turn Channel 1 on

    def stop(self):
        """
        Stops the Rigol
        """
        self.RG.write(':OUTP1 OFF ') #Turn Channel 1 off
