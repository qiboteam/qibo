"""
UTILITY CLASS FOR INTERFACING WITH THE MINI-CIRCUITS ATTENUATORS
"""
import urllib3

class MCAttenuator():
    def __init__(self, address: str):
        self.address = address

    def set_attenuation(self, attenuation: int) -> None:
        http = urllib3.PoolManager()
        http.request('GET', 'http://{}/SETATT={}'.format(self.address, attenuation))
