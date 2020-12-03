import json
import numpy as np
#from qibo.config import DTYPES


def rho_theory(i):
    rho = np.zeros((4, 4), dtype=complex)
    rho[i, i] = 1
    return rho


def extract(filename):
    with open(filename,"r") as r:
        r = r.read()
        raw = json.loads(r)
    return raw


statefile = "data/states_181120.json"

measurementfiles = [("data/tomo_181120-00.json", rho_theory(0)),
                    ("data/tomo_181120-01.json", rho_theory(1)),
                    ("data/tomo_181120-10.json", rho_theory(2)),
                    ("data/tomo_181120-11.json", rho_theory(3))]
