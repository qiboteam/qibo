import json
from qibo import models, gates


def circuit(gates):
    c = models.Circuit(2, density_matrix=True)
    if gates:
        c.add(gates)
    return c


def extract(filename):
    with open(filename,"r") as r:
        r = r.read()
        raw = json.loads(r)
    return raw


statefile = "data/states_181120.json"

measurementfiles = [("data/tomo_181120-00.json", circuit([])),
                    ("data/tomo_181120-01.json", circuit([gates.X(1)])),
                    ("data/tomo_181120-10.json", circuit([gates.X(0)])),
                    ("data/tomo_181120-11.json", circuit([gates.X(0), gates.X(1)]))]
