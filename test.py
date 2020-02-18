import numpy as np
from src.qibo import models
from src.qibo import gates
from src.qibo import run


c = models.Circuit(2)

state = np.array([1, 1, 1, 1])
state = state/np.linalg.norm(state)

c.add(gates.Flatten(state))
c.add(gates.H(0))
c.add(gates.H(1))

final_state = run.run(c)

print(final_state)