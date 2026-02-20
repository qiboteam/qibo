# %%
import mpmath
from pygridsynth.gridsynth import gridsynth_gates

from qibo import Circuit, gates

# %%
mpmath.mp.dps = 256
epsilon = 1e-3
epsilon = mpmath.mpmathify(epsilon)

param = 0.01
gate = gates.RY(0, param)

theta = mpmath.mpmathify(gate.parameters[0])
sequence = gridsynth_gates(theta=theta, epsilon=epsilon)

circuit = Circuit(1)
circuit.add(getattr(gates, gate)(0) for gate in reversed(sequence) if gate != "W")
circuit.draw()

# %%
