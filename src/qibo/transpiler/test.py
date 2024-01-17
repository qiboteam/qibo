from qibo import gates
from qibo.models import Circuit
from qibo.transpiler.placer import StarConnectivityPlacer
from qibo.transpiler.router import StarConnectivityRouter

circ = Circuit(3)
circ.add(gates.H(0))
circ.add(gates.H(1))
circ.add(gates.CZ(0, 1))
circ.add(gates.CZ(1, 2))
circ.add(gates.CZ(0, 2))

placer = StarConnectivityPlacer(middle_qubit=2)
lay = placer(circ)
print(lay)
router = StarConnectivityRouter()
transpiled, final = router(circuit=circ, initial_layout=lay)
print(transpiled.draw())
print(final)
