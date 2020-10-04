import sympy
from qibo import matrices, hamiltonians

nqubits = 5

x_symbols = sympy.symbols(" ".join((f"X{i}" for i in range(nqubits))))
z_symbols = sympy.symbols(" ".join((f"Z{i}" for i in range(nqubits))))
symmap = {x: (i, matrices.X) for i, x in enumerate(x_symbols)}
symmap.update({x: (i, matrices.Z) for i, x in enumerate(z_symbols)})


symham = sum(z_symbols[i] * z_symbols[i + 1] for i in range(4))
symham += z_symbols[0] * z_symbols[-1]
symham += sum(x_symbols)

ham = hamiltonians.TrotterHamiltonian.from_symbolic(symham, symmap)

for part in ham.parts:
    for k, v in part.items():
        print(k, v)