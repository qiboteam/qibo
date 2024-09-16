import qibo

qibo.set_backend("pytorch")
import torch

from qibo import gates, models

torch.set_anomaly_enabled(True)

# Optimization parameters
nepochs = 1
optimizer = torch.optim.Adam
target_state = torch.ones(4, dtype=torch.complex128) / 2.0

# Define circuit ansatz
params = torch.rand(2, dtype=torch.float64, requires_grad=True)
print(params)
optimizer = optimizer([params])
c = models.Circuit(2)
c.add(gates.RX(0, params[0]))
c.add(gates.RY(1, params[1]))
gate = gates.RY(0, params[1])

print("Gate", gate.matrix())
print(torch.norm(gate.matrix()).grad)

# for _ in range(nepochs):
#     optimizer.zero_grad()
#     c.set_parameters(params)
#     final_state = c().state()
#     print("state", final_state)
#     fidelity = torch.abs(torch.sum(torch.conj(target_state) * final_state))
#     loss = 1 - fidelity
#     loss.backward()
#     optimizer.step()
#     print("state", final_state)
#     print("params", params)
#     print("loss", loss.grad)
