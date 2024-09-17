import qibo

qibo.set_backend("pytorch")
import torch

from qibo import gates, models

# Optimization parameters
nepochs = 2
optimizer = torch.optim.Adam
target_state = torch.ones(2, dtype=torch.complex128) / 2.0
params = torch.rand(1, dtype=torch.float64, requires_grad=True)
print("Initial params", params)
c = models.Circuit(1)
gate = gates.RX(0, params)
c.add(gate)

optimizer = optimizer([params])
for _ in range(nepochs):
    optimizer.zero_grad()
    c.set_parameters(params)
    final_state = c().state()
    fidelity = torch.abs(torch.sum(torch.conj(target_state) * final_state))
    loss = 1 - fidelity
    loss.backward()
    print("loss:", loss)
    print("params.grad:", params.grad)
    optimizer.step()
print("Final parameters:", params)
