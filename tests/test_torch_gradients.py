import qibo

qibo.set_backend("pytorch")
import torch

from qibo import gates, models

# Optimization parameters
nepochs = 2
optimizer = torch.optim.Adam
target_state = torch.ones(2, dtype=torch.complex128) / 2.0
# Define circuit ansatz
params = torch.tensor(torch.rand(1, dtype=torch.float64), requires_grad=True)
print(params)
c = models.Circuit(1)
c.add(gates.RX(0, params[0]))
optimizer = optimizer([params])
for _ in range(nepochs):
    optimizer.zero_grad()
    c.set_parameters(params)
    final_state = c().state()
    print("final state:", final_state)
    fidelity = torch.abs(torch.sum(torch.conj(target_state) * final_state))
    loss = 1 - fidelity
    loss.backward()
    print("loss:", loss)
    print("params.grad:", params.grad)
    optimizer.step()
print(params)
