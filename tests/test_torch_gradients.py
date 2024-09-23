import pytest
import torch

import qibo
from qibo import gates, models


def tetst_torch_gradients():
    qibo.set_backend("pytorch")
    torch.seed(42)
    nepochs = 1001
    optimizer = torch.optim.Adam
    target_state = torch.rand(2, dtype=torch.complex128)
    target_state = target_state / torch.norm(target_state)
    params = torch.rand(2, dtype=torch.float64, requires_grad=True)
    c = models.Circuit(1)
    c.add(gates.RX(0, params[0]))
    c.add(gates.RY(0, params[1]))

    initial_params = params.clone()
    initial_loss = 1 - torch.abs(torch.sum(torch.conj(target_state) * c().state()))

    optimizer = optimizer([params])
    for _ in range(nepochs):
        optimizer.zero_grad()
        c.set_parameters(params)
        final_state = c().state()
        fidelity = torch.abs(torch.sum(torch.conj(target_state) * final_state))
        loss = 1 - fidelity
        loss.backward()
        optimizer.step()

    assert initial_loss > loss
    assert initial_params[0] != params[0]
