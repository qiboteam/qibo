import pytest
from qibo import gates
from qibo.models import Circuit


@pytest.mark.parametrize("target", range(5))
@pytest.mark.parametrize("density_matrix", [False, True])
def test_state_representation(backend, target, density_matrix):
    c = Circuit(5, density_matrix=density_matrix)
    c.add(gates.H(target))
    result = backend.execute_circuit(c)
    bstring = target * "0" + "1" + (4 - target) * "0"
    if density_matrix:
        target_str = 3 * [f"(0.5+0j)|00000><00000| + (0.5+0j)|00000><{bstring}| + (0.5+0j)|{bstring}><00000| + (0.5+0j)|{bstring}><{bstring}|"]
    else:
        target_str = [f"(0.70711+0j)|00000> + (0.70711+0j)|{bstring}>",
                      f"(0.7+0j)|00000> + (0.7+0j)|{bstring}>",
                      f"(0.71+0j)|00000> + (0.71+0j)|{bstring}>"]
    assert str(result) == target_str[0]
    assert result.state(decimals=5) == target_str[0]
    assert result.symbolic(decimals=1) == target_str[1]
    assert result.symbolic(decimals=2) == target_str[2]


@pytest.mark.parametrize("density_matrix", [False, True])
def test_state_representation_max_terms(backend, density_matrix):
    from qibo import models, gates
    c = models.Circuit(5, density_matrix=density_matrix)
    c.add(gates.H(i) for i in range(5))
    result = backend.execute_circuit(c)
    if density_matrix:
        assert result.symbolic(max_terms=3) == "(0.03125+0j)|00000><00000| + (0.03125+0j)|00000><00001| + (0.03125+0j)|00000><00010| + ..."
        assert result.symbolic(max_terms=5) == "(0.03125+0j)|00000><00000| + (0.03125+0j)|00000><00001| + (0.03125+0j)|00000><00010| + (0.03125+0j)|00000><00011| + (0.03125+0j)|00000><00100| + ..."
    else:
        assert result.symbolic(max_terms=3) == "(0.17678+0j)|00000> + (0.17678+0j)|00001> + (0.17678+0j)|00010> + ..."
        assert result.symbolic(max_terms=5) == "(0.17678+0j)|00000> + (0.17678+0j)|00001> + (0.17678+0j)|00010> + (0.17678+0j)|00011> + (0.17678+0j)|00100> + ..."
