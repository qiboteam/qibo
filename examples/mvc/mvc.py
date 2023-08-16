import networkx as nx
from qubo_utils import binary2spin, spin2QiboHamiltonian


def qubo_mvc(g: nx.Graph, penalty=None):
    """Given a graph g, return the QUBO of Minimum Vertex Cover (MVC) problem.

    If penalty is not given, it is inferred from g.

    Args:
        g (nx.Graph): graph
        penalty (float): penalty weight of the constraints

    Returns:
        linear (Dict[Any, float]): linear terms
        quadratic (Dict[(Any, Any), float]): quadratic terms

    """

    q = {(k, k): v for k, v in g.nodes(data="weight")}

    if penalty is None:
        penalty = qubo_mvc_penalty(g)

    for s, d in g.edges:
        q[(s, s)] -= penalty
        q[(d, d)] -= penalty

        if s > d:
            s, d = d, s

        if (s, d) not in q:
            q[(s, d)] = penalty
        else:
            q[(s, d)] += penalty

    linear = {k0: v for (k0, k1), v in q.items() if k0 == k1}
    quadratic = {(k0, k1): v for (k0, k1), v in q.items() if k0 != k1 and v != 0}

    return linear, quadratic


def qubo_mvc_penalty(g: nx.Graph):
    """Find appropriate penalty weight to ensure feasibility

    Args:
        g (nx.Graph): graph

    Returns:
        penalth (float): penalty

    """

    # For NISQ devices, you cannot randomly choose a positive penalty weight, otherwise
    # you get infeasible solutions very likely. If all weight equals to 1,
    # set penalty to the largest degree will ensure high feasibility.

    highest_degree = max(
        sum(g.nodes[k]["weight"] for k in g[i].keys()) for i in g.nodes()
    )
    return highest_degree


def mvc_feasibility(g: nx.Graph, solution):
    """Check if the solution violates the constraints of the problem

    Args:
        g (nx.Graph): graph
        solution (Dict[Any, int]): the solution

    Returns:
        feasibility (bool): whether the solution meet the constraints
    """
    for k0, k1 in g.edges:
        if solution[k0] == solution[k1] == 0:
            return False
    return True


def mvc_energy(g: nx.Graph, solution):
    """Calculate the energy of the solution on a MVC problem

    The result is based on the assumption that soltuion is feasible.

    Args:
        g (nx.Graph): graph
        solution (Dict[Any, int]): the solution

    Returns:
        energy (float): the energy of the solution
    """
    return sum(solution[k] * v for k, v in g.nodes.data("weight"))


def mvc_easy_fix(g: nx.Graph, solution):
    """Naively fix violation in an out-of--place manner

    Args:
        g (nx.Graph): graph
        solution (Dict[Any, int]): the solution

    Returns:
        solution (Dict[Any, int]): fixed solution
    """

    t = {k: v for k, v in solution.items()}

    for k0, k1 in g.edges:
        if t[k0] == t[k1] == 0:
            t[k0] = 1

    return t


def hamiltonian_mvc(g: nx.Graph, penalty: float = None, dense: bool = False):
    """Given a graph g, return the hamiltonian of Minimum Vertex Cover (MVC)
    problem.

    If penalty is not given, it is inferred from g.

    Args:
        g (nx.Graph): graph
        penalty (float): penalty weight of the constraints
        dense (bool): sparse or dense hamiltonian

    Returns:
        ham: qibo hamiltonian

    """

    linear, quadratic = qubo_mvc(g, penalty)

    h, J, _ = binary2spin(linear, quadratic)
    h = {k: -v for k, v in h.items()}
    J = {k: -v for k, v in J.items()}

    ham = spin2QiboHamiltonian(h, J, dense=dense)

    return ham
