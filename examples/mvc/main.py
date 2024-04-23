"""Minimum Vertex Cover"""

import argparse
import csv

import networkx as nx
from mvc import (
    hamiltonian_mvc,
    mvc_easy_fix,
    mvc_energy,
    mvc_feasibility,
    qubo_mvc,
    qubo_mvc_penalty,
)
from qubo_utils import binary2spin, spin2QiboHamiltonian

from qibo import callbacks, hamiltonians, models

parser = argparse.ArgumentParser()
parser.add_argument("--filename", default="./mvc.csv", type=str)


def load_csv(filename: str):
    """Load graph from csv file"""
    with open(filename, newline="") as f:
        reader = csv.reader(f)
        data = [[int(row[0]), int(row[1]), float(row[2])] for row in reader]

    nodes = [k0 for k0, k1, v in data if k0 == k1]
    edges = [[k0, k1] for k0, k1, v in data if k0 != k1]

    g = nx.Graph()
    g.add_nodes_from(nodes)
    g.add_edges_from(edges)

    node_weights = {k0: {"weight": v} for k0, k1, v in data if k0 == k1}
    edge_weights = {(k0, k1): {"weight": v} for k0, k1, v in data if k0 != k1}

    nx.set_edge_attributes(g, edge_weights)
    nx.set_node_attributes(g, node_weights)

    return g


def main(filename: str = "./mvc.csv"):
    print(f"Load a graph from {filename} and make it a QUBO")
    g = load_csv(filename)
    penalty = qubo_mvc_penalty(g)
    linear, quadratic = qubo_mvc(g, penalty=penalty)

    print("A random solution with seed 1234 must be infeasible")
    import numpy as np

    rng = np.random.default_rng(seed=1234)
    random_solution = {i: rng.integers(2) for i in g.nodes}
    feasibility = mvc_feasibility(g, random_solution)
    assert not feasibility, "The random solution should be infeasible."

    print("Make a naive fix to the violation of the constraint")
    fixed_solution = mvc_easy_fix(g, random_solution)
    feasibility = mvc_feasibility(g, fixed_solution)
    assert feasibility, "The fixed solution should be feasible."

    print("Calculate the energy of the solution")
    energy = mvc_energy(g, fixed_solution)

    print("Construct a hamiltonian based on the QUBO representation")
    h, J, _ = binary2spin(linear, quadratic)
    h = {k: -v for k, v in h.items()}
    J = {k: -v for k, v in J.items()}
    ham = spin2QiboHamiltonian(h, J)

    print("Construct a hamiltonian directly from a networkx graph")
    ham = hamiltonian_mvc(g)

    print("done.")


if __name__ == "__main__":
    # by defualt, test on the mvc.csv in the same directory
    args = parser.parse_args()
    main(args.filename)
