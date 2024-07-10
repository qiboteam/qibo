import networkx as nx
import numpy as np

def star_connectivity():
    """
    Returns a star graph with 5 nodes and 4 edges.
    """
    Q = [i for i in range(5)]
    chip = nx.Graph()
    chip.add_nodes_from(Q)
    graph_list = [(Q[i], Q[2]) for i in range(5) if i != 2]
    chip.add_edges_from(graph_list)
    return chip

def cycle_connectivity_nx(n):
    """
    Returns a cycle graph with n nodes and n edges.
    Return nx.Graph.
    For Qibo.
    """
    Q = [i for i in range(n)]
    chip = nx.Graph()
    chip.add_nodes_from(Q)
    graph_list = [(Q[i] % n, (Q[i]+1) % n) for i in range(n)]
    # print(graph_list)
    chip.add_edges_from(graph_list)
    return chip

def complete_connectivity_nx(n):
    Q = [i for i in range(n)]
    chip = nx.Graph()
    chip.add_nodes_from(Q)

    elements = np.arange(n)
    graph_list = []
    for i in range(len(elements)):
        for j in range(i + 1, len(elements)):
            graph_list.append((elements[i], elements[j]))
    # print(graph_list)
    chip.add_edges_from(graph_list)
    return chip

def line_connectivity_nx(n):
    """
    Returns a line graph with n nodes and n-1 edges.
    Return nx.Graph.
    For Qibo.
    """
    Q = [i for i in range(n)]
    chip = nx.Graph()
    chip.add_nodes_from(Q)
    graph_list = [(Q[i], (Q[i]+1) % n) for i in range(n-1)]
    # print(graph_list)
    chip.add_edges_from(graph_list)
    return chip

def cycle_connectivity_cmap(n):
    """
    Returns a cycle graph with n nodes and n edges.
    Return list of lists (Coupling map).
    For Qiskit.
    """
    Q = [i for i in range(n)]
    chip = nx.Graph()
    chip.add_nodes_from(Q)
    graph_list = [[Q[i] % n, (Q[i]+1) % n] for i in range(n)]
    graph_list_rev = [[(Q[i]+1) % n, (Q[i]) % n] for i in range(n)]
    graph_list.extend(graph_list_rev)
    # print(graph_list)

    return graph_list

def complete_connectivity_cmap(n):
    Q = [i for i in range(n)]
    chip = nx.Graph()
    chip.add_nodes_from(Q)

    elements = np.arange(n)
    graph_list = []
    for i in range(len(elements)):
        for j in range(i + 1, len(elements)):
            graph_list.append([elements[i], elements[j]])
        
    graph_list_rev = []
    for i in range(len(elements)):
        for j in range(i + 1, len(elements)):
            graph_list_rev.append([elements[j], elements[i]])
    
    graph_list.extend(graph_list_rev)
    # print(graph_list)

    return graph_list

def line_connectivity_cmap(n):
    """
    Returns a line graph with n nodes and n-1 edges.
    Return list of lists (Coupling map).
    For Qiskit.
    """
    Q = [i for i in range(n)]
    chip = nx.Graph()
    chip.add_nodes_from(Q)

    graph_list = [[Q[i], (Q[i]+1) % n] for i in range(n-1)]
    graph_list_rev = [[(Q[i]+1) % n, (Q[i]) % n] for i in range(n-1)]
    graph_list.extend(graph_list_rev)
    # print(graph_list)

    return graph_list

def iqm_connectivity_nx():
    """
    Returns the connectivity graph of the IQM-20 chip architecture (For Qibo).
    """
    g = nx.Graph()
    edges = [
        (1, 2), (1, 4), (3, 4), (2, 5), (4, 5),
        (5, 6), (10, 11), (6, 7), (7, 12), (11, 12),
        (12, 17), (16, 17), (16, 20), (15, 16), (19, 20),
        (11, 16), (15, 19), (18, 19), (14, 18), (14, 15),
        (13, 14), (10, 15), (9, 14), (8, 13), (9, 10), 
        (8, 9), (3, 8), (6, 11), (5, 10), (4, 9)
    ]
    g.add_edges_from(edges)
    # nx.draw(g, with_labels=True)
    return g

def iqm_connectivity_cmap():
    """
    Returns the connectivity coupling graph of the IQM-20 chip architecture (For Qiskit).
    """
    edges = [
        (1, 2), (1, 4), (3, 4), (2, 5), (4, 5),
        (5, 6), (10, 11), (6, 7), (7, 12), (11, 12),
        (12, 17), (16, 17), (16, 20), (15, 16), (19, 20),
        (11, 16), (15, 19), (18, 19), (14, 18), (14, 15),
        (13, 14), (10, 15), (9, 14), (8, 13), (9, 10), 
        (8, 9), (3, 8), (6, 11), (5, 10), (4, 9)
    ]
    graph_list = [[edges[i][0], edges[i][1]] for i in range(len(edges))]
    graph_list_rev = [[edges[i][1], edges[i][0]] for i in range(len(edges))]
    graph_list.extend(graph_list_rev)
    # print(len(graph_list))
    return graph_list
