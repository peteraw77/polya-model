import networkx as nx

def construct_barabasi():
    # gotta learn what the second parameter means
    graph = nx.barabsi_albert_graph(100, 5)

    return graph.edges
