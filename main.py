import networkx as nx

def construct_barabasi(size):
    # gotta learn what the second parameter means
    graph = nx.barabsi_albert_graph(size, 5)

    return graph.edges

# size is the number of nodes
def simulation(size):
    edges = construct_barabasi(size)
    nodes = [None for x in range(size)]

    # build the node objects
    for edge in edges:
        # add connection to first node
        node = nodes[edge[0]]
        if node == None:
            node = MemorylessNode([edge[1]])
        else:
            node.neighborhood = node.neighborhood.append(edge[1])

        # second node
        node = nodes[edge[1]]
        if node == None:
            node = MemorylessNode([edge[0]])
        else:
            node.neighborhood = node.neighborhood.append(edge[0])

        # run the simulation
