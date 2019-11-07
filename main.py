import networkx as nx
from copy import deepcopy
from network import FiniteNode

def construct_barabasi(size):
    # gotta learn what the second parameter means
    graph = nx.barabasi_albert_graph(size, 5)

    return graph.edges

# size is the number of nodes
def simulation(size, runtime):
    edges = construct_barabasi(size)
    nodes = [None for x in range(size)]

    # build the node objects
    for edge in edges:
        # add connection to first node
        if not nodes[edge[0]]:
            nodes[edge[0]] = FiniteNode([edge[1]])
        else:
            nodes[edge[0]].add_neighbor(edge[1])

        # second node
        if not nodes[edge[1]]:
            nodes[edge[1]] = FiniteNode([edge[0]])
        else:
            nodes[edge[1]].add_neighbor(edge[0])

    # run the simulation
    for t in range(runtime):
        # remove values that are out of network's memory
        for node in nodes:
            node.update()

        # draw from urns
        new_nodes = deepcopy(nodes)
        for node in new_nodes:
            node.draw(nodes)
        # don't update any nodes until all draws have been made
        nodes = new_nodes

    # show off results somehow
    print(nodes)

if __name__ == '__main__':
    simulation(10, 1000)
