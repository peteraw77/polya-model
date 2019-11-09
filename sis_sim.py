import networkx as nx
from copy import deepcopy
from sis import SISNode, network_infection_rate_SIS
import matplotlib.pyplot as plt

def construct_barabasi(size):
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
            nodes[edge[0]] = SISNode([edge[1]])
        else:
            nodes[edge[0]].add_neighbor(edge[1])

        # second node
        if not nodes[edge[1]]:
            nodes[edge[1]] = SISNode([edge[0]])
        else:
            nodes[edge[1]].add_neighbor(edge[0])

    # run the simulation
    infection_node_zero = []
    avg_infection_rate = []
    for t in range(runtime):
        avg_infection_rate.append(network_infection_rate_SIS(nodes))
        # Update infection probs
        new_nodes = deepcopy(nodes)
        for node in new_nodes:
            node.change_prob_infection(nodes)
        nodes = new_nodes

    plt.figure('Network')
    plt.plot(range(runtime), avg_infection_rate)
    plt.title('Average Infection Rate of Network (SIS)')
    plt.show()

if __name__ == '__main__':
    simulation(10, 1000)
