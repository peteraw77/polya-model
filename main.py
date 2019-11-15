import networkx as nx
from copy import deepcopy
from polya import FiniteNode, InfiniteNode, network_infection_rate
import matplotlib.pyplot as plt
from scipy.io import loadmat
# hello
def construct_barabasi(size):
    # gotta learn what the second parameter means
    graph = nx.barabasi_albert_graph(size, 5)

    return graph.edges

# size is the number of nodes
def simulation(NodeType, size, runtime):
    #edges = construct_barabasi(size)
    adjacency = loadmat('BA_Adj.mat')['A']
    nodes = [None for x in range(len(adjacency))]

    # build the node objects
    for i in range(len(adjacency)):
        for j in range(len(adjacency[0])):
            if adjacency[i][j] == 1:
                if not nodes[i]:
                    nodes[i] = NodeType([j])
                else:
                    nodes[i].add_neighbor(j)

    # run the simulation
    infection_node_zero = []
    avg_infection_rate = []
    for t in range(runtime):
        # update infection rates
        node_zero_red, node_zero_black = nodes[0].construct_super_urn(nodes)
        infection_node_zero.append(node_zero_red / (node_zero_red + node_zero_black))
        avg_infection_rate.append(network_infection_rate(nodes))

        # remove values that are out of network's memory
        for node in nodes:
            node.update()

        # draw from urns
        new_nodes = deepcopy(nodes)
        for node in new_nodes:
            node.draw(nodes)
        # don't update any nodes until all draws have been made
        nodes = new_nodes

    return (infection_node_zero, avg_infection_rate)

if __name__ == '__main__':
    trials = 5000
    runtime = 1000

    avg_finite_node = []
    avg_finite_network = []
    avg_infinite_node = []
    avg_infinite_network = []

    for x in range(trials):
        print('Trial number ' + str(x))
        finite_node,finite_network = simulation(FiniteNode, 10, 1000)
        infinite_node,infinite_network = simulation(InfiniteNode, 10, 1000)

        avg_finite_node.append(finite_node)
        avg_finite_network.append(finite_network)
        avg_infinite_node.append(infinite_node)
        avg_infinite_network.append(infinite_network)
    avg_finite_node = avg_finite_node / trials
    avg_finite_network = avg_finite_network / trials
    avg_infinite_node = avg_infinite_node / trials
    avg_infinite_network = avg_infinite_network / trials


    plt.figure('FiniteNode')
    plt.plot(range(runtime), avg_finite_node)
    plt.title('Infection rate for Node 0 [FINITE]')

    plt.figure('InfiniteNode')
    plt.plot(range(runtime), avg_infinite_node)
    plt.title('Infection rate for Node 0 [INFINITE]')

    plt.figure('FiniteNetwork')
    plt.plot(range(runtime), avg_finite_network)
    plt.title('Average Infection Rate of Network [FINITE]')

    plt.figure('InfiniteNetwork')
    plt.plot(range(runtime), avg_infinite_network)
    plt.title('Average Infection Rate of Network [INFINITE]')
    plt.show()
