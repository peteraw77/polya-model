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

def load_graph(filename, NodeType=FiniteNode):
    adjacency = loadmat(filename)['A']
    nodes = [None for x in range(len(adjacency))]

    # build the node objects
    for i in range(len(adjacency)):
        for j in range(len(adjacency[0])):
            if adjacency[i][j] == 1:
                if not nodes[i]:
                    nodes[i] = NodeType([j])
                else:
                    nodes[i].add_neighbor(j)
    return nodes

def generate_graph(NodeType, size):
    edges = construct_barabasi(size)
    nodes = [None for x in range(size)]

    # build the node objects
    for edge in edges:
        # add connection to first node
        if not nodes[edge[0]]:
            nodes[edge[0]] = NodeType([edge[1]])
        else:
            nodes[edge[0]].add_neighbor(edge[1])

        # second node
        if not nodes[edge[1]]:
            nodes[edge[1]] = NodeType([edge[0]])
        else:
            nodes[edge[1]].add_neighbor(edge[0])
    return nodes

# size is the number of nodes
def simulation(graph, runtime):
    # run the simulation
    infection_node_zero = []
    avg_infection_rate = []
    for t in range(runtime):
        # update infection rates
        node_zero_red, node_zero_black = graph[0].construct_super_urn(graph)
        infection_node_zero.append(node_zero_red / (node_zero_red + node_zero_black))
        avg_infection_rate.append(network_infection_rate(graph))

        # remove values that are out of network's memory
        for node in graph:
            node.update()

        # draw from urns
        new_graph = deepcopy(graph)
        for node in new_graph:
            node.draw(graph)
        # don't update any nodes until all draws have been made
        graph = new_graph

    return (infection_node_zero, avg_infection_rate)

if __name__ == '__main__':
    trials = 5000
    runtime = 1000
    finite_graph = load_graph('BA_Adj.mat', FiniteNode)
    infinite_graph = load_graph('BA_Adj.mat', InfiniteNode)
    #finite_graph = generate_graph(10, FiniteNode)
    #infinite_graph = generate_graph(10, IniniteNode)

    avg_finite_node = []
    avg_finite_network = []
    avg_infinite_node = []
    avg_infinite_network = []

    for x in range(trials):
        print('Trial number ' + str(x))
        finite_node,finite_network = simulation(finite_graph, runtime)
        infinite_node,infinite_network = simulation(infinite_graph, runtime)

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
