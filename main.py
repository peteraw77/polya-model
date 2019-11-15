import networkx as nx
from copy import deepcopy
from polya import FiniteNode, InfiniteNode, network_infection_rate
import matplotlib.pyplot as plt
from scipy.io import loadmat
from tqdm import tqdm
# hello
def construct_barabasi_graph(size):
    # consider using variable size for number of connections
    graph = nx.barabasi_albert_graph(size, 5)
    edges = graph.edges

    # build adjacency matrix
    adjacency = [ [0 for y in range(size)] for x in range(size) ]
    for edge in edges:
        adjacency[edge[0]][edge[1]] = 1
        adjacency[edge[1]][edge[0]] = 1

    return adjacency

def load_graph(filename):
    adjacency = loadmat(filename)['A']

    return adjacency

def build_network(adjacency, NodeType):
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

# size is the number of nodes
def simulation(adjacency, NodeType, runtime):
    nodes = build_network(adjacency, NodeType)

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
    finite_adj_matrix = load_graph('BA_Adj.mat')
    infinite_adj_matrix = load_graph('BA_Adj.mat')
    #finite_adj_matrix = construct_barabasi_graph(10)
    #infinite_adj_matrix = construct_barabasi_graph(10)

    avg_finite_node = []
    avg_finite_network = []
    avg_infinite_node = []
    avg_infinite_network = []

    print('Starting simulation')
    for x in tqdm(range(trials)):
        finite_node,finite_network = simulation(finite_adj_matrix, FiniteNode, runtime)
        infinite_node,infinite_network = simulation(infinite_adj_matrix, InfiniteNode, runtime)

        avg_finite_node.append(finite_node)
        avg_finite_network.append(finite_network)
        avg_infinite_node.append(infinite_node)
        avg_infinite_network.append(infinite_network)
    avg_finite_node = [ x / trials for x in avg_finite_node ]
    avg_finite_network = [ x / trials for x in avg_finite_network ]
    avg_infinite_node = [ x / trials for x in avg_infinite_node ]
    avg_infinite_network = [ x / trials for x in avg_infinite_network ]


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
