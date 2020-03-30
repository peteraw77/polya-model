import networkx as nx
from copy import deepcopy
from polya import FiniteNode, InfiniteNode, network_infection_rate
import matplotlib.pyplot as plt
from scipy.io import loadmat, savemat
from datetime import date
import numpy as np
from tqdm import tqdm
import sys

METHOD = sys.argv[1]
PARAMETER = sys.argv[2]

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

    # save matrix
    savemat('Barabasi_' + str(size) + '_' + str(date.today()) + '.mat', {'A': adjacency})

    return adjacency

def load_graph(filename):
    adjacency = loadmat(filename)['A']

    return adjacency


def build_network(adjacency):
    nodes = [None for x in range(len(adjacency))]

    graph = nx.from_numpy_matrix(adjacency)
    centr_dict = nx.closeness_centrality(graph)
    centralities = []
    for key in centr_dict:
        centralities.append(-centr_dict[key])
    # Get array of sorted indices to assign curing resources to most central nodes in graph
    sort_index = np.argsort(centralities)
    delta_red = 2
    budget = delta_red*graph.number_of_nodes()
    delta_black = [0]*graph.number_of_nodes()
    
     # calculate eigenvalues
    w,_ = np.linalg.eig(adjacency)
    w = [abs(x) for x in w]
    max_eig = np.amax(w)

    if (budget -  (1.01*max_eig*delta_red) < 0):
        raise ValueError('Black balls required for first node greater than budget.')

    else:
        j = 0
        while (budget -  (1.01*max_eig*delta_red) > 0):        
            delta_black[sort_index[j]] = 1.01*max_eig*delta_red
            budget = budget - (1.01*max_eig*delta_red)
            j = j + 1

        # Allocate any remaining budget
        if (budget > 0):
            delta_black[sort_index[j]] = budget
    
    # build the node objects
    for i in range(len(adjacency)):
        for j in range(len(adjacency[0])):
            if adjacency[i][j] == 1:
                if not nodes[i]:
                    nodes[i] = NodeType([j], delta_red=delta_red, delta_black=delta_black[i])
                else:
                    nodes[i].add_neighbor(j)
    return nodes

# size is the number of nodes
def simulation(adjacency, runtime):
    nodes = build_network(adjacency)

    # run the simulation
    avg_infection_rate = []
    for t in range(runtime):
        # update infection rates
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

    return avg_infection_rate

def main():
    trials = 100
    runtime = 1000
    if METHOD == '-f':
        adj_matrix = load_graph(PARAMETER)
    elif METHOD == '-g':
        adj_matrix = construct_barabasi_graph(int(PARAMETER))
    else:
        raise ValueError('Program expects method flag')

    avg_finite_network = [0 for x in range(runtime)]

    print('Simulating...')
    for x in tqdm(range(trials)):
        finite_network = simulation(adj_matrix, runtime)

        avg_finite_network = [ x + y for x,y in zip(avg_finite_network,finite_network) ]

    avg_finite_network = [ x / trials for x in avg_finite_network ]

    plt.figure('SIS Strategy')
    plt.plot(range(runtime), avg_finite_network, 'r-', label='Memory 150')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()
