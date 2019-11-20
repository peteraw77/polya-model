import networkx as nx
from copy import deepcopy
from sis import SISNode, network_infection_rate_SIS
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys
import numpy
from numpy import linalg as LA

METHOD = sys.argv[1]
PARAMETER = sys.argv[2]
# Can be "infected", "cured", or "neutral"
RESULT = sys.argv[3]

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


# Calculate initial delta/beta values based on eigenvalue of adj matrix
def initial_params(adj_matrix, result):
    # Find eigenvalues of adj matrix
    w, v = LA.eig(adj_matrix)
    # Convert to magnitudes
    w = [ abs(x) for x in w ]
    # Get maximum magnitude eigenvalue
    max_eig = numpy.amax(w)

    if result == 'infected':
        # This makes delta/beta = max_eig/10
        delta = max_eig / (10 + max_eig)
        beta = 10 / (10 + max_eig)
    elif result == 'cured':
        # delta/beta = 1.01*max_eig
        delta = (1.01 * max_eig) / (1.01*max_eig + 1)
        beta = 1 / (1.01*max_eig +1)
    elif result == 'neutral':
        delta = 0.5
        beta = 0.5
    else:
        raise ValueError("Invalid RESULT flag")

    return (delta, beta)

def build_network(adjacency, result):
    nodes = [None for x in range(len(adjacency))]

    # Get delta, beta values for SIS nodes
    delta, beta = initial_params(adjacency, result)

    # build the node objects
    for i in range(len(adjacency)):
        for j in range(len(adjacency[0])):
            if adjacency[i][j] == 1:
                if not nodes[i]:
                    nodes[i] = SISNode(delta, beta, [j])
                else:
                    nodes[i].add_neighbor(j)
    return nodes

def simulation(adjacency, runtime, result):
    nodes = build_network(adjacency, result)

    avg_infection_rate = []
    for t in range(runtime):
        avg_infection_rate.append(network_infection_rate_SIS(nodes))
        # Update infection probs
        new_nodes = deepcopy(nodes)
        for node in new_nodes:
            node.change_prob_infection(nodes)
        nodes = new_nodes

    return avg_infection_rate

def main():
    trials = 1000
    runtime = 1000
    if METHOD == '-f':
        adj_matrix = load_graph(int(PARAMETER))
    elif METHOD == '-g':
        adj_matrix = construct_barabasi_graph(int(PARAMETER))
    else:
        raise ValueError('Program expects method flag')

    adj_matrix = construct_barabasi_graph(int(PARAMETER))

    avg_network = [0 for x in range(runtime)]

    print('Simulating...')
    for x in tqdm(range(trials)):
        network = simulation(adj_matrix, runtime, RESULT)

        avg_network = [ x + y for x,y in zip(avg_network, network) ]
    avg_network = [ x / trials for x in avg_network ]

    plt.figure('Network')
    plt.plot(range(runtime), avg_network)
    plt.title('Average Infection Rate of Network (SIS)')
    plt.show()

if __name__ == '__main__':
    main()
