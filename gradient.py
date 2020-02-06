import networkx as nx
from copy import deepcopy
from polya import FiniteNode, InfiniteNode, network_infection_rate, gradient_function
from sis_sim import simulation as sis_simulation
import matplotlib.pyplot as plt
from scipy.io import loadmat, savemat
import numpy as np
from datetime import date
import numpy as np
from tqdm import tqdm
import sys
from sympy import derive_by_array
from sympy.abc import x,y

METHOD = sys.argv[1]
PARAMETER = sys.argv[2]
RESULT = sys.argv[3]

# hello
def construct_barabasi_graph(size):
    # consider using variable size for number of connections
    #graph = nx.barabasi_albert_graph(size, 5)
    graph = nx.barabasi_albert_graph(size, 1)
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

def build_network(adjacency, NodeType, result):
    nodes = [None for x in range(len(adjacency))]

    delta_red = 2
    if result == 'neutral':
        delta_black = 2
    else:
        # calculate eigenvalues
        w,_ = np.linalg.eig(adjacency)
        w = [abs(x) for x in w]
        max_eig = np.amax(w)

        if result == 'cured':
            delta_black = 1.01 * max_eig * delta_red
        elif result == 'infected':
            delta_black = max_eig / 10 * delta_red
        else:
            raise ValueError("Network type must be 'cured', 'infected', or 'neutral'")

    # build the node objects
    for i in range(len(adjacency)):
        for j in range(len(adjacency[0])):
            if adjacency[i][j] == 1:
                if not nodes[i]:
                    nodes[i] = NodeType([j], delta_red=delta_red, delta_black=delta_black)
                else:
                    nodes[i].add_neighbor(j)
    return nodes

def distribute_curing(nodes, last_nodes, budget):
    # get component derivatives of f
    partials = derive_by_array(gradient_function(nodes, [x, y], last_nodes), [x, y])

    # identify the steepest descent
    # substitute the current values of x1,x2
    descents = [float(partial.subs([(x, nodes[0].delta_black), (y, nodes[1].delta_black)])) \
            for partial in partials]

    # find the deepest descent
    index = descents.index(sorted(descents)[0])

    # construct curing distribution
    distribution = [0 for x in range(len(nodes))]
    distribution[index] = budget

    # limit minimization
    alphas = []
    for i in range(1000):
        alphas.append(gradient_function(nodes, [nodes[0].delta_black + i * \
            (distribution[0] - nodes[0].delta_black), nodes[1].delta_black + \
            i * 0.001 * (distribution[1] - nodes[1].delta_black)], last_nodes))

    alpha_k = sorted(alphas)[0]

    curing = [int(nodes[0].delta_black + alpha_k * \
            (distribution[0] - nodes[0].delta_black)), int(nodes[1].delta_black + \
            alpha_k * (distribution[1] - nodes[1].delta_black))]

    # add remaining budget lost to rounding to the most influential node
    curing[index] += budget - sum(curing)

    return curing


# size is the number of nodes
def simulation(adjacency, NodeType, runtime, result):
    nodes = build_network(adjacency, NodeType, result)

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
        curing_distribution = distribute_curing(new_nodes, nodes, 4)
        nodes = new_nodes
        for i in range(len(curing_distribution)):
            nodes[i].delta_black = curing_distribution[i]

    return avg_infection_rate

def main():
    trials = 500
    runtime = 1000
    if METHOD == '-f':
        adj_matrix = load_graph(PARAMETER)
    elif METHOD == '-g':
        adj_matrix = construct_barabasi_graph(int(PARAMETER))
    else:
        raise ValueError('Program expects method flag')

    avg_finite_network = [0 for x in range(runtime)]
   # avg_infinite_network = [0 for x in range(runtime)]
   # avg_sis_network = [0 for x in range(runtime)]

    print('Simulating...')
    for x in tqdm(range(trials)):
        finite_network = simulation(adj_matrix, FiniteNode, runtime, RESULT)
   #     infinite_network = simulation(adj_matrix, InfiniteNode, runtime, RESULT)
   #     sis_network = sis_simulation(adj_matrix, runtime, RESULT)

        avg_finite_network = [ x + y for x,y in zip(avg_finite_network,finite_network) ]
   #     avg_infinite_network = [ x + y for x,y in zip(avg_infinite_network,infinite_network) ]
   #     avg_sis_network = [ x + y for x,y in zip(avg_sis_network,sis_network) ]
    avg_finite_network = [ x / trials for x in avg_finite_network ]
   # avg_infinite_network = [ x / trials for x in avg_infinite_network ]
   # avg_sis_network = [ x / trials for x in avg_sis_network ]

    plt.figure('SuperimposedInfectionRates')
    plt.plot(range(runtime), avg_finite_network, 'r-', label='Memory 12')
   # plt.plot(range(runtime), avg_infinite_network, 'b-', label='Infinite Memory')
   # plt.plot(range(runtime), avg_sis_network, 'k-', label='SIS')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()
