import networkx as nx
from copy import deepcopy
from polya import FiniteNode, InfiniteNode, network_infection_rate
from sis_sim import simulation as sis_simulation
import matplotlib.pyplot as plt
from scipy.io import loadmat, savemat
from datetime import date
import numpy as np
from tqdm import tqdm
import sys

METHOD = sys.argv[1]
PARAMETER = sys.argv[2]
RESULT = sys.argv[3]

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
        nodes = new_nodes

    return avg_infection_rate

def gradientfunction(nodes, curing, last_nodes):
    f = 0
    for i in range(len(nodes)):
        red_sum = sum(nodes[i].additional_red)
        black_sum = sum(nodes[i].additional_black)
        for neighbor in nodes[i].neighborhood:
            red_sum = red_sum + sum(neighbor.additional_red)
            black_sum = black_sum + sum(neighbor.additional_black)
            
        total_red, total_black = nodes[i].construct_super_urn(nodes)
        total_red_prev, total_black_prev = last_nodes[i].construct_super_urn(last_nodes)
        c = total_red + node.delta_red * (total_red_prev / (total_black_prev + total_red_prev) ) + red_sum
        d = c + total_black + black_sum

        sigma = curing[i]*(1 - (total_red_prev / (total_black_prev + total_red_prev) ))

        for j in range(len(last_nodes[i].neighborhood)):
            n_red, n_black = last_nodes[i].neighborhood[j].construct_super_urn(last_nodes)
            neighbor_exp = (n_red / (n_red + n_black))

            sigma = sigma + curing[j]*(1-neighbor_exp)

        f = f + c / (d + sigma)
    f = f / len(nodes)

    return f
        
    

def main():
    trials = 5000
    runtime = 1000
    if METHOD == '-f':
        adj_matrix = load_graph(PARAMETER)
    elif METHOD == '-g':
        adj_matrix = construct_barabasi_graph(int(PARAMETER))
    else:
        raise ValueError('Program expects method flag')

    avg_finite_network = [0 for x in range(runtime)]
    avg_infinite_network = [0 for x in range(runtime)]
    avg_sis_network = [0 for x in range(runtime)]

    print('Simulating...')
    for x in tqdm(range(trials)):
        finite_network = simulation(adj_matrix, FiniteNode, runtime, RESULT)
        infinite_network = simulation(adj_matrix, InfiniteNode, runtime, RESULT)
        sis_network = sis_simulation(adj_matrix, runtime, RESULT)

        avg_finite_network = [ x + y for x,y in zip(avg_finite_network,finite_network) ]
        avg_infinite_network = [ x + y for x,y in zip(avg_infinite_network,infinite_network) ]
        avg_sis_network = [ x + y for x,y in zip(avg_sis_network,sis_network) ]
    avg_finite_network = [ x / trials for x in avg_finite_network ]
    avg_infinite_network = [ x / trials for x in avg_infinite_network ]
    avg_sis_network = [ x / trials for x in avg_sis_network ]

    plt.figure('SuperimposedInfectionRates')
    plt.plot(range(runtime), avg_finite_network, 'r-', label='Memory 50')
    plt.plot(range(runtime), avg_infinite_network, 'b-', label='Infinite Memory')
    plt.plot(range(runtime), avg_sis_network, 'k-', label='SIS')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()
