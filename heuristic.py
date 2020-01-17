import networkx as nx
from copy import deepcopy
from polya import FiniteNode, network_infection_rate
from datetime import date
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat, savemat
import sys

METHOD = sys.argv[1]
PARAMETER = sys.argv[2]
# Either 'closeness', 'degree', 'exposure', or 'combo'
TYPE = sys.argv[3]

def build_network(adjacency):
    nodes = [None for x in range(len(adjacency))]
    graph = nx.from_numpy_matrix(adjacency)
    centralities = nx.closeness_centrality(graph)
    #format degrees list to be how i want
    degrees = []
    for i in range(graph.number_of_nodes()):
        degrees.append(graph.degree[i])

    delta_red = 2
    budget = delta_red*len(adjacency)

    # Create array of different delta_black for each node
    delta_black = []
    for i in range(len(adjacency)):
        if TYPE == 'closeness':
            delta_black.append(budget*centralities[i] / sum(centralities.values()))
        elif TYPE == 'degree':
            delta_black.append(budget*degrees[i] / sum(degrees.values()))

        elif TYPE == 'exposure':
        # updates at every step during simulation, just initialize to initial proportion in superurns
            delta_black.append(budget*0.5)
        elif TYPE == 'combo':
            # Combine closeness and degree
            # Find sum of products for denominator
            product = 0
            for j in range(len(nodes)):
                for k in range(j+1,len(nodes)):
                    product = product + centralities[j]*degrees[k]
            delta_black.append(budget*centralities[i]*degrees[i] / product)
            

    # build the node objects
    for i in range(len(adjacency)):
        for j in range(len(adjacency[0])):
            if adjacency[i][j] == 1:
                if not nodes[i]:
                    nodes[i] = FiniteNode([j], delta_red=delta_red, delta_black=delta_black[i])
                else:
                    nodes[i].add_neighbor(j)
    return nodes
    

def simulation(adjacency, runtime):
    nodes = build_network(adjacency)
    # Budget for curing is same as total number of delta_red added to network
    budget = nodes[0].delta_red*len(nodes)

    # run the simulation
    avg_infection_rate = []
    for t in range(runtime):
        
        # Create array of network exposure rates for next time step (if necessary)
        if TYPE == 'exposure':
            prev_net_exp = []
            for node in nodes:
                total_red, total_black = node.construct_super_urn(nodes)
                prev_net_exp.append(total_red / (total_black + total_black))
        # update infection rates
        avg_infection_rate.append(network_infection_rate(nodes))

        # remove values that are out of network's memory
        for node in nodes:
            node.update()

        # draw from urns, update delta_black using network exposure if necessary
        new_nodes = deepcopy(nodes)
        for i in range(len(new_nodes)):
            if TYPE == 'exposure':
                new_nodes[i].delta_black = (budget*prev_net_exp[i])/sum(prev_net_exp)
            new_nodes[i].draw(nodes)
        # don't update any nodes until all draws have been made
        nodes = new_nodes

    return avg_infection_rate

def main():
    trials = 100
    runtime = 1000
    if METHOD == '-f':
        adj_matrix = loadmat(PARAMETER)['A']
    else:
        raise ValueError('Program expects method flag')
    
    avg_finite_network = [0 for x in range(runtime)]

    print('Simulating...')
    for x in tqdm(range(trials)):
        finite_network = simulation(adj_matrix, runtime)

        avg_finite_network = [ x + y for x,y in zip(avg_finite_network,finite_network) ]
                          
    avg_finite_network = [ x / trials for x in avg_finite_network ]

    plt.figure('Centrality Heuristic')
    plt.plot(range(runtime), avg_finite_network, 'r-', label='Memory 50')

    plt.legend()
    plt.show()
    
if __name__ == '__main__':
    main()
