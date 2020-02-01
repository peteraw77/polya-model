import networkx as nx
from copy import deepcopy
from polya import FiniteNode, network_infection_rate
from datetime import date
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat, savemat
import sys
from network_exposure_fcn import network_exposure

METHOD = sys.argv[1]
PARAMETER = sys.argv[2]
# Either 'closeness', 'degree', 'exposure', 'centrality', or 'all'
TYPE = sys.argv[3]

def build_network(adjacency):
    nodes = [None for x in range(len(adjacency))]

    delta_red = 2

    # Create array of different delta_black for each node first if not being updated at every time step
    if TYPE != 'exposure' and TYPE != 'all':
        delta_black = get_delta_black(nodes, adjacency, delta_red)
    else:
        # Will be updated at each time step during simulation, just initialize array so it works, not used
        delta_black = []
        for i in range(len(adjacency)):
            delta_black.append(delta_red)
            
    # build the node objects
    for i in range(len(adjacency)):
        for j in range(len(adjacency[0])):
            if adjacency[i][j] == 1:
                if not nodes[i]:
                    nodes[i] = FiniteNode([j], delta_red=delta_red, delta_black=delta_black[i])
                else:
                    nodes[i].add_neighbor(j)
    return nodes

def get_delta_black(nodes, adjacency, delta_red):
    graph = nx.from_numpy_matrix(adjacency)
    centralities = nx.closeness_centrality(graph)
    #format degrees list to be how i want
    degrees = []
    for i in range(graph.number_of_nodes()):
        degrees.append(graph.degree[i])

     # Budget for curing is same as total number of delta_red added to network
    budget = delta_red*graph.number_of_nodes()

    delta_black = []
    
    if TYPE == 'exposure':
        prev_net_exp = network_exposure(nodes)
        for i in range(len(nodes)):
            delta_black.append(budget*prev_net_exp[i] / sum(prev_net_exp))

    elif TYPE == 'closeness':
        for i in range(graph.number_of_nodes()):
            delta_black.append(budget*centralities[i] / sum(centralities.values()))
            
    elif TYPE == 'degree':
        for i in range(graph.number_of_nodes()):
            delta_black.append(budget*degrees[i] / sum(degrees))
            
    elif TYPE == 'centrality':
        # Combine closeness and degree
        for i in range(graph.number_of_nodes()):
            # Find sum of products for denominator
            sop = 0
            for j in range(graph.number_of_nodes()):
                sop = sop + centralities[j]*degrees[j]
            delta_black.append(budget*centralities[i]*degrees[i] / sop)
                       
    elif TYPE == 'all':
        for i in range(len(nodes)):
            prev_net_exp = network_exposure(nodes)
            sop = 0
            for j in range(len(nodes)):
                sop = sop + centralities[j]*degrees[j]*prev_net_exp[j]
            delta_black.append(budget*centralities[i]*degrees[i]*prev_net_exp[i] / sop)
            

    return delta_black
                

def simulation(adjacency, runtime):
    nodes = build_network(adjacency)

    # run the simulation
    avg_infection_rate = []
    
    for t in range(runtime):
        
        # Get array of delta_black if update is required at each time step
        if TYPE == 'exposure' or TYPE == 'all':
            delta_black = get_delta_black(nodes, adjacency, nodes[0].delta_red)
                
        # update infection rates
        avg_infection_rate.append(network_infection_rate(nodes))

        # remove values that are out of network's memory
        for node in nodes:
            node.update()

        # draw from urns, update delta_black using network exposure if necessary
        new_nodes = deepcopy(nodes)
        for i in range(len(new_nodes)):
            if TYPE == 'exposure' or TYPE == 'all':
                new_nodes[i].delta_black = delta_black[i]
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

    plt.figure('Heuristic')
    plt.plot(range(runtime), avg_finite_network, 'r-', label='Memory 50')

    plt.legend()
    plt.show()
    
if __name__ == '__main__':
    main()
