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
#TYPE = sys.argv[3]

def build_network(adjacency, strategy):
    nodes = [None for x in range(len(adjacency))]

    delta_red = 20

    # Create array of different delta_black for each node first if not being updated at every time step
    if strategy != 'exposure' and strategy != 'all':
        delta_black = get_delta_black(nodes, adjacency, delta_red, strategy)
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

def get_delta_black(nodes, adjacency, delta_red, strategy):
    graph = nx.from_numpy_matrix(adjacency)
    centralities = nx.closeness_centrality(graph)
    degrees = []
    for i in range(graph.number_of_nodes()):
        degrees.append(graph.degree[i])

     # Budget for curing is same as total number of delta_red added to network
    budget = delta_red*graph.number_of_nodes()

    delta_black = []
    
    if strategy == 'exposure':
        prev_net_exp = network_exposure(nodes)
        for i in range(len(nodes)):
            delta_black.append(budget*prev_net_exp[i] / sum(prev_net_exp))

    elif strategy == 'closeness':
        for i in range(graph.number_of_nodes()):
            delta_black.append(budget*centralities[i] / sum(centralities.values()))
            
    elif strategy == 'degree':
        for i in range(graph.number_of_nodes()):
            delta_black.append(budget*degrees[i] / sum(degrees))
            
    elif strategy == 'centrality':
        # Combine closeness and degree
        for i in range(graph.number_of_nodes()):
            # Find sum of products for denominator
            sop = 0
            for j in range(graph.number_of_nodes()):
                sop = sop + centralities[j]*degrees[j]
            delta_black.append(budget*centralities[i]*degrees[i] / sop)
                       
    elif strategy == 'all':
        for i in range(len(nodes)):
            prev_net_exp = network_exposure(nodes)
            sop = 0
            for j in range(len(nodes)):
                sop = sop + centralities[j]*degrees[j]*prev_net_exp[j]
            delta_black.append(budget*centralities[i]*degrees[i]*prev_net_exp[i] / sop)
            

    return delta_black
                

def simulation(adjacency, runtime, strategy):
    nodes = build_network(adjacency, strategy)

    # run the simulation
    avg_infection_rate = []
    
    for t in range(runtime):
        
        # Get array of delta_black if update is required at each time step
        if strategy == 'exposure' or strategy == 'all':
            delta_black = get_delta_black(nodes, adjacency, nodes[0].delta_red, strategy)
                
        # update infection rates
        avg_infection_rate.append(network_infection_rate(nodes))

        # remove values that are out of network's memory
        for node in nodes:
            node.update()

        # draw from urns, update delta_black using network exposure if necessary
        new_nodes = deepcopy(nodes)
        for i in range(len(new_nodes)):
            if strategy == 'exposure' or strategy == 'all':
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
    
    avg_degree_network = [0 for x in range(runtime)]
    avg_closeness_network = [0 for x in range(runtime)]
    avg_centrality_network = [0 for x in range(runtime)]
    avg_exposure_network = [0 for x in range(runtime)]
    avg_all_network = [0 for x in range(runtime)]

    print('Simulating...')
    for x in tqdm(range(trials)):
        degree_network = simulation(adj_matrix, runtime, 'degree')
        closeness_network = simulation(adj_matrix, runtime, 'closeness')
        centrality_network = simulation(adj_matrix, runtime, 'centrality')
        exposure_network = simulation(adj_matrix, runtime, 'exposure')
        all_network = simulation(adj_matrix, runtime, 'all')

        avg_degree_network = [ x + y for x,y in zip(avg_degree_network,degree_network) ]
        avg_closeness_network = [ x + y for x,y in zip(avg_closeness_network,closeness_network) ]
        avg_centrality_network = [ x + y for x,y in zip(avg_centrality_network,centrality_network) ]
        avg_exposure_network = [ x + y for x,y in zip(avg_exposure_network,exposure_network) ]
        avg_all_network = [ x + y for x,y in zip(avg_all_network,all_network) ]
                          
    avg_degree_network = [ x / trials for x in avg_degree_network ]
    avg_closeness_network = [ x / trials for x in avg_closeness_network ]
    avg_centrality_network = [ x / trials for x in avg_centrality_network ]
    avg_exposure_network = [ x / trials for x in avg_exposure_network ]
    avg_all_network = [ x / trials for x in avg_all_network ]

    plt.figure('Heuristics')
    plt.plot(range(runtime), avg_degree_network, 'r-', label='Degree, Memory 150')
    plt.plot(range(runtime), avg_closeness_network, 'b-', label='Closeness, Memory 150')
    plt.plot(range(runtime), avg_centrality_network, 'k-', label='Centrality, Memory 150')
    plt.plot(range(runtime), avg_exposure_network, 'g-', label='Exposure, Memory 150')
    plt.plot(range(runtime), avg_all_network, 'm-', label='All, Memory 150')

    plt.legend()
    plt.show()
    
if __name__ == '__main__':
    main()
