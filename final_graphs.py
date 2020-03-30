import networkx as nx
from copy import deepcopy
from polya import FiniteNode, network_infection_rate
from datetime import date
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat, savemat
import sys
import math
from network_exposure_fcn import network_exposure
import random

METHOD = sys.argv[1]
PARAMETER = sys.argv[2]
# TYPE 1 is 60% nodes have higher delta red values, TYPE 2 is 60% of nodes have higher initial red balls, TYPE 3 is initial infection rate 60%, TYPE 4 is infecting budget 60% higher
TYPE = sys.argv[3]

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

def build_network(adjacency, strategy):
    nodes = [None for x in range(len(adjacency))]

    # Initialize array of delta_reds to initialize nodes
    delta_red = [2]*len(adjacency)
    # Initialize array of initial red balls to initialize nodes
    red_balls = [10]*len(adjacency)

    # Set 60% of delta_red vals higher if TYPE 1
    if TYPE == '1':
        for i in range(math.ceil(len(adjacency)*0.6)):
            delta_red[i] = 4
        random.seed(420)
        random.shuffle(delta_red)

    # Set 60% of nodes to have higher initial infection if TYPE 2
    if TYPE == '2':
        for i in range(math.ceil(len(adjacency)*0.6)):
            red_balls[i] = 15
        random.seed(420)
        random.shuffle(red_balls)

    # Set initial infection to 60% if TYPE 3
    if TYPE == '3':
        for i in range(len(adjacency)):
            red_balls[i] = 15

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
                    nodes[i] = FiniteNode([j], red_balls=red_balls[i], delta_red=delta_red[i], delta_black=delta_black[i])
                else:
                    nodes[i].add_neighbor(j)
    return nodes, delta_red

def get_delta_black(nodes, adjacency, delta_red, strategy):
    graph = nx.from_numpy_matrix(adjacency)
    centralities = nx.closeness_centrality(graph)
    degrees = []
    for i in range(graph.number_of_nodes()):
        degrees.append(graph.degree[i])

     # Budget for curing is same as total number of delta_red added to network (unless TYPE 1)
    if TYPE != 1:
        budget = sum(delta_red)
    else:
        budget = 2*len(adjacency)

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

    elif strategy == 'uniform':
        for i in range(graph.number_of_nodes()):
            delta_black.append(budget/graph.number_of_nodes())

    elif strategy == 'sis':
        # Make array of negative degrees for use in arg_sort
        neg_deg = []
        for key in degrees:
            neg_deg.append(-degrees[key])
        # Get array of sorted indices to assign curing resources to most central nodes in graph
        sort_index = np.argsort(neg_deg)
        delta_black = [0]*graph.number_of_nodes()

        # calculate eigenvalues
        w,_ = np.linalg.eig(adjacency)
        w = [abs(x) for x in w]
        max_eig = np.amax(w)

        if (budget -  (1.01*max_eig*delta_red[0]) < 0):
            raise ValueError('Black balls required for first node greater than budget.')

        else:
            j = 0
            while (budget -  (1.01*max_eig*delta_red[0]) > 0):        
                delta_black[sort_index[j]] = 1.01*max_eig*delta_red[0]
                budget = budget - (1.01*max_eig*delta_red[0])
                j = j + 1

            # Allocate any remaining budget
            if (budget > 0):
                delta_black[sort_index[j]] = budget
            
    return delta_black

def simulation(adjacency, runtime, strategy):
    nodes, delta_red = build_network(adjacency, strategy)

    # run the simulation
    avg_infection_rate = []
    
    for t in range(runtime):
        
        # Get array of delta_black if update is required at each time step
        if strategy == 'exposure' or strategy == 'all':
            delta_black = get_delta_black(nodes, adjacency, delta_red, strategy)
                
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
    elif METHOD == '-g':
        adj_matrix = construct_barabasi_graph(int(PARAMETER))
    else:
        raise ValueError('Program expects method flag')

    # Node degree and centrality heuristics
    avg_degree_network = [0 for x in range(runtime)]
    avg_closeness_network = [0 for x in range(runtime)]
    avg_centrality_network = [0 for x in range(runtime)]
    avg_exposure_network = [0 for x in range(runtime)]
    avg_all_network = [0 for x in range(runtime)]

    # Uniform heuristic
    avg_uni_network = [0 for x in range(runtime)]

    # SIS strategy only applicable if delta_red constant for every node
    #if TYPE != '1':
        #avg_sis_network = [0 for x in range(runtime)]

    print('Simulating...')
    for x in tqdm(range(trials)):
        degree_network = simulation(adj_matrix, runtime, 'degree')
        closeness_network = simulation(adj_matrix, runtime, 'closeness')
        centrality_network = simulation(adj_matrix, runtime, 'centrality')
        exposure_network = simulation(adj_matrix, runtime, 'exposure')
        all_network = simulation(adj_matrix, runtime, 'all')
        uni_network = simulation(adj_matrix, runtime, 'uniform')
        #if TYPE != '1':
           # sis_network = simulation(adj_matrix, runtime, 'sis')

        avg_degree_network = [ x + y for x,y in zip(avg_degree_network,degree_network) ]
        avg_closeness_network = [ x + y for x,y in zip(avg_closeness_network,closeness_network) ]
        avg_centrality_network = [ x + y for x,y in zip(avg_centrality_network,centrality_network) ]
        avg_exposure_network = [ x + y for x,y in zip(avg_exposure_network,exposure_network) ]
        avg_all_network = [ x + y for x,y in zip(avg_all_network,all_network) ]
        avg_uni_network = [ x + y for x,y in zip(avg_uni_network,uni_network) ]
        #if TYPE != '1':
            #avg_sis_network = [ x + y for x,y in zip(avg_sis_network,sis_network) ]
                          
    avg_degree_network = [ x / trials for x in avg_degree_network ]
    avg_closeness_network = [ x / trials for x in avg_closeness_network ]
    avg_centrality_network = [ x / trials for x in avg_centrality_network ]
    avg_exposure_network = [ x / trials for x in avg_exposure_network ]
    avg_all_network = [ x / trials for x in avg_all_network ]
    avg_uni_network = [ x / trials for x in avg_uni_network ]
    #if TYPE != '1':
        #avg_sis_network = [ x / trials for x in avg_sis_network ]

    plt.figure('Final Graphs')
    plt.plot(range(runtime), avg_degree_network, 'r-', label='Degree, Memory 150')
    plt.plot(range(runtime), avg_closeness_network, 'b-', label='Closeness, Memory 150')
    plt.plot(range(runtime), avg_centrality_network, 'k-', label='Centrality, Memory 150')
    plt.plot(range(runtime), avg_exposure_network, 'g-', label='Exposure, Memory 150')
    plt.plot(range(runtime), avg_all_network, 'm-', label='All, Memory 150')
    plt.plot(range(runtime), avg_uni_network, 'c-', label='Uniform, Memory 150')
    #if TYPE != '1':
        #plt.plot(range(runtime), avg_sis_network, 'y-', label='SIS, Memory 150')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()
