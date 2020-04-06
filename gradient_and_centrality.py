import networkx as nx
from copy import deepcopy
from polya import FiniteNode, InfiniteNode, network_infection_rate, f_n
import matplotlib.pyplot as plt
from scipy.io import loadmat, savemat
import numpy as np
from datetime import date
import numpy as np
from tqdm import tqdm
import sys
from network_exposure_fcn import network_exposure

from gradient import construct_barabasi_graph, load_graph
from gradient import simulation as gradient_simulation

from heuristic import simulation as heuristic_simulation

METHOD = sys.argv[1]
PARAMETER = sys.argv[2]

def main():
    trials = 50
    runtime = 1000
    if METHOD == '-f':
        adj_matrix = load_graph(PARAMETER)
    elif METHOD == '-g':
        adj_matrix = construct_barabasi_graph(int(PARAMETER))
    else:
        raise ValueError('Program expects method flag')

    avg_gradient_network = [0 for x in range(runtime)]

    print('Simulating...')
    for x in tqdm(range(trials)):
        gradient_network = gradient_simulation(adj_matrix, FiniteNode, runtime, 'neutral')
        centrality_network = heuristic_simulation(np.array(adj_matrix), runtime, 'centrality')

        avg_gradient_network = [ x + y for x,y in zip(avg_gradient_network,gradient_network) ]
        avg_centrality_network = [ x + y for x,y in zip(centrality_network,centrality_network) ]
    avg_gradient_network = [ x / trials for x in avg_gradient_network ]
    avg_centrality_network = [ x / trials for x in avg_centrality_network ]

    plt.figure('SuperimposedInfectionRates')
    plt.plot(range(runtime), avg_gradient_network, 'r-', label='Gradient, Memory 150')
    plt.plot(range(runtime), avg_centrality_network, 'b-', label='Centrality, Memory 150')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()
