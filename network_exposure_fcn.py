from polya import FiniteNode
# Calculates and returns an array of network exposure for each node at current time step
def network_exposure(nodes):

    net_exp = []
    for i in range(len(nodes)):
        red_sum = sum(nodes[i].additional_red)
        for index in nodes[i].neighborhood:
            red_sum = red_sum + sum(nodes[index].additional_red)
            
        total_red, total_black = nodes[i].construct_super_urn(nodes)

        net_exp.append((total_red + red_sum) / (total_red + total_black +red_sum))

    return net_exp
