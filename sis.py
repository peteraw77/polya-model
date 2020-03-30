import random
random.seed()

class SISNode:
    def __init__(self, delta, beta, neighborhood=[], status = 0, prob_infection = (2/3)):
        self.neighborhood = neighborhood
        self.status = status
        self.delta = delta
        self.beta = beta
        # Initial prob same as initial ratio of red balls in urns
        self.prob_infection = prob_infection

    def add_neighbor(self, neighbor):
        self.neighborhood.append(neighbor)

    def change_status(self, nodes):
        if self.status == 1:
            # attempt to recover from infection
            if (random.random() < delta):
                self.status = 0
        else:
            # check to see if infected by a neighbor
            probability_no_infection = 1
            for address in self.neighborhood:
                node = nodes[address]
                if node.status == 1:
                    probability_no_infection = probability_no_infection * (1 - beta)

                if (random.seed() < 1 - probability_no_infection):
                    self.status = 1

        return self.status

    # Function to calculate node's prob of infection at time t
    def change_prob_infection(self, nodes):
        product = 1
        # Calculate that weird product in the formula
        for address in self.neighborhood:
            product = product*(1-self.beta*nodes[address].prob_infection)
        self.prob_infection = self.prob_infection*(1-self.delta) + (1-self.prob_infection)*(1-product)

# Calculate network infection rate at time t
def network_infection_rate_SIS(nodes):
    infection_rate = 0
    for node in nodes:
        infection_rate = infection_rate + node.prob_infection

    return infection_rate / len(nodes)
