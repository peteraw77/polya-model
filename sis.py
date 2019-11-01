import random
random.seed()

class SISNode:
    def __init__(self, neighborhood=[], status=0, delta=0.5, beta=0.5):
        self.neighborhood = neighborhood
        self.status = status
        self.delta = delta
        self.beta = beta

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
