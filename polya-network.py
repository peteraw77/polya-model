import random
random.seed()

class Node:
    def __init__(self, neighborhood=[], red_balls=1, black_balls=1, delta=3):
        self.neighborhood = neighborhood
        self.red_balls = red_balls
        self.black_balls = black_balls

    def draw(self, nodes):
        # construct super urn
        total_red = self.red_balls
        total_black = self.black_balls

        for address in self.neighborhood:
            node = nodes[address]
            total_red = total_red + node.red_balls
            total_black = total_black + node.black_balls

        # draw
        red_prob = total_red / (total_red + total_black)
        # should this be <= or < ?
        if (random.uniform(0,1) <= red_prob):
            self.red_balls = self.red_balls + self.delta
        else:
            self.black_balls = self.black_balls + delta

class MemorylessNode:
    def __init__(self, neighborhood=[], red_balls=1, black_balls=1, memory=3, delta=3):
        self.neighborhood = neighborhood
        self.red_balls = red_balls
        self.black_balls = black_balls
        self.additional_red = [0 for x in range(memory)]
        self.additional_black = [0 for x in range(memory)]
        self.delta = delta

    def update(self):
        self.red_balls = self.red_balls - self.additional_red.pop(0)
        self.black_balls = self.black_balls - self.additional_black.pop(0)

    def add_neighbor(self, neighbor):
        self.neighborhood.append(neighbor)

    def construct_super_urn(self, nodes):
        total_red = self.red_balls
        total_black = self.black_balls

        for address in self.neighborhood:
            node = nodes[address]
            total_red = total_red + node.red_balls
            total_black = total_black + node.black_balls

        return (total_red, total_black)

    def draw(self, nodes):
        total_red, total_black = self.construct_super_urn(nodes)

        # draw
        red_prob = total_red / (total_red + total_black)
        # should this be <= or < ?
        if (random.uniform(0,1) <= red_prob):
            self.red_balls = self.red_balls + self.delta
            self.additional_red.append(self.delta)
            self.additional_black.append(0)
            return 1
        else:
            self.black_balls = self.black_balls + self.delta
            self.additional_black.append(self.delta)
            self.additional_red.append(0)
            return 0

# functions for network analysis
def network_infection_rate(nodes):
    infection_rate = 0
    for node in nodes:
        total_red, total_black = node.construct_super_urn(nodes)
        infection_rate = infection_rate + total_red / (total_red + total_black)

    return infection_rate / len(nodes)
