import random
random.seed()

def Node:
    def __init__(self, neighborhood=[], red_balls=1, black_balls=1, delta=3):
        self.neighborhood = neighborhood
        self.red_balls = red_balls
        self.black_balls = black_balls

    def draw(self, nodes):
        # construct super urn
        total_red = red_balls
        total_black = black_balls

        for address in neighborhood:
            node = nodes[address]
            total_red = total_red + node.red_balls
            total_black = total_black + node.black_balls

        # draw
        red_prob = total_red / (total_red + total_black)
        # should this be <= or < ?
        if (random.uniform(0,1) <= red_prob):
            red_balls = red_balls + delta
        else:
            black_balls = black_balls + delta

def MemorylessNode:
    def __init__(self, neighborhood=[], red_balls=1, black_balls=1, memory=3, delta=3):
        self.neighborhood = neighborhood
        self.red_balls = red_balls
        self.black_balls = black_balls
        self.additional_red = [0 for x in range(memory)]
        self.additional_black = [0 for x in range(memory)]

    def update(self):
        red_balls = red_balls - additional_red.pop(0)
        black_balls = black_balls - additional_black.pop(0)

    def draw(self, nodes):
        # construct super urn
        total_red = red_balls
        total_black = black_balls

        for address in neighborhood:
            node = nodes[address]
            total_red = total_red + node.red_balls
            total_black = total_black + node.black_balls

        # draw
        red_prob = total_red / (total_red + total_black)
        # should this be <= or < ?
        if (random.uniform(0,1) <= red_prob):
            red_balls = red_balls + delta
            additional_red.append(delta)
            additional_black.append(0)
        else:
            black_balls = black_balls + delta
            additional_black.append(delta)
            additional_red.append(delta)
