import random
random.seed()

class InfiniteNode:
    def __init__(self, neighborhood=[], red_balls=1, black_balls=1, delta_red=1, delta_black=1):
        self.neighborhood = neighborhood
        self.red_balls = red_balls
        self.black_balls = black_balls
        self.delta_red = delta_red
        self.delta_black = delta_black

    def update(self):
        # do nothing!
        return None

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
        # construct super urn
        total_red,total_black = self.construct_super_urn(nodes)

        # draw
        red_prob = total_red / (total_red + total_black)
        if (random.random() < red_prob):
            self.red_balls = self.red_balls + self.delta_red
            return 1
        else:
            self.black_balls = self.black_balls + self.delta_black
            return 0

class FiniteNode:
    def __init__(self, neighborhood=[], red_balls=1, black_balls=1, memory=12, delta_red=1, delta_black=1):
        self.neighborhood = neighborhood
        self.red_balls = red_balls
        self.black_balls = black_balls
        self.additional_red = [0 for x in range(memory)]
        self.additional_black = [0 for x in range(memory)]
        self.delta_red = delta_red
        self.delta_black = delta_black

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
        if (random.random() < red_prob):
            self.red_balls = self.red_balls + self.delta_red
            self.additional_red.append(self.delta_red)
            self.additional_black.append(0)
            return 1
        else:
            self.black_balls = self.black_balls + self.delta_black
            self.additional_black.append(self.delta_black)
            self.additional_red.append(0)
            return 0

# functions for network analysis
def network_infection_rate(nodes):
    infection_rate = 0
    for node in nodes:
        total_red, total_black = node.construct_super_urn(nodes)
        infection_rate = infection_rate + total_red / (total_red + total_black)

    return infection_rate / len(nodes)

def gradient_function(nodes, curing, last_nodes):
    f = 0
    for i in range(len(nodes)):
        red_sum = sum(nodes[i].additional_red)
        black_sum = sum(nodes[i].additional_black)
        for neighbor in nodes[i].neighborhood:
            red_sum = red_sum + sum(nodes[neighbor].additional_red)
            black_sum = black_sum + sum(nodes[neighbor].additional_black)

        total_red, total_black = nodes[i].construct_super_urn(nodes)
        total_red_prev, total_black_prev = last_nodes[i].construct_super_urn(last_nodes)
        c = total_red + nodes[i].delta_red * (total_red_prev / (total_black_prev + total_red_prev) ) + red_sum
        d = c + total_black + black_sum

        sigma = curing[i]*(1 - (total_red_prev / (total_black_prev + total_red_prev) ))

        for j in range(len(last_nodes[i].neighborhood)):
            n_red, n_black = last_nodes[last_nodes[i].neighborhood[j]].construct_super_urn(last_nodes)
            neighbor_exp = (n_red / (n_red + n_black))

            sigma = sigma + curing[j]*(1-neighbor_exp)

        f = f + c / (d + sigma)
    f = f / len(nodes)

    return f

def f_n(nodes, curing, last_nodes):
    f = 0
    for i in range(len(nodes)):
        for previous in last_nodes:
            delta_red_sum = 0
            delta_black_sum = 0
            for j in previous[i].neighborhood:
                delta_red_sum += sum(previous[j].additional_red)
                delta_black_sum += sum(previous[j].additional_black)
        total_red, total_black = nodes[i].construct_super_urn(nodes)
        total_red_prev, total_black_prev = last_nodes[-1][i].construct_super_urn(last_nodes[-1])

        c = total_red + delta_red_sum + nodes[i].delta_red * \
                (total_red_prev / (total_red_prev + total_black_prev))
        d = c + total_black + delta_black_sum

        sigma = 0
        for j in last_nodes[-1][i].neighborhood:
            red,black = last_nodes[-1][j].construct_super_urn(last_nodes[-1])
            sigma += curing[j] * (1 - red / (red + black))

        f += c / (d + sigma)

    return f / len(nodes)

def f_n_memoryless(nodes, curing, last_nodes):
    f = 0
    for i in range(len(nodes)):
        for previous in last_nodes:
            delta_red_sum = 0
            delta_black_sum = 0
            for j in nodes[i].neighborhood:
                # should be previous[j].delta_x, not sum()
                delta_red_sum += sum(nodes[j].additional_red)
                delta_black_sum += sum(nodes[j].additional_black)
        total_red, total_black = nodes[i].construct_super_urn(nodes)
        total_red_prev = total_red - nodes[i].additional_red[-1]
        total_black_prev = total_black - nodes[i].additional_black[-1]

        c = total_red + delta_red_sum + nodes[i].delta_red * \
                (total_red_prev / (total_red_prev + total_black_prev))
        d = c + total_black + delta_black_sum

        sigma = 0
        for j in last_nodes[-1][i].neighborhood:
            red,black = last_nodes[-1][j].construct_super_urn(last_nodes[-1])
            sigma += curing[j] * (1 - red / (red + black))

        f += c / (d + sigma)

    return f / len(nodes)
