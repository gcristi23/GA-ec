import copy

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class Node:
    operations = ['+', '-', '*', '/', '**']
    max_depth = 3

    def __init__(self, val=None, parent=None, left=None, right=None, level=0):
        self.level = level
        if val is None and parent is None and left is None and right is None:
            self.gen_random(['vars[0]', 'vars[1]', 'vars[2]', 'vars[3]'])
        else:
            self.val = val
            if val not in Node.operations:
                self.leaf = True
            else:
                self.leaf = False
            self.parent = parent
            self.left = left
            self.right = right

    def propagate(self):
        if self.leaf is True:
            return self.val
        else:
            return "(%s%s%s)" % (self.left.propagate(), self.val, self.right.propagate())

    def get_random(self):
        chance = np.random.rand()
        if chance > 0.5:
            if self.leaf is True:
                return None
            node = self.left.get_random()
            if node is None:
                node = self.right.get_random()
                if node is None:
                    return None
                else:
                    return node
            else:
                return node
        else:
            return self

    def get_parent(self):
        return self.parent

    def set_parent(self, parent):
        self.parent = parent

    def set_left_child(self, left):
        self.left = left

    def set_right_child(self, right):
        self.right = right

    def get_left_child(self):
        return self.left

    def get_right_child(self):
        return self.right

    def is_leaf(self):
        return self.leaf

    def gen_random(self, variables):
        is_leaf = np.random.rand()
        if is_leaf > 0.4 or self.level == Node.max_depth:
            self.leaf = True
            is_variable = np.random.rand()
            if is_variable > 0.5:
                self.val = np.random.choice(variables)
            else:
                self.val = str(np.random.rand() * np.random.randint(1, 10) * np.random.randint(1, 10))
        else:
            self.val = np.random.choice(Node.operations)
            self.leaf = False
            self.right = Node(level=self.level + 1)
            self.left = Node(level=self.level + 1)
            self.right.set_parent(self)
            self.left.set_parent(self)


size = 1000
time = np.arange(size) * 0.01
data = 1 / (1 + np.exp(-time))
pop = []


def initialize(pop_size, vars):
    j = 0
    pop = []
    print(vars)
    while j < pop_size:
        try:
            new_node = Node()

            pop.append(new_node)
            eval(pop[-1].propagate())
            j += 1
        except Exception as e:
            print("Error!")
    return pop


def crossover(parents):
    # print("Before:")
    # print(parents[0].propagate())
    # print(parents[1].propagate())

    swap_nodes = [parent.get_random() for parent in parents]
    while swap_nodes[0] is None:
        swap_nodes[0] = parents[0].get_random()
    while swap_nodes[1] is None:
        swap_nodes[1] = parents[1].get_random()
    child = np.random.rand()
    if not swap_nodes[0].is_leaf():

        if child > 0.5:
            aux = swap_nodes[0].get_left_child()
            if not swap_nodes[1].is_leaf():
                swap_nodes[0].set_left_child(swap_nodes[1].get_left_child())
                swap_nodes[1].set_left_child(aux)
                return 1
        else:
            aux = swap_nodes[0].get_right_child()
            if not swap_nodes[1].is_leaf():
                swap_nodes[0].set_right_child(swap_nodes[1].get_right_child())
                swap_nodes[1].set_right_child(aux)
                return 1
    return 0
    # print("After:")
    # print(parents[0].propagate())
    # print(parents[1].propagate())


def mutate(node):
    random_node = node[0].get_random()
    while random_node is None:
        random_node = node[0].get_random()
    chance = np.random.rand()
    if chance > 0.5:
        random_node.set_left_child(Node(level=random_node.level))
    else:
        random_node.set_right_child(Node(level=random_node.level))


def algoritm_genetic_blanao(data, vars):
    pop_size = 5000
    max_error = 159000
    n_parents = int(pop_size * 0.3)
    scores = [99999999]
    pop = initialize(pop_size, vars)
    while min(scores) > max_error or np.isnan(min(scores)):
        scores = []
        for p in pop:
            try:
                data_eval = eval(p.propagate())
                score = np.sum((abs(data) - abs(data_eval)) ** 2)
                if np.isnan(score):
                    score = 9999999999.0
                scores.append(score)
            except:
                print("Can't")
                scores.append(9999999999.0)

        if min(scores) <= max_error:
            break
        ordered = sorted(zip(scores, pop), key=lambda x: x[0])[:n_parents]
        print(ordered[0][0])
        new_pop = [ordered[0][1]]
        for i in range(1, pop_size):
            parents = np.random.random_integers(0, n_parents - 1, 2)
            parents = [copy.deepcopy(ordered[x][1]) for x in parents]
            crossover(parents)
            chance_mutation = np.random.rand()
            if chance_mutation < 0.35:
                mutate([parents[0]])
            if chance_mutation > 0.65:
                mutate([parents[1]])
            new_pop.extend(parents)
        pop = new_pop

    plt.plot(data)
    print(pop[scores.index(min(scores))].propagate())
    plt.plot(eval(pop[scores.index(min(scores))].propagate()))
    plt.show()
    print(data)
    print(eval(pop[scores.index(min(scores))].propagate()))


if __name__ == "__main__":
    data = pd.read_csv('data.csv')

    # algoritm_genetic_blanao(data.C.values[:93500], [data.S.values[:93500], data.K.values[:93500], data.R.values[:93500],
    #                                                np.exp(1)])
    vars = [data.S.values[93500:], data.K.values[93500:], data.R.values[93500:], np.exp(1)]
    res = eval(
        "((((((((((((((((((((((((vars[1]/vars[0])/((((vars[0]/vars[1])+32.15060988297109)/13.811417830841105)/((((11.16352503810073/vars[1])+(4.536802971077117-0.6189935353746554))+vars[0])-vars[1])))+32.15060988297109)/13.811417830841105)-vars[2])-vars[2])-(vars[0]**((0.40300496752683923/24.527078532119337)-(20.904920847355058/vars[1]))))-((vars[1]+vars[1])**((0.533556692792978/vars[3])-(vars[1]+0.27594239833794676))))-(vars[0]**((0.40300496752683923/24.527078532119337)-(20.904920847355058/vars[1]))))-((vars[1]+vars[1])**((0.533556692792978/vars[3])-(vars[1]+0.27594239833794676))))-(vars[2]**34.10501879603735))-((5.4880150385650275-(2.4662211862675383**vars[3]))/((vars[0]+vars[1])*1.316012105775213)))-vars[2])-((5.4880150385650275-(2.4662211862675383**vars[3]))/((vars[0]+vars[1])*1.316012105775213)))-vars[2])-vars[2])-((vars[1]+vars[1])**((0.533556692792978/vars[3])-(vars[1]+0.27594239833794676))))-vars[2])-vars[2])-((1.7866782688039722+vars[2])/(vars[2]-(31.269626114424153/0.8930555937643061))))-vars[2])-((vars[1]+vars[1])**((0.533556692792978/vars[3])-(vars[1]+0.27594239833794676))))-(((7.8258837542966955+0.6195162032507688)*0.7001929432690519)/vars[0]))/(3.1153305930530175/((((28.841315686144544/vars[1])+(4.536802971077117-0.6189935353746554))+vars[0])-vars[1])))")
    plt.plot(data.C.values[93500:])
    error= np.sum(np.abs(data.C.values[93500:]-res)**2)
    print(error)
    plt.plot(res)
    plt.show()
# In[ ]:
