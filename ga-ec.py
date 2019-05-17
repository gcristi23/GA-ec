import copy

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


class Node:
    operations = ['+', '-', '*', '/', '**', 'np.log']
    unary = ['np.log']
    max_depth = 5

    def __init__(self, val=None, parent=None, left=None, right=None, level=0):
        self.level = level
        if val is None and parent is None and left is None and right is None:
            self.gen_random(['vars[0]', 'vars[1]', 'vars[2]'])
        else:
            self.val = val
            if val not in Node.operations:
                self.leaf = True
            else:
                self.leaf = False
            self.parent = parent
            self.left = None
            self.right = None

    def propagate(self):
        if self.leaf is True:
            return self.val
        else:
            if self.val in Node.unary:
                return "%s(%s)" % (self.val, self.left.propagate())
            return "(%s%s%s)" % (self.left.propagate(), self.val, self.right.propagate())

    def get_random(self, level=None):
        if level is None:
            chance = np.random.rand()
            if chance > 0.5:
                if self.leaf is True:
                    return None
                node = self.left.get_random()
                if node is None:
                    if self.val in Node.unary:
                        return None
                    else:
                        node = self.right.get_random()
                    if node is None:
                        return None
                    else:
                        return node
                else:
                    return node
            else:
                return self
        else:
            if self.level == level or self.leaf:
                return self
            else:
                chance = np.random.rand()
                if chance > 0.5 or self.val in Node.unary:
                    return self.left.get_random(level)
                else:
                    return self.right.get_random(level)

    def get_parent(self):
        return self.parent

    def set_parent(self, parent):
        self.parent = parent

    def set_left_child(self, child):
        self.left = child

    def set_right_child(self, child):
        if self.val not in Node.unary:
            self.right = child

    def get_left_child(self):
        return self.left

    def get_right_child(self):
        if self.val in Node.unary:
            return self.left
        return self.right

    def is_leaf(self):
        return self.leaf

    def set_level(self, level):
        self.level = level
        if not self.leaf:
            self.left.set_level(level + 1)
            if self.val not in Node.unary:
                self.right.set_level(level + 1)

    def gen_random(self, variables):
        is_leaf = np.random.rand()
        if is_leaf > 1 - 0.1 * self.level or self.level == Node.max_depth:
            self.leaf = True
            is_variable = np.random.rand()
            if is_variable > 0.5:
                self.val = np.random.choice(variables)
            else:
                self.val = str(np.random.rand() * np.random.randint(1, 10) * np.random.randint(1, 10))
        else:
            self.val = np.random.choice(Node.operations)
            self.leaf = False
            if self.val not in Node.unary:
                self.right = Node(level=self.level + 1)
                self.right.set_parent(self)
            else:
                self.right = None
            self.left = Node(level=self.level + 1)

            self.left.set_parent(self)

    def get_height(self, level=0):
        level += 1
        if self.leaf:
            return level
        a = self.left.get_height(level)
        if self.right is not None:
            b = self.right.get_height(level)
        else:
            b = -1
        return a if a > b else b


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
    level = np.random.randint(0, Node.max_depth)
    swap_nodes = [parent.get_random(level=level) for parent in parents]

    while swap_nodes[0] is None:
        swap_nodes[0] = parents[0].get_random(level=level)
    while swap_nodes[1] is None:
        swap_nodes[1] = parents[1].get_random(level=level)
    child = np.random.rand()
    if not swap_nodes[0].is_leaf():

        if child > 0.5:
            aux = swap_nodes[0].get_left_child()
            if not swap_nodes[1].is_leaf():
                swap_nodes[0].set_left_child(swap_nodes[1].get_left_child())
                swap_nodes[0].get_left_child().set_level(swap_nodes[0].level + 1)
                swap_nodes[1].set_left_child(aux)
                swap_nodes[1].get_left_child().set_level(swap_nodes[1].level + 1)
                return 1
        else:
            aux = swap_nodes[0].get_right_child()
            if not swap_nodes[1].is_leaf():
                swap_nodes[0].set_right_child(swap_nodes[1].get_right_child())
                swap_nodes[1].set_right_child(aux)
                swap_nodes[0].get_right_child().set_level(swap_nodes[0].level + 1)
                swap_nodes[1].get_right_child().set_level(swap_nodes[1].level + 1)
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
    pop_size = 100
    max_error = 0.98
    n_parents = int(pop_size * 0.3)
    scores = [0]
    pop = initialize(pop_size, vars)
    # for i in range(100):
    #     print(pop[i].propagate())
    # exit()
    while max(scores) < max_error or np.isnan(min(scores)):
        scores = []
        for p in pop:
            try:
                data_eval = eval(p.propagate())
                data_eval = (data_eval - np.mean(data_eval)) / np.std(data_eval)
                # score = np.sum((abs(data) - abs(data_eval)) ** 2)
                score = np.corrcoef(data, data_eval)[0, 1]
                if np.isnan(score):
                    score = 0.0
                scores.append(score)
            except:
                print("Can't")
                scores.append(0.0)
        print(max(scores))
        if max(scores) >= max_error:
            break
        ordered = sorted(zip(scores, pop), key=lambda x: x[0], reverse=True)[:n_parents]
        print(ordered[0][0])
        new_pop = [ordered[0][1]]
        print(new_pop[0].get_height())
        for i in range(1, pop_size):
            parents = np.random.random_integers(0, n_parents - 1, 2)
            parents = [copy.deepcopy(ordered[x][1]) for x in parents]
            crossover(parents)
            chance_mutation = np.random.rand()
            if chance_mutation < 0.35:
                mutate([parents[0]])
            if chance_mutation > 0.65:
                mutate([parents[1]])
            if parents[0].get_height() > Node.max_depth:
                parents[0] = Node()

            if parents[1].get_height() > Node.max_depth:
                parents[1] = Node()
            new_pop.extend(parents)
        pop = new_pop

    plt.plot(data)
    print(pop[scores.index(min(scores))].propagate())
    plt.plot(eval(pop[scores.index(min(scores))].propagate()), color='r')
    plt.show()
    print(data)
    print(eval(pop[scores.index(min(scores))].propagate()))
    return pop[scores.index(min(scores))].propagate()


if __name__ == "__main__":
    data = pd.read_csv('data.csv')
    X = data
    y = data['C']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    ecuation = algoritm_genetic_blanao((X_train.C.values - np.mean(X_train.C.values)) / np.std(X_train.C.values),
                                       [X_train.S.values / X_train.K.values, X_train.Tau_cal.values,
                                        np.exp(1)])
    # algoritm_genetic_blanao(data.C.values[:93500], [data.S.values[:93500], data.K.values[:93500], data.R.values[:93500],
    #                                                np.exp(1)])
    vars = [X_test.S.values, X_test.K.values, X_test.R.values, np.exp(1)]
    res = eval(ecuation)
    # plt.plot(X_test.BS_impl.values)
    error = np.sum(np.abs(X_test.C.values - res) ** 2)
    print(error)
    error = np.sum(np.abs(X_test.C.values - X_test.BS_impl.values) ** 2)
    print(error)
    plt.plot(res, color='r')
    plt.plot(X_test.C.values)
    plt.show()
# In[ ]:
