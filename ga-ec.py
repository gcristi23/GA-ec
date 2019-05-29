import copy

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


class Node:
    operations = ['+', '-', '*', '/', '**', 'np.log', 'np.maximum', 'np.minimum', 'norm.cdf', 'np.sin', 'np.cos']
    unary = ['np.log', 'norm.cdf', 'np.sin', 'np.cos']
    functions = ['np.maximum', 'np.minimum']
    vars = ['vars[0]', 'vars[1]', 'vars[2]']
    max_depth = 7

    def __init__(self, val=None, parent=None, left=None, right=None, level=0):
        self.level = level
        if val is None and parent is None and left is None and right is None:
            self.gen_random(Node.vars)
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
            elif self.val in Node.functions:
                return "%s(%s,%s)" % (self.val, self.left.propagate(), self.right.propagate())
            return "(%s%s%s)" % (self.left.propagate(), self.val, self.right.propagate())

    def get_random(self, level=None):
        if level is None:
            chance = np.random.rand()
            if chance > 0.3:
                if self.leaf is True:
                    return self

                if np.random.rand() > 0.5 or self.val in Node.unary:
                    return self.left.get_random()
                else:
                    return self.right.get_random()
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

    def get_random_leaf(self):
        if self.leaf:
            return self
        else:
            chance = np.random.rand()
            if chance > 0.5 or self.val in Node.unary:
                return self.left.get_random_leaf()
            else:
                return self.right.get_random_leaf()

    def get_random_node(self):
        if self.is_leaf():
            return None
        if self.left.is_leaf():
            return self
        else:
            chance = np.random.rand()
            if chance > 0.5 or self.val in Node.unary:
                return self.left.get_random_node()
            else:
                return self.right.get_random_node()

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
            if is_variable > 0.3:
                self.val = np.random.choice(variables)
            else:
                self.val = str(np.random.normal(0, 1, 1)[0] * np.random.randint(1, 100))
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


def initialize(pop_size, vars):
    j = 0
    pop = []
    while j < pop_size:
        try:
            new_node = Node()

            pop.append(new_node)
            eval(pop[-1].propagate())
            j += 1
        except Exception as e:
            pass
    return pop


def crossover(parents):
    # print("Before:")
    # print(parents[0].propagate())
    # print(parents[1].propagate())
    # level = np.random.randint(0, Node.max_depth)
    # swap_nodes = [parent.get_random(level=level) for parent in parents]
    #
    # while swap_nodes[0] is None:
    #     swap_nodes[0] = parents[0].get_random(level=level)
    # while swap_nodes[1] is None:
    #     swap_nodes[1] = parents[1].get_random(level=level)
    # child = np.random.rand()
    # if not swap_nodes[0].is_leaf():
    #
    #     if child > 0.5:
    #         aux = swap_nodes[0].get_left_child()
    #         if not swap_nodes[1].is_leaf():
    #             swap_nodes[0].set_left_child(swap_nodes[1].get_left_child())
    #             swap_nodes[0].get_left_child().set_level(swap_nodes[0].level + 1)
    #             swap_nodes[1].set_left_child(aux)
    #             swap_nodes[1].get_left_child().set_level(swap_nodes[1].level + 1)
    #             return 1
    #     else:
    #         aux = swap_nodes[0].get_right_child()
    #         if not swap_nodes[1].is_leaf():
    #             swap_nodes[0].set_right_child(swap_nodes[1].get_right_child())
    #             swap_nodes[1].set_right_child(aux)
    #             swap_nodes[0].get_right_child().set_level(swap_nodes[0].level + 1)
    #             swap_nodes[1].get_right_child().set_level(swap_nodes[1].level + 1)
    #             return 1
    # return 0
    # print("After:")
    # print(parents[0].propagate())
    # print(parents[1].propagate())
    chance = np.random.rand()
    if chance > 0:
        values = [p.get_random_leaf() for p in parents]
    else:
        values = [p.get_random_node() for p in parents]
        if values.count(None) > 0:
            return
    aux = values[0].val
    values[0].val = values[1].val
    values[1].val = aux


def mutate_structure(node):
    random_node = node[0].get_random_node()
    while random_node is None:
        random_node = node[0].get_random()
    chance = np.random.rand()
    if chance > 0.5:
        random_node.set_left_child(Node(level=random_node.level))
    else:
        random_node.set_right_child(Node(level=random_node.level))


def mutate_value(node):
    if np.random.rand() > 0:
        a = node[0].get_random_leaf()
    else:
        a = node[0].get_random_node()
        if a is None:
            return
    chance = np.random.rand()
    if a.is_leaf():
        if chance > 0.5:
            is_variable = np.random.rand()
            if is_variable > 0.3:
                a.val = np.random.choice(Node.vars)
            else:
                a.val = str(np.random.normal(1, 1, 1)[0] * np.random.randint(1, 100))
        elif a.val.replace('.', '', 1).isdigit():
            a.val = str(eval(a.val) * np.random.normal(0, 1, 1)[0] * np.random.randint(1, 100))
        else:
            a.val = np.random.choice(Node.vars)
    else:
        a.val = np.random.choice(Node.operations)


def algoritm_genetic_blanao(data, vars):
    pop_size = 2000
    max_error = 100000
    n_parents = int(pop_size * 0.4)
    scores = [10000000]
    pop = initialize(pop_size, vars)
    # for i in range(100):
    #     print(pop[i].propagate())
    # exit()
    while min(scores) > max_error or np.isnan(min(scores)):
        scores = []
        for p in pop:
            try:
                data_eval = eval(p.propagate())
                # data_eval = (data_eval - np.mean(data_eval)) / np.std(data_eval)
                score = np.sum((abs(data) - abs(data_eval)) ** 2)
                # score = np.corrcoef(data, data_eval)[0, 1]
                if np.isnan(score):
                    score = 100000000.0
                scores.append(score)
            except:
                # print("Can't")
                scores.append(100000000.0)
        if min(scores) <= max_error:
            break
        ordered = sorted(zip(scores, pop), key=lambda x: x[0])[:n_parents]
        print(ordered[0][0])
        new_pop = [ordered[0][1]]
        print(new_pop[0].propagate())
        for i in range(1, pop_size):
            parents = np.random.random_integers(0, n_parents - 1, 2)
            parents = [copy.deepcopy(ordered[x][1]) for x in parents]
            crossover(parents)
            chance_mutation = np.random.rand()
            if chance_mutation < 0.2:
                mutate_structure([parents[0]])
                mutate_structure([parents[1]])
            if chance_mutation > 0.8:
                mutate_value([parents[0]])
                mutate_value([parents[1]])
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
    # - np.mean(X_train.C.values)) / np.std(X_train.C.values)
    ecuation = algoritm_genetic_blanao(X_train.C.values,
                                       [X_train.S.values, X_train.K.values, X_train.Tau_cal.values])
    # algoritm_genetic_blanao(data.C.values[:93500], [data.S.values[:93500], data.K.values[:93500], data.R.values[:93500],
    #                                                np.exp(1)])
    vars = [X_test.S.values, X_test.K.values, X_test.R.values]
    res = eval(ecuation)
    # plt.plot(X_test.BS_impl.values)
    error = np.sum(np.abs(X_test.C.values - res) ** 2)
    print(error)
    error = np.sum(np.abs(X_test.C.values - X_test.BS_impl.values) ** 2)
    print(error)
    plt.plot(res, color='r')
    plt.plot(X_test.C.values)
    plt.show()
    #
    # size = 1000
    # time = np.arange(size) * 0.01
    # data = 5.3*time**3
    # ecuation = algoritm_genetic_blanao(data, [time, np.exp(1)])
    # print(ecuation)
# In[ ]:
