#!/usr/bin/env python
# coding: utf-8

# In[194]:


import numpy as np
import matplotlib.pyplot as plt
import copy

class Node:
    operations = ['+', '-', '*', '/', '**']

    def __init__(self, val=None, parent=None, left=None, right=None):
        if val is None and parent is None and left is None and right is None:
            self.gen_random(['time', 'e'])
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
        if is_leaf > 0.4:
            self.leaf = True
            is_variable = np.random.rand()
            if is_variable > 0.5:
                self.val = np.random.choice(variables)
            else:
                self.val = str(np.random.rand()*np.random.randint(1,100)*np.random.randint(1,10))
        else:
            self.val = np.random.choice(Node.operations)
            self.leaf = False
            self.right = Node()
            self.left = Node()
            self.right.set_parent(self)
            self.left.set_parent(self)




size = 1000
time = np.arange(size) * 0.01
data = 1/(1 + np.exp(time))
pop = []


def initialize(pop_size):
    j = 0
    pop = []
    e = np.exp(1)
    while j < pop_size:
        try:
            new_node = Node()

            pop.append(new_node)
            eval(pop[-1].propagate())
            j += 1
        except:
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
        random_node.set_left_child(Node())
    else:
        random_node.set_right_child(Node())


def algoritm_genetic_blanao(data, time):
    pop_size = 60
    e = np.exp(1)
    n_parents = int(pop_size * 0.2)
    scores = [100000]
    pop = initialize(pop_size)
    while min(scores) > 66 or np.isnan(min(scores)):
        scores = []
        for p in pop:
            data_eval = eval(p.propagate())
            score = np.sum((abs(data) - abs(data_eval)) ** 2)
            if np.isnan(score):
                score = 9999999999.0
            scores.append(score)
        if min(scores) <= 100:
            break
        ordered = sorted(zip(scores, pop), key=lambda x: x[0])[:n_parents]
        print(ordered[0][0])
        new_pop = [ordered[0][1]]
        for i in range(1, pop_size):
            parents = np.random.random_integers(0, n_parents-1, 2)
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
    time = time[-1] * 200
    data = 0.24 * time * time + 0.87 * time * time + 0.11 * time
    print(data)
    print(eval(pop[scores.index(min(scores))].propagate()))

algoritm_genetic_blanao(data, time)

# In[ ]:
