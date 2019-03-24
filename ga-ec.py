#!/usr/bin/env python
# coding: utf-8

# In[194]:


import numpy as np
class Node:
    operations = ['+', '-', '*', '/']
    
    def __init__(self, val, left = None, right = None):
        self.val = val
        if val not in Node.operations:
            self.leaf = True
        else:
            self.leaf = False
        self.left = left
        self.right = right

    def propagate(self):
        if self.leaf == True:
            return self.val
        else:
            return "(%s%s%s)"%(self.left.propagate(),self.val,self.right.propagate())
        
    def left_child(self, left):
        self.left = left
        
    def right_child(self, right):
        self.right = right
        
    def get_random(self):
        chance = np.random.rand()
        if chance > 0.1:
            if self.leaf == True:
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
          
            


# In[158]:


import numpy as np


def gen_random():
        is_leaf = np.random.rand()
        if is_leaf > 0.4:
            is_variable = np.random.rand()
            if is_variable > 0.5:
                node = Node('time')
            else:
                node = Node(str(np.random.rand()))
        else:
            node = Node(np.random.choice(Node.operations), gen_random(), gen_random())
            
        return node
    


for i in range(30):
    print("-=-=-=-=-=-=-=-=-=-=======================================================-=-=-=-=-=-=-=-=-")
    try:
        root = Node(np.random.choice(Node.operations), gen_random(), gen_random())
        t = 0.4
        ec = root.propagate()
        print('%s=%s'%(ec,str(eval(ec))))
    except:
        print('ERROR!')


# In[206]:


import matplotlib.pyplot as plt
size = 1000
time = np.arange(size)*0.01
data = 0.24*time*time + 0.87*time*time + 0.11*time
pop = []
def do_this():
    pop_size = 20000
    i = 0
    pop = []
    while i < pop_size:
        try:
            new_node = gen_random()
            pop.append(new_node)
            ec = eval(pop[-1].propagate())
            i += 1
    #         print('%s=%s'%(ec,str(eval(ec))))
        except:
            print("Error!")
    return pop


def algoritm_genetic_blanao(data, time):
    scores = [100000]
    pop = do_this()
    while min(scores)>2 or np.isnan(min(scores)) :
        scores = []
        pop = do_this()
        for p in pop:
            data_eval = eval(p.propagate())
            score = np.sum((abs(data) - abs(data_eval))**2)
            if np.isnan(score):
                score = 9999999999.0
            scores.append(score)

        ordered = sorted(zip(scores, pop),key = lambda x: x[0])[:int(pop_size*0.2)]
        new_pop = [ordered[0][1]]
        print(ordered[0][0])

        i = 0

    time = time[-1]*200
    data = 0.24*time*time + 0.87*time*time + 0.11*time
    print(data)
    print(eval(pop[scores.index(min(scores))].propagate()))
    plt.plot(data)

    plt.plot(eval(pop[scores.index(min(scores))].propagate()))
    plt.show()
#         while i < pop_size:
#             try:
#                 new_node = Node(np.random.choice(Node.operations), gen_random(), gen_random())
#                 pop.append(new_node)
#                 ec = eval(pop[-1].propagate())
#                 i += 1
#             except:
#                 print("Error!")



algoritm_genetic_blanao(data, time)   


# In[ ]:




