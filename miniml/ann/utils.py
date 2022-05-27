#########################################
#         MLPlayground for Numpy        #
#      Machine Learning Techniques      #
#                 v1.0.0                #
#                                       #
#         Written by Lujia Zhong        #
#       https://lujiazho.github.io/     #
#              26 May 2022              #
#              MIT License              #
#########################################
import numpy as np
from .common import Placeholder

# use topology to generate sorted graph for better forward and backward operations
def topology(graph):
    sorted_node = []
    
    while len(graph) > 0: 

        all_inputs = []
        all_outputs = []
        
        for n in graph:
            all_outputs += graph[n]
            all_inputs.append(n)
        
        all_outputs = set(all_outputs)
        all_inputs = set(all_inputs)
    
        # find out node without indegree
        need_remove = all_inputs - all_outputs
    
        if len(need_remove) > 0: 
            # only choose one
            node = np.random.choice(list(need_remove))

            need_to_visited = [node]

            # when this is the last input node, add its all output to make sure all nodes are in sorted_node
            if len(graph) == 1: need_to_visited += graph[node]
            
            # delete the chosen node
            graph.pop(node)
            
            sorted_node += need_to_visited
        else: # is cycle
            break
        
    return sorted_node

# generate graph from dict structure
def feed_dict_2_graph(feed_dict):
    computing_graph = dict()
    
    nodes = [n for n in feed_dict]
    
    while nodes:
        n = nodes.pop(0) 
        
        if isinstance(n, Placeholder):
            n.value = feed_dict[n]
        
        if n in computing_graph:
            continue
        computing_graph[n] = list()
        
        for m in n.outputs:
            computing_graph[n].append(m)
            nodes.append(m)
    
    return computing_graph