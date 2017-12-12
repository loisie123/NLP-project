"""
TO DO
    - Hoe zien de nodes eruit?? Zijn het getallen of strings?
    - matrix_to_graph verwijderen
    - 

"""



import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
#% matplotlib inline

import itertools

def matrix_to_graph(A, sent, labels, digraph=False):
    """
    Turns a numpy 
    """
    k = A.shape[0]
    nodes = range(k)
    if digraph:
        # This function turns a np matrix into a directed graph
        G = nx.from_numpy_matrix(A, create_using=nx.DiGraph())
        # Trying to get labels on the arcs... No succes yet
        labels = {(i, j, A[i,j]): np.random.choice(labels) for i,j in itertools.product(nodes, nodes)}
    else:
        G = nx.from_numpy_matrix(A)
        
    for k, w in enumerate(sent):
        G.node[k]['word'] = w
        
    weighted_edges = {(i, j): A[i,j] for i,j in itertools.product(nodes, nodes)}
    return G, weighted_edges


def contract(graph, cycle, cycle_node):
    """
        Contract graph: replace cycle in directed graph by a node.
        :param graph: a nx.DiGraph() object
        :param cycle: list of nodes which form a cycle in the graph
        :param cycle_node: node which will replace the cycle
    """
    
    # create new directed graph with nodes
    contracted_graph = nx.DiGraph()
    contracted_graph.add_nodes_from(graph)
    contracted_graph.remove_nodes_from(cycle)
    contracted_graph.add_node(cycle_node)
    
    # add the edges in right form
    # remember the corresponding edge from graph
    # calculate the weight of this edge
    for (u,v) in graph.edges:
        print("edge:", (u,v))
        if u not in cycle and v in cycle:
            contracted_graph.add_edge(u, cycle_node)
            best_node = arg_max(graph, v)
            contracted_graph[u][cycle_node]['edge'] = (u, v)
            print("contracted_graph[u][cycle_node]['edge']=", contracted_graph[u][cycle_node]['edge'])
            contracted_graph[u][cycle_node]['weight'] = graph[u][v]['weight'] + graph[best_node][v]['weight']
        elif u in cycle and v not in cycle:
            contracted_graph.add_edge(cycle_node, v)
            contracted_graph[cycle_node][v]['edge'] = (u, v)
            print("contracted_graph[cycle_node][v]['edge']=",contracted_graph[cycle_node][v]['edge'])
            contracted_graph[cycle_node][v]['weight'] = graph[u][v]['weight']
        elif u not in cycle and v not in cycle:
            contracted_graph.add_edge(u, v)
            contracted_graph[u][v]['edge'] = (u, v)
            print("contracted_graph[u][v]['edge']=",contracted_graph[u][v]['edge'])
            contracted_graph[u][v]['weight'] = graph[u][v]['weight']
    print("graph=", contracted_graph.edges(data=True))
    return contracted_graph


def expand(graph, contracted_tree, cycle, cycle_node):
    """
    Expand graph: replace node in a tree by cycle.
    Input:
        :param graph: a nx.DiGraph() object
        :param cycle: list of nodes which form a cycle in the graph
        :param cycle_node: node which will replace the cycle
    Output: tree
    """
    
    # create new directed graph with nodes
    tree = nx.DiGraph()
    tree.add_nodes_from(graph)
    
    # add the edges in right form
    # calculate the weight of this edge
    for (u,v) in contracted_tree.edges:
        # add corresponding edge from ... graph with right weight
        (m, n) = contracted_tree[u][v]['edge']
        tree.add_edge(m, n)
        tree[m][n]['weight'] = graph[m][n]['weight']
        tree[m][n]['edge'] = graph[m][n]['edge']
        
        # add all cycle egdes expect for (pi(v),v) with right weight
        if v == cycle_node:
            cycle_prime = cycle
            cycle_prime.remove(n)
            best_node = arg_max(graph, n, nodes = cycle_prime)
            if not(cycle[-1] == best_node and cycle[0]==n):
                tree.add_edge(cycle[-1], cycle[0])
                tree[cycle[-1]][cycle[0]]['weight'] = graph[cycle[-1]][cycle[0]]['weight']
                tree[cycle[-1]][cycle[0]]['edge'] = graph[cycle[-1]][cycle[0]]['edge']
            for i in range(len(cycle)-1):
                if not(cycle[i] == best_node and cycle[i+1]==n):
                    tree.add_edge(cycle[i], cycle[i+1])
                    tree[cycle[i]][cycle[i+1]]['weight'] = graph[cycle[i]][cycle[i+1]]['weight']
                    tree[cycle[i]][cycle[i+1]]['edge'] = graph[cycle[i]][cycle[i+1]]['edge']
    return tree


def arg_max(graph, receiver, nodes=None):
    
    """
    Input is directed graph and a node.
    Output is tuple which represents edge with maximal weight and this maximal weight.
    """
    if nodes == None: nodes = graph.nodes
    max_score = None
    for node in nodes:
        if (node, receiver) in graph.edges:
            if max_score == None or graph[node][receiver]['weight'] > max_score:
                max_score = graph[node][receiver]['weight']
                best_node = node
    return best_node



def find_cycle(edges):
    """
    Input is a list of edges.
    Returns a list of nodes of the first cycle which has been found.
    Returns empty list if there is no cycle in the graph.
    """
    for (i,j) in edges:
        cycle = [i, j]
        for (k,l) in edges:
            if k == cycle[-1] and l == i:
                return cycle
            elif k == cycle[-1]:
                cycle.append(l)
    return []


def maximum_spanning_tree(graph, root):
    # checks if root is in graph
    if root not in graph.nodes: raise ValueError("The root is not a node of the graph.")
    
    # remove edges with as destination root
    destination_root = []
    for edge in graph.edges:
        if root == edge[1]: destination_root.append(edge)
    graph.remove_edges_from(destination_root)
    
    # add edge information
    for (u,v) in graph.edges:
        if 'edge' not in graph[u][v]:
            graph[u][v]['edge'] = (u,v)
    
    # remove parallel edges
    edge_dict = 
    for edge in edge_dict:
        if edge_dict[edge] > 1:
            
    
    
    # for i, edge in enumerate(graph.edges):
    best_edges = []
    for node in graph.nodes:
        if node != root:
            best_node = arg_max(graph, node)
            best_edges.append((best_node, node))
    # find a cycle
    cycle = find_cycle(best_edges)
        
    if len(cycle) == 0:
        graph.remove_edges_from(graph.edges - best_edges)
        return graph
    else:
        # afhankelijk van hoe de nodes eruit zien moet je ze kiezen !!!!!!!!!!!!!
        cycle_node = max(graph.nodes) + 1
        # eleminate cycle in graph
        contracted_graph = contract(graph, cycle, cycle_node)
        # find max spanning tree if this smaller graph
        contracted_tree = maximum_spanning_tree(contracted_graph, root)
        # use this to find max spanning tree
        return expand(graph, contracted_tree, cycle, cycle_node)