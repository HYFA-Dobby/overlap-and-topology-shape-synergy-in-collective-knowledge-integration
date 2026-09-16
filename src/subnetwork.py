import numpy as np
import networkx as nx

def generate_subnetworks(
        sub_network_sizes: list[int],
        original_network : nx.Graph,
        base_node = int,
        overlapped_node_num = int,
        edge_wiring_prob = float
):  
    """
    Create sub networks that (sub_network_size - unoverlapped_node_num) nodes are overlapped
    
    Args:
        sub_network_sizes (int) : list of target size of each sub-network. The amount of sub_network_size - 1 nodes will be added to sub network. Must be > 0.
        original_network (nx.Graph) : The original network of sub graph. e.g. Random network, scale-free network , etc
        base_node (int) : The label number of base node for sub networks.
        unoverlapped_node_num (int) : The number of nodes that are not overlapped. Must be < sub_network_size.
        edge_wiring_prob (float) : The Probability of wiring edge to create sub network.
    """

    N = len(sub_network_sizes)

    #---------- containers ----------
    sub_networks = [nx.Graph() for _ in range(N)]
    sub_nodes = [set() for _ in range(N)]

    #---------- initialize base node ----------
    for i in range(N):
        sub_networks[i].add_node(base_node)
        sub_nodes[i].add(base_node)

    #---------- overlapped part generation ----------
    for _ in range(overlapped_node_num - 1):

        neighbor = set() #set for neighbor. reset by every step
        for node in sub_nodes[0]:
            neighbor.update(original_network.neighbors(node)) #neighbor of node in sub network node
        neighbor -= sub_nodes[0] #delete duplicated node which are already chosen

        if not neighbor:
            break

        new_node = np.random.choice(list(neighbor))#choose new node

        for i in range(N):
            start_candidates = [
                node for node in original_network.neighbors(new_node) if node in sub_nodes[i]
            ]  #start point list of edge on sub network

            if start_candidates:
                start = np.random.choice(start_candidates) #node that will be connected to new node

                sub_nodes[i].add(new_node)
                sub_networks[i].add_node(new_node)
                sub_networks[i].add_edge(start, new_node)

            else:
                break
    
    #---------- independent non-overlap growth ----------
    ban_nodes = set()
    while True:
        growing = [i for i in range(N) if len(sub_nodes[i]) < sub_network_sizes[i]]

        if not growing:
            break
            
        candidates = {}
        for i in growing:
            neighbor = set()
            for node in sub_nodes[i]:
                neighbor.update(original_network.neighbors(node))
            neighbor -= sub_nodes[i]
            neighbor -= ban_nodes
            candidates[i] = neighbor

        growing = [i for i in growing if candidates[i]]
        if not growing:
            print("Growth stopped : no candidates available")
            break

        used = set()
        new_nodes = {}

        for i in growing:
            available = list(candidates[i] - used)
            if not available:
                print("No feasible unique assignment")
                growing = []
                break 
            
            choice = np.random.choice(available)
            new_nodes[i] = choice
            used.add(choice)

        for i in new_nodes:
            start_candidates = [ 
                node for node in original_network.neighbors(new_nodes[i]) if node in sub_nodes[i]
            ] #start point list of edge on sub network

            start = np.random.choice(start_candidates) #node that will be connected to new node
            sub_nodes[i].add(new_nodes[i])
            sub_networks[i].add_node(new_nodes[i])
            sub_networks[i].add_edge(start, new_nodes[i])

            ban_nodes.add(new_nodes[i])

    #---------- edge wiring ----------
    for i in range(N):
        possible_edges = original_network.subgraph(sub_nodes[i]).edges()
        possible_edges = {(min(u,v), max(u,v)) for u, v in possible_edges}

        existing_edges = sub_networks[i].edges()
        existing_edges = {(min(u,v), max(u,v)) for u, v in existing_edges}
                          
        candidates = list(possible_edges - existing_edges)

        if candidates:
            probs = np.random.rand(len(candidates))
            wiring_edges = [e for e, p in zip(candidates, probs) if p < edge_wiring_prob]
            sub_networks[i].add_edges_from(wiring_edges)

    return sub_nodes, sub_networks