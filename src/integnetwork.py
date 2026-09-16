import numpy as np
import networkx as nx

def generate_integrated_network(
        original_network : nx.Graph,
        *sub_networks : nx.Graph
) -> nx.Graph:
    """
    Cerate integrated network from sub networks

    Args:
        original_network (nx.Graph) : The original network e.g. Random, scale-free, Small World network etc.
        *sub_networks (nx.Graph) : The sub networks to be integrated.
    """
    #---------- union node set ----------
    integrated_nodes = set()
    for sub in sub_networks:
        integrated_nodes.update(sub.nodes)

    #---------- union of actual edges ----------
    sub_edge_union = set()
    for sub in sub_networks:
        sub_edge_union |= {(min(u,v), max(u,v)) for u, v in sub.edges}

    #---------- union of full-connectivity edges (within each sub network) ----------
    sub_fc_edge_union = set()
    for sub in sub_networks:
        fc = original_network.subgraph(sub.nodes)
        sub_fc_edge_union |= {(min(u,v), max(u,v)) for u, v in fc.edges }

    #---------- temp network over all nodes ----------
    temp = original_network.subgraph(integrated_nodes)
    temp_edge = {(min(u,v), max(u,v)) for u, v in temp.edges}


    #---------- build integrated ----------
    integrated = nx.Graph()
    integrated.add_nodes_from(integrated_nodes)
    integrated.add_edges_from(sub_edge_union)

    return integrated