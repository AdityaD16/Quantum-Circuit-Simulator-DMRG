# src/utils/graph_gen.py

import networkx as nx

def nearest_neighbour_edge_list(no_qubits):
    """Generates a linear chain (1D nearest-neighbor) edge list."""
    G = nx.Graph()
    G.add_nodes_from(range(no_qubits))
    for i in range(no_qubits-1):
        G.add_edge(i,i+1)
    return G

def generate_3_regular_graph(num_qbts, seed=None):
    """Generates a random 3-regular graph."""
    if num_qbts % 2 != 0 or num_qbts <= 3:
        raise ValueError("A 3-regular graph requires an even number of vertices > 3.")
    G = nx.random_regular_graph(3, num_qbts, seed=seed)
    return G

def tree_like_edge_list(network_structure):
    """Generates a tree-like edge list based on the network hierarchy."""
    G = nx.Graph()
    num_nodes = network_structure[-1]
    G.add_nodes_from(range(num_nodes))
    for l in range(len(network_structure)-1):
        cluster_size = network_structure[-1]//network_structure[-2-l]
        clusters = []

        for i in range(0, num_nodes, cluster_size):
            cluster_nodes = list(range(i, i + cluster_size,network_structure[-1]//network_structure[-1-l]))
            clusters.append(cluster_nodes)
            # Fully connect nodes within each cluster
            for j in cluster_nodes:
                for k in cluster_nodes:
                    if j!=k:
                        G.add_edge(j,k)   
    return G

def remove_edge_and_get_endpoints(G):
    for u, v in list(G.edges()):
        if G.degree[u] == 3 and G.degree[v] == 3:
            G.remove_edge(u, v)
            return (u, v)
    raise ValueError("No suitable edge found to remove.")

def construct_bridged_graph(num_units, vertices_per_unit):
    """Constructs a graph from smaller, highly-connected units linked by bridges."""
    units = []
    removed_edges = []  
    for i in range(num_units):
        G_unit = generate_3_regular_graph(vertices_per_unit)
        ep = remove_edge_and_get_endpoints(G_unit)
        removed_edges.append(ep)  
        units.append(G_unit)
    
    G_union = nx.disjoint_union_all(units)
    offsets = [i * vertices_per_unit for i in range(num_units)]
    dangling = [] 
    for i in range(num_units):
        u, v = removed_edges[i]
        offset = offsets[i]
        dangling.append((u + offset, v + offset))
    
    for i in range(num_units):
        u_i = dangling[i][0]
        next_index = (i + 1) % num_units
        v_next = dangling[next_index][1]
        G_union.add_edge(u_i, v_next)
        
    return G_union

def generate_Erdo_Renyi_graph(num_qbts, edge_prob, seed=None):
    """Generates an Erdős-Rényi random graph."""
    G = nx.erdos_renyi_graph(num_qbts, edge_prob, seed=seed)
    return G

def edge_extract(G,Str="Blind"):
    assert Str == "Blind" or Str=="Naive",print("Str should be Naive or Blind")
    if Str == "Blind":
        return list(G.edges())
    else:
        communities = nx.community.greedy_modularity_communities(G, resolution=.8)
        community_lists = [list(c) for c in communities]
        mapping = {}
        new_label = 0
        for cluster_idx, cluster in enumerate(community_lists):
            for node in cluster:
                mapping[node] = new_label
                new_label += 1
        G_relabeled = nx.relabel_nodes(G, mapping)
        return list(G_relabeled.edges())        


