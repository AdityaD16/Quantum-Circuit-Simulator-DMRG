# src/simulation.py

import numpy as np
import opt_einsum as oe
import time
import gc
from collections import deque

from src.network import Network
from src.utils.TN_gen import circuit_from_edge_list, Tree, MPS, full_network_from_edge_list
from src.utils.quantum_gates import hadamard_gate, X_QAOA, ZZ_QAOA

def partition_into_k_parts(lst, k):
    """
    Utility to partition a list into k roughly equal-sized sublists.

    Args:
        lst (list): The list to partition.
        k (int): The number of partitions.

    Returns:
        list: A list of k sublists.
    """
    avg = len(lst) // k
    remainder = len(lst) % k
    result = []
    start = 0
    for i in range(k):
        end = start + avg + (1 if i < remainder else 0)
        result.append(lst[start:end])
        start = end
    return result

def find_path(network,start,target):
    """
    Finds the shortest path between two nodes in the network using Breadth-First Search.

    Args:
        network (Network): The tensor network object.
        start_node (str): The name of the starting node.
        target_node (str): The name of the target node.

    Returns:
        list: A list of node names representing the path, or None if no path exists.
    """

    queue = deque([(start, [start])])

    while queue:
        node, path = queue.popleft()

        if node == target:
            return path  

        for neighbor in network.nodes[node].neighbours:
            if neighbor not in path:  
                queue.append((neighbor, path + [neighbor]))

    return None 

def pre_order_traversal(network):
    """
    Performs a pre-order traversal of a tree-like tensor network, starting from the root.
    Assumes a specific structure where the first neighbor is the parent.

    Args:
        network (Network): The tensor network to traverse.

    Returns:
        list: A list of node names in pre-order.
    """
    order = []
    _,root =  list(network.nodes.items())[0]
    def dfs(node):
        if node is None:
            return
        order.append(node.name)
        if node.name[3:] == "00":
            for child in node.neighbours:
                dfs(network.nodes[child])
        else:
            for child in node.neighbours[1:]:
                dfs(network.nodes[child])
    dfs(root)
    return order

def all_paths(network):
    """
    Generates all sequential paths needed for one half of a DMRG sweep.

    Args:
        network (Network): The tensor network.

    Returns:
        list: A list of paths, where each path is a list of node names.
    """
    path = []
    
    L = pre_order_traversal(network)
    # print(L)
    for i in range(len(L)-1):
        start = L[i]
        target = L[i+1]
        path.append(find_path(network,start,target))
    return path

def qr_series(network,path):
    """
    Performs a series of QR decompositions along a path to re-orthogonalize
    the tensor network, moving the center of orthogonality.

    Args:
        network (Network): The tensor network to modify.
        path (list): The path of nodes along which to perform the QR series.
    """
    tensor = []
    for i in range(len(path)-1):

        name1 = path[i]
        node1 = network.nodes[name1]
        name2 = path[i+1]
        node2 = network.nodes[name2]

        common_element = list(set(node1.subscript) & set(node2.subscript))
        common = common_element[0]

        idx1 = node1.subscript.index(common)
        idx2 = node2.subscript.index(common)

        A = node1.tensor
        A = np.moveaxis(A,idx1,-1)
        shape = A.shape
        A = A.reshape(-1,A.shape[-1])

        Q,R = np.linalg.qr(A)

        Q = Q.reshape(shape)
        Q = np.moveaxis(Q,-1,idx1)



        # ein_in = node1.subscript.copy()
        # ein_in[idx1] = 'a'
        # ein_str = ''.join(node1.subscript) +","+''.join(ein_in) +f"->{common}a"

        
        # I = oe.contract(ein_str, np.conjugate(Q),Q)

        # assert (np.allclose(np.identity(node1.dim[idx1]),I,1e-12)) == True, "Q not isometry!!"
        

        # A_ = oe.contract(f"{common}a," +''.join(node1.subscript)+ "->" + ''.join(ein_in),R,Q)
 
        # assert (np.allclose(A_,node1.tensor,1e-14)) == True, "R wrong!!"

        ein_out = node2.subscript.copy()
        for c in range(5):
            com =  oe.get_symbol(c) 
            if com not in ein_out:
                break
            if c==5-1:
                assert 1!=0, "Increase c in qr_series"
        ein_out[idx2] = com


        M = oe.contract(f"{com}{common}," +''.join(node2.subscript)+ "->" + ''.join(ein_out),R,node2.tensor)

        network.replace_tensor(name1,Q)
        network.replace_tensor(name2,M)
    
def sweep(ns,full_network,bra):
    """
    Core iterative optimization algorithm (a DMRG-like sweep). It finds the
    best tensor network approximation (`bra`) for the state produced by the `full_network`.

    Args:
        num_sweeps (int): The number of full forward and backward sweeps. Must be even.
        full_network (Network): The complete network (ket, circuit, bra) for calculating fidelity.
        bra_network (Network): The variational tensor network (`bra`) to be optimized.

    Returns:
        np.ndarray: An array containing the fidelity at each step of the sweep.
    """
    m = -1
    path = all_paths(bra)
    partial_fidelity = np.zeros((ns,bra.number_nodes))
    L = pre_order_traversal(bra)
    for i in range(ns):
        # L = list(bra.nodes.items())
        f = 0
        if i%2 == 0:
            m +=1
            counter = 1
            p = path
        else:
            m = -1
            counter = -1
            p = path[::-1]
        for k in range(len(L)):
            # j,v = L[m]
            j = L[m]
            # print(v.dim)
            F = full_network.contract_all_but_one(j)
            f = oe.contract("...,...->", F, np.conjugate(F)) 
            A = np.conjugate(F)/np.sqrt(f)
            bra.replace_tensor(j,A)
            full_network.replace_tensor(j,A)
            if k!=len(L)-1:
                p_ = p[k][::counter]
                assert j== p_[0], "Wrong path"
                qr_series(bra,p_)
                for names in p_:
                    full_network.replace_tensor(names,bra.nodes[names].tensor)
            partial_fidelity[i,m] = f.real
            # print(partial_fidelity)
            # if i==0:
            #     assert np.round(partial_fidelity[i,m],4) >= np.round(partial_fidelity[i,m-1],4), print(partial_fidelity[i,m],partial_fidelity[i,m-1])
            m +=counter
    return partial_fidelity

def DMRG(compression_steps,depth, no_sweeps,no_qubits,D,network_structure,full_edge_list, network_type,circuit_type,run):
    """
    Orchestrates a full quantum circuit simulation using partitioned
    circuits and DMRG-style compression.

    Args:
        compression_steps (int): How many partitions to divide the circuit into at each depth.
        depth (int): The number of layers in the quantum algorithm (e.g., p in QAOA).
        no_sweeps (int): The number of sweeps for the `sweep` optimizer.
        no_qubits (int): The total number of qubits.
        D (list): A list of bond dimensions for the tensor network.
        network_structure (list): The hierarchical structure for a Tree Tensor Network.
        full_edge_list (list): The list of all edges defining two-qubit gate locations.
        network_type (str): The type of tensor network ('tree' or 'mps').
        circuit_type (str): The type of circuit ('qaoa' or 'random').
        run (int): The current run number, used for seeding random number generators.

    Returns:
        float: The final cumulative fidelity of the simulation.
    """
    assert no_sweeps%2 ==0, "Number of sweeps must be even"

    print(f" Total Compression Step = {compression_steps*depth} and network = {network_type} and nodes: {network_structure}" )
    
    Fid = []
    partial_edge_list = partition_into_k_parts(full_edge_list, compression_steps)
 
    for d in range(depth):
        assert circuit_type == "qaoa" or circuit_type == "random", "Wrong circuit type"
        single_qubit_ket = np.identity(2)
        if circuit_type == "random":
            two_qubit_gate = None
            single_qubit_bra = np.identity(2)
        else: 
            
            if d==0:
                single_qubit_ket =  hadamard_gate()
            np.random.seed(d + run*10)
            two_qubit_gate = ZZ_QAOA((np.random.uniform(0, 2*np.pi)))
            if compression_steps == 1:
                np.random.seed(d+1+run*10)
                single_qubit_bra = X_QAOA((np.random.uniform(0, np.pi)))
            else:
                single_qubit_bra =np.identity(2)
                
        
        if network_type == "tree":
            assert no_qubits == network_structure[-1], "Number of qubits mismatch"
            if d==0:
                ket = Tree(D,"Ket",0,"zero",network_structure)
                bra = Tree(D,"Bra",ket.rank_all+1,"random",network_structure)
            else:
                for (ket_name, ket_node), (bra_name, bra_node) in zip(ket.nodes.items(), bra.nodes.items()):
                    assert bra_name[3:] == ket_name[3:], "Wrong ket/bra"
                    ket.replace_tensor(ket_name, np.conjugate(bra_node.tensor))
            circ = circuit_from_edge_list(bra.rank_all+1,partial_edge_list[0],no_qubits,single_qubit_ket,single_qubit_bra,circuit_type,two_qubit_gate)
            print("Number of 2 qubit gates:",len(partial_edge_list[0]))
            N = full_network_from_edge_list(ket,circ,bra,no_qubits)
            print("Memory = ", bra.memory_footprint())
            start_time = time.time()
            f = sweep(no_sweeps,N,bra)
            F_ = np.max(f[-1,:])
            Fid.append(F_)

        elif network_type == 'mps':

            assert no_qubits == len(D)+1, "Number of qubits mismatch"
            if d == 0:
                ket = MPS(D,"Ket",0,"zero")
                bra = MPS(D,"Bra",ket.rank_all+1,"random")
            else:
                for (ket_name, ket_node), (bra_name, bra_node) in zip(ket.nodes.items(), bra.nodes.items()):
                    assert bra_name[3:] == ket_name[3:], "Wrong ket/bra"
                    ket.replace_tensor(ket_name, np.conjugate(bra_node.tensor))
            circ = circuit_from_edge_list(bra.rank_all+1,partial_edge_list[0],no_qubits,single_qubit_ket,single_qubit_bra,circuit_type,two_qubit_gate)

            print("Number of 2 qubit gates:",len(partial_edge_list[0]))
            N = full_network_from_edge_list(ket,circ,bra,no_qubits)
            print("Memory = ", bra.memory_footprint())
            start_time = time.time()
            f = sweep(no_sweeps,N,bra)
            F_ = np.max(f[-1,:])
            Fid.append(F_)

        print(f"Compression step = {1} and Depth = {d+1}; Fidelity = {np.cumprod(np.array(Fid))[-1]}; sweep time = {time.time() - start_time}")

        for i in range(compression_steps-1):
            if circuit_type == "random":
                two_qubit_gate = None
                single_qubit_ket = np.identity(2)
                single_qubit_bra = np.identity(2)
            else:
                single_qubit_ket =  np.identity(2)
                if i == compression_steps-2:
                    np.random.seed(d+1+run*10)
                    single_qubit_bra = X_QAOA((np.random.uniform(0, np.pi)))
                else:
                    single_qubit_bra =np.identity(2)
            del circ
            del N
            gc.collect()
            
            for (ket_name, ket_node), (bra_name, bra_node) in zip(ket.nodes.items(), bra.nodes.items()):
                assert bra_name[3:] == ket_name[3:], "Wrong ket/bra"
                ket.replace_tensor(ket_name, np.conjugate(bra_node.tensor))

            circ = circuit_from_edge_list(bra.rank_all+1,partial_edge_list[i+1],no_qubits,single_qubit_ket,single_qubit_bra,circuit_type,two_qubit_gate)
            N = full_network_from_edge_list(ket,circ,bra,no_qubits)
            
            print("Number of 2 qubit gates:",len(partial_edge_list[i+1]))

            start_time = time.time()
            f = sweep(no_sweeps,N,bra)
            F_ = np.max(f[-1,:])
            Fid.append(F_)

            print(f"Compression step = {i+2}  and Depth = {d+1}; Fidelity = {np.cumprod(np.array(Fid))[-1]}; sweep time = {time.time() - start_time}")
        
        del circ
        del N
        gc.collect()


    return np.cumprod(np.array(Fid))[-1]
