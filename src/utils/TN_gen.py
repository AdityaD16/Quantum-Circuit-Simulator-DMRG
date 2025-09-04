# src/utils/circuit_gen.py

import numpy as np
import copy
from src.network import Network
from src.node import Node
from src.utils.quantum_gates import random_isometry, random_gate

def D_tree(network_structure, D_max): #Returns the bond dimension list for each layer of tree
    return [min(2**x, D_max) for x in network_structure][:-1]

def D_mps(no_qubits,D_max): # Returns the bond dimension list for MPS
    Dmps_ = [min(D_max, 2**(i+1)) for i in range(no_qubits // 2)]
    if no_qubits % 2 == 0:
        Dmps = Dmps_ + Dmps_[::-1][1:]
    else:
        Dmps = Dmps_ + Dmps_[::-1]
    return Dmps

def Tree(D,node_name,TR,initial,no_nodes):
    N = Network()
    N.rank_all = TR
    N.Layers = len(D)
    assert node_name == "Ket" or node_name == "Bra", "Wrong Name"
    if initial == "zero":
        for i in range(len(D)):
            for j in range(no_nodes[i]):
                if i==0:
                    rank = no_nodes[i+1]
                    tensor = np.zeros(([D[-1-i]]*rank))
                    tensor.flat[0] = 1
                    A = Node(node_name+f"{i}{j}",rank,[D[-1-i]]*rank,tensor)
                    N.add_node(A)
                else:

                    rank = no_nodes[i+1]//no_nodes[i] + 1
                    
                    pi = i-1
                    pj = j//(no_nodes[i]//no_nodes[i-1])
                    if pi == 0:
                        idx = j 
                    else:
                        idx = j+1 - (no_nodes[i]//no_nodes[i-1])*pj
                    dimensions = [D[-i]] + [D[-i-1]]*(rank-1)
                    tensor = np.zeros((dimensions))
                    tensor.flat[0] = 1
                    A = Node(node_name+f"{i}{j}",rank,dimensions,tensor)
                    N.add_node(A)
                    # print(pi,pj,idx)
                    # N.print_network()
                    N.add_to_node(node_name+f"{pi}{pj}",node_name+f"{i}{j}",idx,0)
                    
    if initial == "random":
        for i in range(len(D)):
            for j in range(no_nodes[i]):
                if i==0:
                    rank = no_nodes[i+1]
                    dimensions = D[-i-1]**rank
                    A = Node(node_name+f"{i}{j}",rank,[D[-1-i]]*rank,(np.random.randn(dimensions) + 1j * np.random.randn(dimensions)).reshape([D[-i-1]]*rank))
                    N.add_node(A)
                else:
                    rank = no_nodes[i+1]//no_nodes[i] + 1
                    pi = i-1
                    pj = j//(no_nodes[i]//no_nodes[i-1])
                    if pi == 0:
                        idx = j 
                    else:
                        idx = j +1 - (no_nodes[i]//no_nodes[i-1])*pj
                    dimensions = [D[-i]] + [D[-i-1]]*(rank-1)
                    tensor = random_isometry(D[-i],D[-i-1]**(rank-1))
                    # print(D[-i-1]*D[-i-1])
                    A = Node(node_name+f"{i}{j}",rank,dimensions,tensor.reshape(dimensions))
                    N.add_node(A)
                    N.add_to_node(node_name+f"{pi}{pj}",node_name+f"{i}{j}",idx,0)
    return N

def MPS(D,node_name,TR,initial):

    N = Network()
    N.rank_all = TR
    N.Layers = 1
    assert node_name == "Ket" or node_name == "Bra", "Wrong Name"
    if initial == "random":
        for i in range(len(D)+1):
            if i == 0:
                A = Node(node_name+f"{i}",2,[D[i],2],np.random.randn(D[i],2) + 1j * np.random.randn(D[i],2))
                N.add_node(A)
            elif i == len(D):
                tensor = random_isometry(D[i-1],2)
                A = Node(node_name+f"{i}",2,[D[i-1],2],tensor.reshape(D[i-1],2))
                N.add_node(A)
                idx = 0 if (i-1 == 0) else 2
                N.add_to_node(node_name+f"{i-1}",node_name+f"{i}",idx,0)
            else:
                tensor = random_isometry(D[i-1],2*D[i])
                # print(tensor.shape)
                A = Node(node_name+f"{i}",3,[D[i-1],2,D[i]],tensor.reshape(D[i-1],2,D[i]))
                N.add_node(A)
                idx = 0 if (i-1 == 0) else 2
                N.add_to_node(node_name+f"{i-1}",node_name+f"{i}",idx,0)
                # N.print_network()    
    
    if initial == "zero":
        for i in range(len(D)+1):
            zero_2 = np.zeros((1,2))
            zero_2[0,0] = 1
            zero_3 = np.zeros((1,2,1))
            zero_3[0,0,0] = 1
            if i == 0:
                A = Node(node_name+f"{i}",2,[D[i],2],zero_2)
                N.add_node(A)
            elif i == len(D):
                A = Node(node_name+f"{i}",2,[D[i-1],2],zero_2)
                N.add_node(A)
                idx = 0 if (i-1 == 0) else 2
                N.add_to_node(node_name+f"{i-1}",node_name+f"{i}",idx,0)
            else:
                A = Node(node_name+f"{i}",3,[D[i-1],2,D[i]],zero_3)
                N.add_node(A)
                idx = 0 if (i-1 == 0) else 2
                N.add_to_node(node_name+f"{i-1}",node_name+f"{i}",idx,0)    
    return N

def circuit_from_edge_list(TR,edge_list,no_qubits,single_qubit_gate_ket,single_qubit_gate_bra,circuit_type,two_qubit_gate=None): #Assumes balanced trees

    new_network = Network()    
    new_network.rank_all = TR
    current_gates = []

    for i in range(no_qubits):
        A = Node(f"1G_K{i}",2,[2,2],single_qubit_gate_ket)
        current_gates.append(f"1G_K{i}")
        new_network.add_node(A)  
    
    assert circuit_type == "random" or circuit_type == "qaoa", "Wrong circuit type"
    
    for i in edge_list:
        x,y = i
        
        # if x<y:
        #     name_2q = f"2G_{x}_{y}"
        # else:
        #     name_2q = f"2G_{y}_{x}"
        name_2q = f"2G_{x}_{y}"
        # print(i,current_gates[x],name_2q)
        if circuit_type == "random":
            A = Node(name_2q,4,[2,2,2,2],random_gate())
        else:
            assert two_qubit_gate is not None, "Wrong two qubit gate for QAOA"
            A = Node(name_2q,4,[2,2,2,2],two_qubit_gate)
        new_network.add_node(A)  

        if current_gates[x][0] == "1":
            new_network.add_to_node(current_gates[x],name_2q,1,0)
            current_gates[x] = name_2q
        else:
            indices = [dash for dash, val in enumerate(current_gates[x]) if val == '_']
            if current_gates[x][indices[0]+1:indices[1]] == str(x):
                new_network.add_to_node(current_gates[x],name_2q,2,0)
                current_gates[x] = name_2q
            else:
                new_network.add_to_node(current_gates[x],name_2q,3,0)
                current_gates[x] = name_2q

        if current_gates[y][0] == "1":
            new_network.add_to_node(current_gates[y],name_2q,1,1)
            current_gates[y] = name_2q
        else:
            indices = [dash for dash, val in enumerate(current_gates[y]) if val == '_']
            if current_gates[y][indices[0]+1:indices[1]] == str(y):
                new_network.add_to_node(current_gates[y],name_2q,2,1)
                current_gates[y] = name_2q
            else:
                new_network.add_to_node(current_gates[y],name_2q,3,1)
                current_gates[y] = name_2q
        


    for i in range(no_qubits):
        A = Node(f"1G_B{i}",2,[2,2],single_qubit_gate_bra)
        new_network.add_node(A)  
        node = new_network.nodes[current_gates[i]]
        # idx = node.subscript.index(node.open_legs[0])
        if current_gates[i][0] =="1":
            new_network.add_to_node(f"1G_B{i}",current_gates[i],0,1)
        else:
            indices = [dash for dash, val in enumerate(current_gates[i]) if val == '_']
            if str(i) == current_gates[i][indices[0]+1:indices[1]]:
                new_network.add_to_node(f"1G_B{i}",current_gates[i],0,2)
            else:   
                new_network.add_to_node(f"1G_B{i}",current_gates[i],0,3)
    return new_network

def full_network_from_edge_list(Ket,Circuit,Bra,no_qubits):

    new_network = Network()
    for name, node in Ket.nodes.items():
        new_network.nodes[name] = copy.deepcopy(node)

    for name, node in Circuit.nodes.items():
        new_network.nodes[name] = copy.deepcopy(node)

    for name, node in Bra.nodes.items():
        new_network.nodes[name] = copy.deepcopy(node)
    
    # print(new_network.leaf_nodes)

    new_network.leaf_nodes = copy.deepcopy(Ket.leaf_nodes) 
    # print(new_network.leaf_nodes)
    # new_network.print_network()
    for i in range(no_qubits):
        node = new_network.nodes[new_network.leaf_nodes[0]]
        idx = node.subscript.index(node.open_legs[0])
        new_network.add_to_node(f"1G_K{i}",new_network.leaf_nodes[0],0,idx) # Leaf node gets removed
    

    new_network.leaf_nodes = copy.deepcopy(Bra.leaf_nodes) 
    for i in range(no_qubits):
        node = new_network.nodes[new_network.leaf_nodes[0]]
        idx = node.subscript.index(node.open_legs[0])
        new_network.add_to_node(f"1G_B{i}",new_network.leaf_nodes[0],1,idx)
    
    return new_network

def ket_network_from_edge_list(Ket,Circuit,no_qubits):

    new_network = Network()
    for name, node in Ket.nodes.items():
        new_network.nodes[name] = copy.deepcopy(node)

    for name, node in Circuit.nodes.items():
        new_network.nodes[name] = copy.deepcopy(node)


    new_network.leaf_nodes = copy.deepcopy(Ket.leaf_nodes) 
    for i in range(no_qubits):
        node = new_network.nodes[new_network.leaf_nodes[0]]
        idx = node.subscript.index(node.open_legs[0])
        new_network.add_to_node(f"1G_K{i}",new_network.leaf_nodes[0],0,idx)

    
    return new_network