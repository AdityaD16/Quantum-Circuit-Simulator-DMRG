# src/network.py

import numpy as np
import opt_einsum as oe
import cotengra as ctg
import gc
import jax

# It's recommended to enable 64-bit precision for scientific computing
jax.config.update("jax_enable_x64", True)

from src.node import Node

class Network:
    """
    Manages a collection of nodes, representing a tensor network.
    Includes methods for adding nodes, contracting tensors, and managing connections.
    """
    rank_all = 0
    Layers = 0
    number_nodes = 0

    def __init__(self):
        self.nodes = {}  # Dictionary to store all nodes in the network
        self.leaf_nodes = []  # Leaf nodes with open physical legs
        self.network_structure = []

    def add_node(self, node: Node):
        if node.name in self.nodes:
            raise ValueError(f"Node {node.name} already exists in the network.")
        if not node.subscript:
            for i in range(self.rank_all, self.rank_all + node.rank):
                node.subscript.append(oe.get_symbol(i))
        if not node.open_legs:
            node.open_legs = node.subscript.copy()
        
        self.nodes[node.name] = node
        self.rank_all += node.rank
        self.number_nodes += 1
        self.leaf_nodes.append(node.name)

    def add_to_node(self, node1_name, node2_name, index1, index2):
        if node1_name not in self.nodes or node2_name not in self.nodes:
            raise ValueError("One or both nodes do not exist in the network.")
        node1 = self.nodes[node1_name]
        node2 = self.nodes[node2_name]

        if index1 >= len(node1.dim) or index2 >= len(node2.dim):
            raise IndexError("Index out of range for one of the nodes")
        
        assert node1.dim[index1] == node2.dim[index2], "Dimension mismatch between connected legs."

        node1.open_legs.remove(node1.subscript[index1])
        node2.open_legs.remove(node2.subscript[index2])
        node2.subscript[index2] = node1.subscript[index1]
        node1.neighbours.append(node2_name)
        node2.neighbours.append(node1_name)

        if not node1.open_legs and node1_name in self.leaf_nodes:
            self.leaf_nodes.remove(node1_name)
        if not node2.open_legs and node2_name in self.leaf_nodes:
            self.leaf_nodes.remove(node2_name)

    def contract_all(self):
        einsum_str_parts = []
        tensor_list = []
        open_legs_chars = []

        for v in self.nodes.values():
            einsum_str_parts.append(''.join(v.subscript))
            tensor_list.append(v.tensor)
            open_legs_chars.extend(v.open_legs)
        
        einsum_str = f"{','.join(einsum_str_parts)}->{''.join(sorted(list(set(open_legs_chars))))}"
        
        return oe.contract(einsum_str, *tensor_list, optimize='auto', memory_limit=16e9)

    def contract_all_but_one(self, node_name):
        opt = ctg.ReusableHyperOptimizer(
            methods=["greedy", "random-greedy"],
            max_repeats=16,
            progbar=False,
            parallel=True
        )
        
        einsum_str_parts = []
        tensor_list = []
        target_node = self.nodes[node_name]

        for name, v in self.nodes.items():
            if name != node_name:
                einsum_str_parts.append(''.join(v.subscript))
                tensor_list.append(v.tensor)
        
        einsum_str = f"{','.join(einsum_str_parts)}->{''.join(target_node.subscript)}"
        
        return oe.contract(einsum_str, *tensor_list, optimize=opt, memory_limit=128e9)

    def replace_tensor(self, node_name, tensor):
        node = self.nodes[node_name]
        assert list(np.shape(tensor)) == node.dim, "Dimensions of new tensor must match the node's dimensions."
        node.tensor = tensor

    def memory_footprint(self):
        mem = sum(np.prod(np.array(v.dim, dtype=object)) for v in self.nodes.values())
        return mem
        
    def print_network(self):
        for i, v in self.nodes.items():
            print(f"Node: {i}, Neighbours: {v.neighbours}, Dims: {v.dim}, Subscripts: {v.subscript}, Open Legs: {v.open_legs}")

