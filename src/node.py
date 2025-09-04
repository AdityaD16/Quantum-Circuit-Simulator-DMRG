# src/node.py

import numpy as np

class Node:
    """
    Represents a single tensor (a node) in a tensor network.
    """
    def __init__(self, name, rank, dim=None, tensor=None, subscript=None, open_legs=None):
        self.name = name
        self.rank = rank
        self.dim = dim if dim is not None else []
        self.tensor = tensor
        self.subscript = subscript if subscript is not None else []
        self.open_legs = open_legs if open_legs is not None else []
        self.neighbours = []
