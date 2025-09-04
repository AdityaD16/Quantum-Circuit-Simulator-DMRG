# src/utils/quantum_gates.py

import numpy as np
from scipy.linalg import expm

def hadamard_gate():
    """Returns the Hadamard gate as a 2x2 tensor."""
    return (1 / np.sqrt(2)) * np.array([[1, 1], [1, -1]])

def cnot_gate():
    """Returns the CNOT gate as a 2x2x2x2 tensor."""
    return np.array([[[[1, 0], [0, 0]], [[0, 0], [0, 1]]], [[[0, 0], [0, 1]], [[1, 0], [0, 0]]]])

def random_gate():
    """Returns a random 2-qubit unitary gate as a 2x2x2x2 tensor."""
    A = np.random.randn(4, 4) + 1j * np.random.randn(4, 4)
    Q, _ = np.linalg.qr(A)
    return Q.reshape(2, 2, 2, 2)

def random_isometry(rows, cols):
    """Returns a random isometry matrix reshaped as a tensor."""
    A = np.random.randn(rows, cols) + 1j * np.random.randn(rows, cols)
    Q, _ = np.linalg.qr(A)
    if cols > rows:
        Q_t, _ = np.linalg.qr(A.T)
        Q = Q_t.T
    return Q

def ZZ_QAOA(gamma):
    """Returns the ZZ interaction gate for QAOA with parameter gamma."""
    A = np.diag([np.exp(-0.5j * gamma), np.exp(0.5j * gamma), np.exp(0.5j * gamma), np.exp(-0.5j * gamma)])
    return A.reshape(2, 2, 2, 2)

def X_QAOA(beta):
    """Returns the X rotation gate for QAOA with parameter beta."""
    X_gate = np.array([[0, 1], [1, 0]])
    return expm(-0.5j * beta * X_gate)
