# Quantum Ciruit SImulation with Tree Tensor Network and MPS using Density-Matrix Renormalization Group Algorithm 

This repository implements simulations of **quantum circuits** with **Tree Tensor Networks (TTNs)** and **Matrix Product States (MPS)** using the **Density Matrix Renormalization Group (DMRG)** algorithm. 

The code was utilized to obtain the results in the work: “Simulating Quantum Circuits with Tree Tensor Networks using Density-Matrix Renormalization Group Algorithm” ([arXiv:2504.16718](https://arxiv.org/abs/2504.16718)). 

## Installation
Install all the required dependencies with `pip install -r requirements.txt`.

## Simulation Parameters

 `main.py` simulates the circuit using TTN or MPS and outputs fidelity results for each bond dimension. The simulation is configured via parameters in `main.py`. Below is a description of each parameter:

| Parameter            | Description                                                                                      |
|----------------------|--------------------------------------------------------------------------------------------------|
| `no_qubits`          | Total number of qubits in the circuit. |
| `qubit_graph`        | Connectivity graph of qubits. e.g. nearest neighbour, 3-regular etc. (See `graph_gen.py`) |
| `circuit_type`       | Type of quantum circuit to simulate. Options include `"random"` for random unitary circuit, or `"qaoa"` for QAOA solving MaxCut problem. |
| `Depth`              | Circuit depth, i.e., the number of times the qubit graph is repeated to form the full circuit.|
| `network_type`       | Tensor network representation: `"tree"` for Tree Tensor Network (TTN) or `"mps"` for Matrix Product State (MPS) |
| `qubit_order`        | Qubit ordering strategy used in TTN construction. `"Naive"` or`"Blind"`(see Sec. III B [arXiv:2504.16718](https://arxiv.org/abs/2504.16718)). |
| `network_structure`  | Defines the hierarchical structure of the TTN, e.g., `[1, 3, 9, 27]` for a ternary tree, `[1, 2,4,8,16]` for a binary tree.  |
| `compression_steps`  | Number of tensor compression steps per circuit depth |
| `no_sweeps`          | Number of DMRG sweeps|
| `Dmax`               | List of max bond dimensions to use in the tensor network |
| `runs`               | Number of independent simulations to average over |


## Reference

Aditya Dubey, Zeki Zeybek, and Peter Schmelcher, “Simulating Quantum Circuits with Tree Tensor Networks using Density-Matrix Renormalization Group Algorithm,” [arXiv:2504.16718](https://arxiv.org/abs/2504.16718) (2025).
