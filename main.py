# main.py

from src.simulation import DMRG
from src.utils.TN_gen import D_tree, D_mps
from src.utils.graph_gen import * # graph types
from tqdm import tqdm
import numpy as np

def main():
    """Configures and runs the simulation experiment."""


    no_qubits = 27                  # Total number of qubits
    qubit_graph = nearest_neighbour_edge_list(no_qubits)  # Qubit connectivity graph
    circuit_type = "random"         # Circuit type: "random" or "qaoa"
    Depth = 5                       # Circuit depth (number of layers)

    network_type = "tree"           # Network type: "tree" (TTN) or "mps"
    qubit_order = "Blind"           # TTN ordering: "Naive" or "Blind"
    network_structure = [1, 3, 9, 27]  # TTN structure (branching hierarchy)

    compression_steps = 2           # Compression steps per depth
    no_sweeps = 2                   # Number of DMRG sweeps
    Dmax = [4, 8, 12]               # List of bond dimensions to test
    runs = 2                        # Number of independent runs per setup

    

    Fidelity_list = []

    print("Starting simulations...")

    for d_max in tqdm(Dmax, desc="D values"):

        if network_type == "tree":
            D = D_tree(network_structure,d_max)
        else:
            D = D_mps(no_qubits,d_max)
        Fidelity_run = 0
        for i in tqdm(range(runs), desc="Runs", leave=False):
            edge_list = edge_extract(qubit_graph,qubit_order)
            fidelity_results = DMRG(
                compression_steps=compression_steps,
                depth=Depth,
                no_sweeps=no_sweeps,
                no_qubits=no_qubits,
                D=D,
                network_structure=network_structure,
                full_edge_list=edge_list,
                network_type=network_type,
                circuit_type=circuit_type,
                run=i
            )
            print(f"Run {i+1} with Dmax={d_max} complete. Final Fidelity: {fidelity_results}")
            Fidelity_run += fidelity_results
        Fidelity_list.append(Fidelity_run/runs)

    print("Bond Dimension, Fidelity")
    data = np.column_stack(( Dmax, Fidelity_list))
    print(data)

if __name__ == "__main__":
    main()
