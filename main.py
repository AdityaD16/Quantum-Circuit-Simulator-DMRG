# main.py

from src.simulation import DMRG
from src.utils.TN_gen import D_tree, D_mps
from src.utils.graph_gen import * # graph types
from tqdm import tqdm
import numpy as np

def main():
    """Configures and runs the simulation experiment."""
    # Experiment configuration parameters
    Depth = 5 # Depth of the circuit
    compression_steps = 2 # Number of compression steps per depth
    no_sweeps = 2 # Number of DMRG sweeps
    Dmax = [10]
    no_qubits = 27
    network_structure = [1, 3,9,27]  # TTN Structure
    circuit_type = "random"
    TTN_type = "Blind"
    runs = 2
    network_type = "tree"

    Fidelity_list = []

    print("Starting simulations...")

    for d_max in tqdm(Dmax, desc="D values"):

        if network_type == "tree":
            D = D_tree(network_structure,d_max)
        else:
            D = D_mps(network_structure,d_max)
        Fidelity_run = 0
        for i in tqdm(range(runs), desc="Runs", leave=False):
            edge_list = edge_extract(nearest_neighbour_edge_list(no_qubits),TTN_type)
            
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
