import numpy as np
from data_loader import load_qap_instance, load_qap_solution
from ao_algorithm import ArtemisininOptimizer


def main():
    # 1. Define paths
    instance_path = 'Scenarios\Christofides\chr12a.dat'
    solution_path = 'Scenarios\Christofides\solution\chr12a.sln'

    # 2. Load the data using your functions
    # Note: Ensure load_qap_instance and load_qap_solution are imported correctly
    n, matrix_a, matrix_b = load_qap_instance(instance_path)
    n_sol, optimal_score, optimal_permutation = load_qap_solution(solution_path)

    print(f"--- Testing Instance: {instance_path} ---")

    if n != n_sol:
        print("Warning: The instance size and solution size do not match!")
        return
    
    print(f"Instance Size (n): {n}")
    print(f"Optimal Score from QAPLIB: {optimal_score}")

    # 3. Initialize the Optimizer
    # You can tweak pop_size and max_f based on the complexity of the instance
    optimizer = ArtemisininOptimizer(
        n_dim=n, 
        flow_matrix=matrix_a, 
        dist_matrix=matrix_b, 
        pop_size=50, 
        max_f=20000
    )

    # 4. Run the optimization
    best_p, best_score = optimizer.optimize()

    # 5. Final Results & Comparison
    print("\n--- Optimization Finished ---")
    print(f"Best Score Found: {best_score}")
    print(f"Best Permutation: {best_p}")

    # Calculate Error Gap (%)
    gap = ((best_score - optimal_score) / optimal_score) * 100
    print(f"Relative Error Gap: {gap:.2f}%")

    if best_score <= optimal_score:
        print("Success! You found the optimal solution.")
    else:
        print(f"The algorithm was {gap:.2f}% away from the optimum.")


if __name__ == "__main__":
    main()