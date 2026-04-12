import numpy as np
from data_loader import load_instance, load_solution
from ao_algorithm import ArtemisininOptimizer
import platform

system = platform.system()



def main():
    system = platform.system()

    instance_path = r'Scenarios\Christofides\chr12a.dat'
    solution_path = r'Scenarios\Christofides\solution\chr12a.sln'
    
    if system == "Linux":
        instance_path = instance_path.replace("\\", "/")
        solution_path = solution_path.replace("\\", "/")


    n, matrix_a, matrix_b = load_instance(instance_path)
    n_sol, optimal_score_raw, optimal_permutation = load_solution(solution_path)

    print(f"--- Testing Instance: {instance_path} ---")

    if n != n_sol:
        print("Warning: The instance size and solution size do not match!")
        return
    
    try:
        opt_val = float(str(optimal_score_raw).strip())
    except ValueError:
        print(f"Error: Could not convert {optimal_score_raw} to float.")
        return

    print(f"Instance Size (n): {n}")
    print(f"Optimal Score from QAPLIB: {opt_val}")

    # 3. Inicjalizacja Optymalizatora
    optimizer = ArtemisininOptimizer(
        n_dim=n, 
        flow_matrix=matrix_a, 
        dist_matrix=matrix_b, 
        pop_size=500, 
        max_f=200000
    )

    best_p, best_score = optimizer.optimize()

    print("\n--- Optimization Finished ---")
    print(f"Best Score Found: {best_score}")
    print(f"Best Permutation: {best_p}")

    gap = ((best_score - opt_val) / opt_val) * 100
    print(f"Relative Error Gap: {gap:.2f}%")

    if best_score <= opt_val:
        print("Success! You found the optimal solution.")
    else:
        print(f"The algorithm was {gap:.2f}% away from the optimum.")

if __name__ == "__main__":
    main()