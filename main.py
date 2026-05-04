import numpy as np
from benchmark import benchmark
from data_loader import load_instance, load_solution
from ao_algorithm import ArtemisininOptimizer
import platform
import matplotlib.pyplot as plt

system = platform.system()



def main():
    system = platform.system()

    instance_path = r'Scenarios\Christofides\chr25a.dat'
    solution_path = r'Scenarios\Christofides\solution\chr25a.sln'
    
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
    print(f"Optimal Permutation from QAPLIB: {optimal_permutation}")


    # ================================
    # BENCHMARK 
    # ================================

    n_runs = 50
    pop_size = 1500
    max_f = 2500000

    benchmark(n_runs, opt_val, n, matrix_a, matrix_b, pop_size=pop_size, max_f=max_f)


if __name__ == "__main__":
    main()