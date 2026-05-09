import matplotlib.pyplot as plt
import numpy as np
from ao_algorithm import ArtemisininOptimizer
from wrao_algorithm import WeightedArtemisininOptimizer
import time


def benchmark(n_runs, opt_val, n=None, matrix_a=None, matrix_b=None, pop_size=200, max_f=1000000, version="AO", portions=0.1):
    all_scores = []
    all_gaps = []
    all_perms = []
    all_histories = []

    global_best_score = float('inf')
    global_best_perm = None
    global_best_history = None

    # ================================
    # MULTI-RUN LOOP
    # ================================
    print(f"\nStarting benchmark with {n_runs} runs, version: {version}")

    for run in range(n_runs):
        start_time = time.perf_counter()
        print(f"\n========== RUN {run+1}/{n_runs} ==========")

        if version == "AO":
            optimizer = ArtemisininOptimizer(
                n_dim=n,
                flow_matrix=matrix_a,
                dist_matrix=matrix_b,
                pop_size=pop_size,
                max_f=max_f,
                optimum=opt_val
            )

        elif version == "WRAO":
            optimizer = WeightedArtemisininOptimizer(
                n_dim=n,
                flow_matrix=matrix_a,
                dist_matrix=matrix_b,
                pop_size=pop_size,
                max_f=max_f,
                optimum=opt_val,
                ranking_portion=portions
            )

        best_p, best_score, best_cost_history = optimizer.optimize()

        gap = ((best_score - opt_val) / opt_val) * 100

        all_scores.append(best_score)
        all_gaps.append(gap)
        all_perms.append(best_p)
        all_histories.append(best_cost_history)

        print(f"Best Score: {best_score}")
        print(f"GAP: {gap:.4f}%")

        if best_score < global_best_score:
            global_best_score = best_score
            global_best_perm = best_p
            global_best_history = best_cost_history

        end_time = time.perf_counter()
        duration = end_time - start_time
        print(f"Run Duration: {duration:.2f} seconds")

    # ================================
    # FINAL STATISTICS
    # ================================
    print("\n\n==============================================")
    print("FINAL BENCHMARK STATISTICS")
    print("==============================================")

    # print("\nRun-wise GAP values:")
    # for i, gap in enumerate(all_gaps):
    #     print(f"Run {i+1:02d}: {gap:.4f}%")

    print("\n----------------------------------------------")
    print(f"BEST SCORE OVERALL : {global_best_score}")
    print(f"BEST GAP OVERALL   : {min(all_gaps):.4f}%")
    print(f"MEAN GAP           : {np.mean(all_gaps):.4f}%")
    print(f"WORST GAP          : {max(all_gaps):.4f}%")
    print(f"STD GAP            : {np.std(all_gaps):.4f}%")

    success_count = sum(1 for s in all_scores if s <= opt_val)
    print(f"OPTIMUM HITS       : {success_count}/{n_runs}")

    print(f"\nBest Permutation Found Overall:\n{global_best_perm}")

    # ================================
    # GAP BAR CHART
    # ================================
    plt.figure(figsize=(10,6))
    bars = plt.bar(range(1, n_runs+1), all_gaps)
    plt.xticks(range(1, n_runs+1))
    plt.xlim(0, n_runs+1)
    plt.ylim(0, max(all_gaps) + 0.5)

    plt.xlabel("Run Number")
    plt.ylabel("Relative GAP [%]")
    plt.title(f"GAP for different runs, n = {n}, pop_size={pop_size}, max_f={max_f}")
    plt.grid(False)

    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width()/2,
            height,
            f'{height:.2f}',
            ha='center',
            va='bottom',
            fontsize=8
        )
    plt.show()

    # ================================
    # BEST CURVE
    # ================================
    plt.figure(figsize=(10,6))
    plt.plot(global_best_history)
    plt.xlabel("Global Improvements")
    plt.ylabel("Best Cost")
    plt.title(f"Best Run, n = {n}")
    plt.grid(True)
    plt.show()
