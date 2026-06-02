import matplotlib.pyplot as plt
import numpy as np
from ao_algorithm import ArtemisininOptimizer
from wrao_algorithm import WeightedArtemisininOptimizer
import time
from ao_algorithm_pmx import PMXOptimizer

def benchmark(n_runs, opt_val, n=None, matrix_a=None, matrix_b=None, pop_size=200, max_f=1000000, version="AO", portions=0.1, injection_period=10, injection_rate=0.3):
    all_scores = []
    all_gaps = []
    all_perms = []
    all_histories = []
    all_durations = []
    all_populations = []  # <--- Tutaj będą zapisywane migawki populacji (start, mid_1, mid_2, end) dla każdego runu

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
        elif version == "PMX":
            optimizer = PMXOptimizer(
                n_dim=n,
                flow_matrix=matrix_a,
                dist_matrix=matrix_b,
                pop_size=pop_size,
                max_f=max_f,
                optimum=opt_val,
                injection_period=injection_period,
                injection_rate=injection_rate
            )

        # Odbieramy 4 parametry z metody optimize: dodany pop_snapshots
        best_p, best_score, best_cost_history, pop_snapshots = optimizer.optimize()

        gap = ((best_score - opt_val) / opt_val) * 100 if opt_val != 0 else 0.0

        end_time = time.perf_counter()
        duration = end_time - start_time

        all_scores.append(best_score)
        all_gaps.append(gap)
        all_perms.append(best_p)
        all_histories.append(best_cost_history)
        all_durations.append(duration)
        all_populations.append(pop_snapshots)  # <--- Zapisujemy strukturę populacji

        print(f"Best Score: {best_score}")
        print(f"GAP: {gap:.4f}%")
        print(f"Run Duration: {duration:.2f} seconds")

        if best_score < global_best_score:
            global_best_score = best_score
            global_best_perm = best_p
            global_best_history = best_cost_history

    # ================================
    # FINAL STATISTICS
    # ================================
    mean_gap = np.mean(all_gaps)
    min_gap = min(all_gaps)
    max_gap = max(all_gaps)
    std_gap = np.std(all_gaps)
    success_count = sum(1 for s in all_scores if s <= opt_val)

    print("\n\n==============================================")
    print("FINAL BENCHMARK STATISTICS")
    print("==============================================")
    print(f"BEST SCORE OVERALL : {global_best_score}")
    print(f"BEST GAP OVERALL   : {min_gap:.4f}%")
    print(f"MEAN GAP           : {mean_gap:.4f}%")
    print(f"WORST GAP          : {max_gap:.4f}%")
    print(f"STD GAP            : {std_gap:.4f}%")
    print(f"OPTIMUM HITS       : {success_count}/{n_runs}")
    print(f"\nBest Permutation Found Overall:\n{global_best_perm}")

    # ================================
    # GAP BAR CHART
    # ================================
    # plt.figure(figsize=(10,6))
    # bars = plt.bar(range(1, n_runs+1), all_gaps)
    # plt.xticks(range(1, n_runs+1))
    # plt.xlim(0, n_runs+1)
    
    # plt.ylim(0, max(all_gaps) + 0.5 if max(all_gaps) > 0 else 1)

    # plt.xlabel("Run Number")
    # plt.ylabel("Relative GAP [%]")
    # plt.title(f"GAP for different runs, n = {n}, pop_size={pop_size}, max_f={max_f}")
    # plt.grid(False)

    # for bar in bars:
    #     height = bar.get_height()
    #     plt.text(
    #         bar.get_x() + bar.get_width()/2,
    #         height,
    #         f'{height:.2f}',
    #         ha='center',
    #         va='bottom',
    #         fontsize=8
    #     )
    #plt.show()

    # ================================
    # RESULTS DICTIONARY TO RETURN
    # ================================
    results = {
        # Zgrupowane statystyki końcowe
        "best_score_overall": global_best_score,
        "best_gap_overall": min_gap,
        "mean_gap": mean_gap,
        "worst_gap": max_gap,
        "std_gap": std_gap,
        "optimum_hits": success_count,
        "best_permutation_overall": global_best_perm,
        "best_history_overall": global_best_history,
        
        # Szczegółowe dane z każdego uruchomienia (run-by-run)
        "all_scores": all_scores,
        "all_gaps": all_gaps,
        "all_permutations": all_perms,
        "all_histories": all_histories,
        "all_durations": all_durations,
        
        # TUTAJ MASZ DOSTĘP DO POPULACJI Z KAŻDEGO URUCHOMIENIA
        "all_populations": all_populations, 
        
        # Metadane i użyte parametry wejściowe
        "metadata": {
            "version": version,
            "n_runs": n_runs,
            "n_dim": n,
            "pop_size": pop_size,
            "max_f": max_f,
            "opt_val": opt_val,
            "portions": portions,
            "injection_period": injection_period,
            "injection_rate": injection_rate
        }
    }

    return results