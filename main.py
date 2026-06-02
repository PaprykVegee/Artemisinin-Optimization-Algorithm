import platform
import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time

from benchmark import benchmark
from data_loader import load_instance, load_solution

SCENARIOS_FOLDER = Path("Scenarios")
RESULT_FOLDER = Path("Result")

# Główny folder na wyniki
RESULT_FOLDER.mkdir(exist_ok=True)

def save_population_snapshots(file_path, version_name, instance_name, bench_data):
    """
    Pomocnicza funkcja do zapisu migawek populacji dla konkretnej metody do osobnego pliku.
    """
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(f"POPULATION SNAPSHOTS HISTORY FOR INSTANCE: {instance_name} | METHOD: {version_name}\n")
        f.write("=" * 80 + "\n\n")
        
        # Przechodzimy przez każdy run (od 0 do n_runs-1)
        for run_idx, run_pops in enumerate(bench_data["all_populations"]):
            f.write(f"=== RUN {run_idx + 1} ===\n")
            
            # Dla każdego runu wypisujemy 4 kluczowe momenty
            for moment in ["start", "mid_1", "mid_2", "end"]:
                f.write(f"  > Moment: {moment.upper()}\n")
                
                # Każdy agent w populacji staje się jedną linijką w pliku
                for agent_idx, permutation in enumerate(run_pops[moment]):
                    perm_str = " ".join(map(str, permutation))
                    f.write(f"    Agent {agent_idx+1:03d}: [{perm_str}]\n")
                
                f.write("\n")
            f.write("-" * 50 + "\n\n")

def main():
    # 1. Pobranie plików
    instances = list(SCENARIOS_FOLDER.rglob("*.dat"))
    solutions = list(SCENARIOS_FOLDER.rglob("*.sln"))

    # Zapisujemy ścieżki jako str, jeśli plik istnieje
    sol_dict = {sol.stem: str(sol) for sol in solutions}

    data = []
    for inst in instances:
        name = inst.stem
        sol_path = sol_dict.get(name, None)

        data.append({
            'name': name,
            'scen_path': str(inst),
            'sol_path': sol_path 
        })

    df_paths = pd.DataFrame(data)

    # 2. Pętla główna po instancjach
    for _, row in df_paths.iterrows():
        instance_path = row['scen_path']
        solution_path = row['sol_path']
        name = row['name']

        if solution_path is None:
            print(f"Warning: Missing solution file for instance {name}. Skipping...")
            continue

        n, matrix_a, matrix_b = load_instance(instance_path)
        n_sol, optimal_score_raw, optimal_permutation = load_solution(solution_path)

        print(f"\n=========================================")
        print(f"--- Testing Instance: {name} ---")
        print(f"=========================================")

        if n != n_sol:
            print("Warning: The instance size and solution size do not match! Skipping...")
            continue 
        
        try:
            opt_val = float(str(optimal_score_raw).strip())
        except ValueError:
            print(f"Error: Could not convert {optimal_score_raw} to float. Skipping...")
            continue

        print(f"Instance Size (n): {n}")
        print(f"Optimal Score from QAPLIB: {opt_val}")

        # ============================================================
        # TWORZENIE PODFOLDERU DLA INSTANCJI
        # ============================================================
        instance_result_folder = RESULT_FOLDER / name
        instance_result_folder.mkdir(exist_ok=True)

        # ================================
        # BENCHMARK CONFIGURATION
        # ================================
        n_runs = 30
        pop_size = 5000
        max_f = 2000000

        summary_data = []

        # --- RUN PMX ---
        print("\n>>> Running PMX...")
        start_pmx = time.perf_counter()
        pmx_bench = benchmark(
            n_runs, opt_val, n, matrix_a, matrix_b, 
            pop_size=pop_size, max_f=max_f, version='PMX', 
            portions=0.5, injection_period=20, injection_rate=0.5
        )
        time_pmx = time.perf_counter() - start_pmx
        print(f"PMX Execution Time: {time_pmx:.2f} seconds")
        
        summary_data.append({
            'Version': 'PMX', 'Best_GAP': pmx_bench['best_gap_overall'], 'Mean_GAP': pmx_bench['mean_gap'],
            'Worst_GAP': pmx_bench['worst_gap'], 'Std_GAP': pmx_bench['std_gap'], 'Best_Score': pmx_bench['best_score_overall'],
            'Optimum_Hits': f"{pmx_bench['optimum_hits']}/{n_runs}", 'Total_Time_Sec': round(time_pmx, 2)
        })

        # --- RUN WRAO ---
        print("\n>>> Running WRAO...")
        start_wrao = time.perf_counter()
        wrao_bench = benchmark(
            n_runs, opt_val, n, matrix_a, matrix_b, 
            pop_size=pop_size, max_f=max_f, version='WRAO', 
            portions=0.5, injection_period=20, injection_rate=0.5
        )
        time_wrao = time.perf_counter() - start_wrao
        print(f"WRAO Execution Time: {time_wrao:.2f} seconds")
        
        summary_data.append({
            'Version': 'WRAO', 'Best_GAP': wrao_bench['best_gap_overall'], 'Mean_GAP': wrao_bench['mean_gap'],
            'Worst_GAP': wrao_bench['worst_gap'], 'Std_GAP': wrao_bench['std_gap'], 'Best_Score': wrao_bench['best_score_overall'],
            'Optimum_Hits': f"{wrao_bench['optimum_hits']}/{n_runs}", 'Total_Time_Sec': round(time_wrao, 2)
        })
        
        # --- RUN AO ---
        print("\n>>> Running AO...")
        start_ao = time.perf_counter()
        ao_bench = benchmark(
            n_runs, opt_val, n, matrix_a, matrix_b, 
            pop_size=pop_size, max_f=max_f, version='AO', 
            portions=0.5, injection_period=20, injection_rate=0.5
        )
        time_ao = time.perf_counter() - start_ao
        print(f"AO Execution Time: {time_ao:.2f} seconds")
        
        summary_data.append({
            'Version': 'AO', 'Best_GAP': ao_bench['best_gap_overall'], 'Mean_GAP': ao_bench['mean_gap'],
            'Worst_GAP': ao_bench['worst_gap'], 'Std_GAP': ao_bench['std_gap'], 'Best_Score': ao_bench['best_score_overall'],
            'Optimum_Hits': f"{ao_bench['optimum_hits']}/{n_runs}", 'Total_Time_Sec': round(time_ao, 2)
        })

        # ============================================================
        # ZAPIS STATYSTYK I RAPORTÓW
        # ============================================================
        df_summary = pd.DataFrame(summary_data)
        
        # 1. Ogólne podsumowanie tabelaryczne (CSV)
        csv_summary_name = instance_result_folder / "summary_comparison.csv"
        df_summary.to_csv(csv_summary_name, index=False)
        
        # 2. Szczegółowe dane run-by-run
        detailed_data = {
            'Run_Number': range(1, n_runs + 1),
            'PMX_Scores': pmx_bench['all_scores'], 'PMX_Gaps': pmx_bench['all_gaps'], 'PMX_Times': pmx_bench['all_durations'],
            'WRAO_Scores': wrao_bench['all_scores'], 'WRAO_Gaps': wrao_bench['all_gaps'], 'WRAO_Times': wrao_bench['all_durations'],
            'AO_Scores': ao_bench['all_scores'], 'AO_Gaps': ao_bench['all_gaps'], 'AO_Times': ao_bench['all_durations']
        }
        df_detailed = pd.DataFrame(detailed_data)
        csv_detailed_name = instance_result_folder / "detailed_runs.csv"
        df_detailed.to_csv(csv_detailed_name, index=False)

        # 3. Czytelny zbiorczy raport tekstowy
        report_txt_name = instance_result_folder / "final_report.txt"
        with open(report_txt_name, 'w', encoding='utf-8') as f:
            f.write(f"REPORT FOR INSTANCE: {name} (Size: {n}, Optimal Score: {opt_val})\n")
            f.write("="*70 + "\n\n")
            f.write(df_summary.to_string(index=False))
            f.write("\n\n" + "="*70 + "\n")
            f.write("BEST PERMUTATIONS FOUND BY EACH METHOD:\n")
            f.write("-" * 40 + "\n")
            f.write(f"PMX  Best Permutation: {pmx_bench['best_permutation_overall']}\n")
            f.write(f"WRAO Best Permutation: {wrao_bench['best_permutation_overall']}\n")
            f.write(f"AO   Best Permutation: {ao_bench['best_permutation_overall']}\n")

        # ============================================================
        # 4. ZAPIS POPULACJI DO 3 OSOBNYCH PLIKÓW TEXTOWYCH
        # ============================================================
        print(">>> Saving detailed population snapshots to 3 separate files...")
        
        save_population_snapshots(instance_result_folder / "snapshots_PMX.txt", "PMX", name, pmx_bench)
        save_population_snapshots(instance_result_folder / "snapshots_WRAO.txt", "WRAO", name, wrao_bench)
        save_population_snapshots(instance_result_folder / "snapshots_AO.txt", "AO", name, ao_bench)

        print(f"\n[SUCCESS] Saved all metrics and 3 separate snapshot logs to: '{instance_result_folder}'")

if __name__ == "__main__":
    main()