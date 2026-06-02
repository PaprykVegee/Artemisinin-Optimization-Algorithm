import numpy as np
from numba import njit

# ============================================================================
# NUMBA-ACCELERATED UTILITIES
# ============================================================================

@njit(cache=True)
def rov_mapping_numba(continuous_vector):
    """
    HEADER: Rank Order Value (ROV) Mapping
    -----------------------------------------
    Description: Converts a continuous real-number vector into a discrete
    permutation by sorting the indices based on their values (ranks).
    """
    return np.argsort(continuous_vector)


@njit(cache=True)
def calculate_qap_fitness_numba(A, B, permutation):
    """
    HEADER: QAP Objective Function Calculation
    ------------------------------------
    Description: Computes the total cost of an assignment based on the
    Frobenius inner product of the flow and distance matrices.
    """
    n = permutation.shape[0]
    total = 0

    for i in range(n):
        pi = permutation[i]
        for j in range(n):
            pj = permutation[j]
            total += A[i, j] * B[pi, pj]

    return total


@njit(cache=True)
def map_back_numba(permutation):
    """
    HEADER: Reverse Mapping to Continuous Space
    -----------------------------------------------
    Description: Converts a discrete permutation back into a continuous
    vector to allow further mathematical operations by the AO algorithm.
    """
    D = permutation.shape[0]

    ranks = np.empty(D, dtype=np.int64)

    for i in range(D):
        ranks[permutation[i]] = i

    new_vec = np.empty(D, dtype=np.float64)

    for i in range(D):
        new_vec[i] = (ranks[i] / (D - 1)) * 2.0 - 1.0

    return new_vec


@njit(cache=True)
def full_2opt_numba(permutation, A, B, current_f, max_f):
    """
    HEADER: Local Search Optimizer (Full 2-opt)
    ---------------------------------------------------
    Description: Systematically explores the neighborhood of a permutation
    by performing pairwise swaps.
    """

    D = permutation.shape[0]

    p = permutation.copy()
    best_p = permutation.copy()

    best_c = calculate_qap_fitness_numba(A, B, best_p)

    improved = True

    while improved:

        improved = False

        for i in range(D):

            for j in range(i + 1, D):

                if current_f >= max_f:
                    return best_p, best_c, current_f

                # test swap
                tmp = p[i]
                p[i] = p[j]
                p[j] = tmp

                curr_c = calculate_qap_fitness_numba(A, B, p)

                current_f += 1

                if curr_c < best_c:

                    best_c = curr_c
                    best_p = p.copy()
                    improved = True

                else:
                    # rollback swap
                    tmp = p[i]
                    p[i] = p[j]
                    p[j] = tmp

        p = best_p.copy()

    return best_p, best_c, current_f


# ============================================================================
# CLASS: ArtemisininOptimizer (Hybrid WRAO + 2-opt)
# ============================================================================

class ArtemisininOptimizer:
    """
    ============================================================================
    CLASS: ArtemisininOptimizer (Hybrid WRAO + 2-opt)
    ============================================================================
    Description: A nature-inspired optimization algorithm based on the extraction
    of Artemisinin, adapted for the Quadratic Assignment Problem (QAP).
    It combines global exploration via Artemisinin Optimization (AO) with
    intensive local exploitation using the 2-opt strategy.
    """

    def __init__(
        self,
        n_dim,
        flow_matrix,
        dist_matrix,
        pop_size=15,
        max_f=100000,
        optimum=None
    ):
        """
        HEADER: Algorithm Parameter Initialization
        -------------------------------------------
        :param n_dim: Problem dimension (number of objects/locations).
        :param flow_matrix: Matrix representing flows between objects (A).
        :param dist_matrix: Matrix representing distances between locations (B).
        :param pop_size: Number of agents in the population.
        :param max_f: Computational budget (maximum number of fitness evaluations).
        :param optimum: The known optimal value for the problem instance.
        """

        self.D = n_dim

        # Ensure contiguous arrays for Numba performance
        self.A = np.ascontiguousarray(flow_matrix.astype(np.int64))
        self.B = np.ascontiguousarray(dist_matrix.astype(np.int64))

        self.N = pop_size
        self.MaxF = max_f

        self.f = 0  # Evaluation counter

        self.optimum = optimum

        # Initial population in continuous space [-1, 1]
        self.population = np.random.uniform(-1, 1, (self.N, self.D))

        self.fitness = np.full(self.N, np.inf)

        self.best_agent = None
        self.best_perm = None

        self.best_fitness = np.inf

        self.best_cost_history = []

    def rov_mapping(self, continuous_vector):
        """
        HEADER: Rank Order Value (ROV) Mapping
        -----------------------------------------
        Description: Converts a continuous real-number vector into a discrete
        permutation by sorting the indices based on their values (ranks).
        """
        return rov_mapping_numba(continuous_vector)

    def calculate_qap_fitness(self, permutation):
        """
        HEADER: QAP Objective Function Calculation
        ------------------------------------
        Description: Computes the total cost of an assignment based on the
        Frobenius inner product of the flow and distance matrices.
        """
        return calculate_qap_fitness_numba(self.A, self.B, permutation)

    def _full_2opt(self, permutation):
        """
        HEADER: Local Search Optimizer (Full 2-opt)
        ---------------------------------------------------
        Description: Systematically explores the neighborhood of a permutation
        by performing pairwise swaps. Logs global improvements to history.
        """

        best_p, best_c, updated_f = full_2opt_numba(permutation.astype(np.int64), self.A, self.B, self.f, self.MaxF)

        self.f = updated_f
        return best_p, best_c

    def _map_back(self, p):
        """
        HEADER: Reverse Mapping to Continuous Space
        -----------------------------------------------
        Description: Converts a discrete permutation back into a continuous
        vector to allow further mathematical operations by the AO algorithm.
        """
        return map_back_numba(p)

    def initialize(self):
        """
        HEADER: Population Initialization and Refinement
        -------------------------------------------
        Description: Initializes the population and refines each agent's
        starting position using a full 2-opt local search.
        """

        for i in range(self.N):
            perm = self.rov_mapping(self.population[i])
            p_ls, f_ls = self._full_2opt(perm)
            self.population[i] = self._map_back(p_ls)
            self.fitness[i] = f_ls

            # Ensure a global leader is assigned to avoid NoneType errors
            if self.best_agent is None or f_ls < self.best_fitness:

                self.best_fitness = f_ls
                self.best_perm = p_ls.copy()
                self.best_agent = self.population[i].copy()
                self.best_cost_history.append(self.best_fitness)

    def optimize(self):
            """
            HEADER: Main Optimization Loop (WRAO + Local Search)
            -----------------------------------------------
            Description: Iteratively executes the algorithm, combining "Shaking"
            (perturbation to escape local minima) with local intensification.

            :return: (best_permutation, best_cost, convergence_history, population_snapshots)
            """

            self.initialize()

            # Słownik na migawki populacji permutacji
            population_snapshots = {}

            # Fail-safe to ensure best_agent exists
            if self.best_agent is None:
                self.best_agent = self.population[0].copy()
                self.best_fitness = self.fitness[0]
                self.best_perm = self.rov_mapping(self.population[0]).copy()

            # 1. Zrzut na POCZĄTKU algorytmu (zmapowany na permutacje + 1 dla formatu 1-indexed)
            population_snapshots["start"] = [(self.rov_mapping(agent) + 1).tolist() for agent in self.population]

            # Definiujemy punkty kontrolne dla parametrów MaxF (środek 1 i środek 2)
            midpoint_1 = self.MaxF * 0.33
            midpoint_2 = self.MaxF * 0.66
            
            captured_mid1 = False
            captured_mid2 = False

            while self.f < self.MaxF:
                progress = self.f / self.MaxF

                # Dynamic WRAO parameters
                K = 0.4 * (1 - progress)
                c = 2.0 * np.exp(-(progress ** 2))

                # --- PRZECHWYTYWANIE POPULACJI W TRAKCIE (ŚRODEK 1 i 2) ---
                if not captured_mid1 and self.f >= midpoint_1:
                    population_snapshots["mid_1"] = [(self.rov_mapping(agent) + 1).tolist() for agent in self.population]
                    captured_mid1 = True
                    
                if not captured_mid2 and self.f >= midpoint_2:
                    population_snapshots["mid_2"] = [(self.rov_mapping(agent) + 1).tolist() for agent in self.population]
                    captured_mid2 = True

                for i in range(self.N):

                    if self.f >= self.MaxF:
                        break

                    candidate_pos = self.population[i].copy()

                    for j in range(self.D):
                        if np.random.rand() < K:
                            # Elimination phase / Shaking
                            # (random noise injection)
                            candidate_pos[j] = (self.best_agent[j] + np.random.normal(0, 0.4))
                        else:
                            # Movement towards the leader
                            # (WRAO trajectory)
                            candidate_pos[j] = (c * self.best_agent[j] + (1 - c) * candidate_pos[j])

                    # Refine new position with Local Search
                    perm_cand = self.rov_mapping(candidate_pos)
                    p_ls, f_ls = self._full_2opt(perm_cand)

                    # Update agent if a better local minimum is discovered
                    if f_ls < self.fitness[i]:
                        self.fitness[i] = f_ls
                        self.population[i] = self._map_back(p_ls)

                        if f_ls < self.best_fitness:
                            self.best_fitness = f_ls
                            self.best_perm = p_ls.copy()
                            self.best_agent = self.population[i].copy()
                            self.best_cost_history.append(self.best_fitness)

                if (self.optimum is not None and self.best_fitness <= self.optimum):
                    print(f"Optimal solution found with fitness {self.best_fitness} at evaluation {self.f}.")
                    break

            # Upewniamy się, że jeśli pętla przerwała się wcześniej (np. trafiono optimum), 
            # brakujące zrzuty ze środka zostaną uzupełnione aktualnym stanem
            if not captured_mid1:
                population_snapshots["mid_1"] = [(self.rov_mapping(agent) + 1).tolist() for agent in self.population]
            if not captured_mid2:
                population_snapshots["mid_2"] = [(self.rov_mapping(agent) + 1).tolist() for agent in self.population]

            # 4. Zrzut na KONIEC algorytmu
            population_snapshots["end"] = [(self.rov_mapping(agent) + 1).tolist() for agent in self.population]

            # Zwracamy 4 elementy: permutację (skorygowaną o +1), koszt, historię oraz słownik zrzutów
            return np.array(self.best_perm) + 1, self.best_fitness, self.best_cost_history, population_snapshots