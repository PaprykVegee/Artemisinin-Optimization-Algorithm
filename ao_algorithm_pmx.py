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
def hamming_distance_numba(p1, p2):
    """
    HEADER: Hamming Distance Calculation
    -----------------------------------------------
    Description: Computes the number of differing positions
    between two permutations.
    """

    D = p1.shape[0]

    dist = 0

    for i in range(D):

        if p1[i] != p2[i]:

            dist += 1

    return dist


@njit(cache=True)
def elite_injection_numba(target_perm, elite_perm, injection_size):
    """
    HEADER: PMX-like Elite Injection Operator
    ---------------------------------------------------
    Description:
    Injects ordered fragments from the elite permutation into
    the target permutation while preserving permutation validity.

    The operator behaves similarly to PMX/OX crossover.
    """

    D = target_perm.shape[0]

    child = np.full(D, -1, dtype=np.int64)

    # ============================================================
    # STEP 1:
    # Copy first 'injection_size' genes from elite permutation
    # ============================================================

    inserted = 0

    for i in range(D):

        if inserted >= injection_size:

            break

        child[i] = elite_perm[i]

        inserted += 1

    # ============================================================
    # STEP 2:
    # Fill remaining positions using target permutation order
    # ============================================================

    child_idx = injection_size

    for i in range(D):

        gene = target_perm[i]

        exists = False

        for j in range(injection_size):

            if child[j] == gene:

                exists = True

                break

        if not exists:

            child[child_idx] = gene

            child_idx += 1

    return child


@njit(cache=True)
def full_2opt_numba(permutation, A, B, current_f, max_f):
    """
    HEADER: Local Search Optimizer (Full 2-opt)
    ---------------------------------------------------
    Description: Systematically explores the neighborhood
    of a permutation by performing pairwise swaps.
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
# CLASS: ArtemisininOptimizer (Hybrid WRAO + 2-opt + PMX Injection)
# ============================================================================

class PMXOptimizer:
    """
    ============================================================================
    CLASS: PMXOptimizer (Hybrid WRAO + 2-opt + PMX Injection)
    ============================================================================
    Description: A nature-inspired optimization algorithm based on the
    extraction of Artemisinin, adapted for the Quadratic Assignment Problem.

    The algorithm combines:
    - Global exploration via Artemisinin Optimization,
    - Intensive local exploitation using 2-opt,
    - PMX-like elite permutation injection based on Hamming distance.
    """

    def __init__(
        self,
        n_dim,
        flow_matrix,
        dist_matrix,
        pop_size=15,
        max_f=100000,
        optimum=None,
        injection_period=10,
        injection_rate=1.0
    ):
        """
        HEADER: Algorithm Parameter Initialization
        -------------------------------------------
        :param n_dim: Problem dimension.
        :param flow_matrix: Flow matrix A.
        :param dist_matrix: Distance matrix B.
        :param pop_size: Number of agents.
        :param max_f: Computational budget.
        :param optimum: Known optimum value.
        :param injection_period: Number of iterations between injections.
        :param injection_rate: Scaling coefficient controlling
                               injection intensity.
        """

        self.D = n_dim
        self.A = np.ascontiguousarray(flow_matrix.astype(np.int64))
        self.B = np.ascontiguousarray(dist_matrix.astype(np.int64))

        self.N = pop_size
        self.MaxF = max_f
        self.f = 0
        self.optimum = optimum
        self.injection_period = injection_period
        self.injection_rate = injection_rate

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
        Description: Converts a continuous real-number vector
        into a discrete permutation.
        """

        return rov_mapping_numba(continuous_vector)

    def calculate_qap_fitness(self, permutation):
        """
        HEADER: QAP Objective Function Calculation
        ------------------------------------
        Description: Computes total assignment cost.
        """

        return calculate_qap_fitness_numba(self.A, self.B, permutation)

    def _full_2opt(self, permutation):
        """
        HEADER: Local Search Optimizer (Full 2-opt)
        ---------------------------------------------------
        Description: Pairwise swap local search.
        """

        best_p, best_c, updated_f = full_2opt_numba(permutation.astype(np.int64),self.A,self.B,self.f,self.MaxF)

        self.f = updated_f

        return best_p, best_c

    def _map_back(self, p):
        """
        HEADER: Reverse Mapping to Continuous Space
        -----------------------------------------------
        Description: Converts permutation back into
        continuous representation.
        """

        return map_back_numba(p)

    def initialize(self):
        """
        HEADER: Population Initialization and Refinement
        -------------------------------------------
        Description: Initializes the population and refines
        each agent using full 2-opt.
        """

        for i in range(self.N):

            perm = self.rov_mapping(self.population[i])

            p_ls, f_ls = self._full_2opt(perm)
            self.population[i] = self._map_back(p_ls)
            self.fitness[i] = f_ls

            # Ensure a global leader is assigned
            if (self.best_agent is None or f_ls < self.best_fitness):

                self.best_fitness = f_ls
                self.best_perm = p_ls.copy()
                self.best_agent = self.population[i].copy()
                self.best_cost_history.append(self.best_fitness)

    def optimize(self):
            """
            HEADER: Main Optimization Loop
            -----------------------------------------------
            Description:
            Executes iterative AO optimization enhanced with:
            - shaking,
            - local search,
            - PMX-like elite injection.

            :return: (best_permutation, best_cost, convergence_history, population_snapshots)
            """

            self.initialize()

            # Słownik na migawki populacji permutacji
            population_snapshots = {}

            # Fail-safe
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

            iteration = 0

            while self.f < self.MaxF:

                iteration += 1
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
                            candidate_pos[j] = (self.best_agent[j] + np.random.normal(0, 0.4))
                        else:
                            # Movement towards leader
                            candidate_pos[j] = (c * self.best_agent[j] + (1 - c) * candidate_pos[j])

                    # ============================================================
                    # CONVERT TO PERMUTATION
                    # ============================================================
                    perm_cand = self.rov_mapping(candidate_pos)

                    # ============================================================
                    # PMX-LIKE ELITE INJECTION
                    # ============================================================
                    if (iteration % self.injection_period == 0 and self.best_perm is not None):
                        # Measure Hamming distance
                        hamming_dist = hamming_distance_numba(perm_cand, self.best_perm)

                        # Determine injection size
                        injection_size = max(1, int(hamming_dist * self.injection_rate))

                        # Inject elite structure
                        perm_cand = elite_injection_numba(perm_cand, self.best_perm, injection_size)

                    # ============================================================
                    # LOCAL REFINEMENT
                    # ============================================================
                    p_ls, f_ls = self._full_2opt(perm_cand)

                    # ============================================================
                    # UPDATE POPULATION
                    # ============================================================
                    if f_ls < self.fitness[i]:
                        self.fitness[i] = f_ls
                        self.population[i] = self._map_back(p_ls)

                        if f_ls < self.best_fitness:
                            self.best_fitness = f_ls
                            self.best_perm = p_ls.copy()
                            self.best_agent = self.population[i].copy()
                            self.best_cost_history.append(self.best_fitness)

                if (self.optimum is not None and self.best_fitness <= self.optimum):
                    print(f"Optimal solution found with fitness {self.best_fitness}at evaluation {self.f}.")
                    break

            # Zabezpieczenie na wypadek przedwczesnego zatrzymania (np. znalezienia optimum)
            if not captured_mid1:
                population_snapshots["mid_1"] = [(self.rov_mapping(agent) + 1).tolist() for agent in self.population]
            if not captured_mid2:
                population_snapshots["mid_2"] = [(self.rov_mapping(agent) + 1).tolist() for agent in self.population]

            # 4. Zrzut na KONIEC algorytmu
            population_snapshots["end"] = [(self.rov_mapping(agent) + 1).tolist() for agent in self.population]

            # Zwracamy dokładnie ten sam zestaw 4 elementów
            return np.array(self.best_perm) + 1, self.best_fitness, self.best_cost_history, population_snapshots