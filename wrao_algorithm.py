import numpy as np
from numba import njit

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
                    tmp = p[i]
                    p[i] = p[j]
                    p[j] = tmp

        p = best_p.copy()

    return best_p, best_c, current_f


class WeightedArtemisininOptimizer:
    """
    ============================================================================
    CLASS: WeightedArtemisininOptimizer (Hybrid WRAO + 2-opt)
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
        optimum=None,
        ranking_portion=0.3
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
        :param ranking_portion: Fraction of the best population used
                                in weighted ranking [0, 1].
        """

        self.D = n_dim

        self.A = np.ascontiguousarray(flow_matrix.astype(np.int64))

        self.B = np.ascontiguousarray(dist_matrix.astype(np.int64))

        self.N = pop_size

        self.MaxF = max_f

        self.f = 0

        self.optimum = optimum

        self.ranking_portion = ranking_portion

        self.population = np.random.uniform(
            -1,
            1,
            (self.N, self.D)
        )

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

        return calculate_qap_fitness_numba(
            self.A,
            self.B,
            permutation
        )

    def _full_2opt(self, permutation):
        """
        HEADER: Local Search Optimizer (Full 2-opt)
        ---------------------------------------------------
        Description: Systematically explores the neighborhood of a permutation
        by performing pairwise swaps. Logs global improvements to history.
        """

        best_p, best_c, updated_f = full_2opt_numba(
            permutation.astype(np.int64),
            self.A,
            self.B,
            self.f,
            self.MaxF
        )

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

    def _calculate_weighted_leader(self):
        """
        HEADER: Weighted Ranking Leader Calculation
        ---------------------------------------------------
        Description: Computes a weighted average leader based on the best
        ranked agents in the population. Better solutions receive
        larger weights.
        """

        sorted_idx = np.argsort(self.fitness)

        top_k = max(1, int(self.N * self.ranking_portion))

        selected_idx = sorted_idx[:top_k]

        selected_population = self.population[selected_idx]

        selected_fitness = self.fitness[selected_idx]

        weights = 1.0 / (selected_fitness + 1e-12)

        weights = weights / np.sum(weights)

        weighted_leader = np.sum(
            selected_population * weights[:, np.newaxis],
            axis=0
        )

        return weighted_leader

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

            if self.best_agent is None or f_ls < self.best_fitness:

                self.best_fitness = f_ls

                self.best_perm = p_ls.copy()

                self.best_agent = self.population[i].copy()

                self.best_cost_history.append(
                    self.best_fitness
                )

    def optimize(self):
            """
            HEADER: Main Optimization Loop (WRAO + Local Search)
            -----------------------------------------------
            Description: Iteratively executes the algorithm, combining "Shaking"
            (perturbation to escape local minima) with local intensification.

            :return: (best_permutation, best_cost, convergence_history, population_snapshots)
            """

            self.initialize()

            population_snapshots = {}

            if self.best_agent is None:
                self.best_agent = self.population[0].copy()
                self.best_fitness = self.fitness[0]
                self.best_perm = self.rov_mapping(self.population[0]).copy()

            population_snapshots["start"] = [(self.rov_mapping(agent) + 1).tolist() for agent in self.population]

            midpoint_1 = self.MaxF * 0.33
            midpoint_2 = self.MaxF * 0.66
            
            captured_mid1 = False
            captured_mid2 = False

            while self.f < self.MaxF:
                progress = self.f / self.MaxF

                K = 0.4 * (1 - progress)
                c = 2.0 * np.exp(-(progress ** 2))

                ranking_leader = self._calculate_weighted_leader()

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
                            candidate_pos[j] = (
                                ranking_leader[j]
                                + np.random.normal(0, 0.4)
                            )
                        else:
                            candidate_pos[j] = (
                                c * ranking_leader[j]
                                + (1 - c) * candidate_pos[j]
                            )

                    perm_cand = self.rov_mapping(candidate_pos)
                    p_ls, f_ls = self._full_2opt(perm_cand)

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

            if not captured_mid1:
                population_snapshots["mid_1"] = [(self.rov_mapping(agent) + 1).tolist() for agent in self.population]
            if not captured_mid2:
                population_snapshots["mid_2"] = [(self.rov_mapping(agent) + 1).tolist() for agent in self.population]

            population_snapshots["end"] = [(self.rov_mapping(agent) + 1).tolist() for agent in self.population]

            return (
                np.array(self.best_perm) + 1,
                self.best_fitness,
                self.best_cost_history,
                population_snapshots
            )