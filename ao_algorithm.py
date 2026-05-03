import numpy as np

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

    def __init__(self, n_dim, flow_matrix, dist_matrix, pop_size=15, max_f=100000):
        """
        HEADER: Algorithm Parameter Initialization
        -------------------------------------------
        :param n_dim: Problem dimension (number of objects/locations).
        :param flow_matrix: Matrix representing flows between objects (A).
        :param dist_matrix: Matrix representing distances between locations (B).
        :param pop_size: Number of agents in the population.
        :param max_f: Computational budget (maximum number of fitness evaluations).
        """
        self.D = n_dim
        self.A = flow_matrix
        self.B = dist_matrix
        self.N = pop_size
        self.MaxF = max_f
        self.f = 0  # Evaluation counter

        # Initial population in continuous space [-1, 1]
        self.population = np.random.uniform(-1, 1, (self.N, self.D))
        self.fitness = np.full(self.N, float('inf'))
        self.best_agent = None
        self.best_fitness = float('inf')
        self.best_cost_history = []

    def rov_mapping(self, continuous_vector):
        """
        HEADER: Rank Order Value (ROV) Mapping
        -----------------------------------------
        Description: Converts a continuous real-number vector into a discrete 
        permutation by sorting the indices based on their values (ranks).
        """
        return np.argsort(continuous_vector)

    def calculate_qap_fitness(self, permutation):
        """
        HEADER: QAP Objective Function Calculation
        ------------------------------------
        Description: Computes the total cost of an assignment based on the 
        Frobenius inner product of the flow and distance matrices.
        """
        return np.sum(self.A * self.B[permutation][:, permutation])

    def _full_2opt(self, permutation):
        """
        HEADER: Local Search Optimizer (Full 2-opt)
        ---------------------------------------------------
        Description: Systematically explores the neighborhood of a permutation 
        by performing pairwise swaps. Logs global improvements to history.
        """
        p = list(permutation)
        best_c = self.calculate_qap_fitness(p)
        improved = True
        
        while improved:
            improved = False
            for i in range(self.D):
                for j in range(i + 1, self.D):
                    if self.f >= self.MaxF:
                        return p, best_c
                    
                    # Perform pairwise swap
                    p[i], p[j] = p[j], p[i]
                    curr_c = self.calculate_qap_fitness(p)
                    self.f += 1
                    
                    if curr_c < best_c:
                        best_c = curr_c
                        improved = True
                        # Update global history for monotonic convergence visualization
                        if best_c < self.best_fitness:
                            self.best_fitness = best_c
                            self.best_cost_history.append(self.best_fitness)
                    else:
                        p[i], p[j] = p[j], p[i] # Revert swap if no improvement
            if not improved:
                break
        return p, best_c

    def _map_back(self, p):
        """
        HEADER: Reverse Mapping to Continuous Space
        -----------------------------------------------
        Description: Converts a discrete permutation back into a continuous 
        vector to allow further mathematical operations by the AO algorithm.
        """
        new_vec = np.zeros(self.D)
        ranks = np.argsort(p)
        new_vec = (ranks / (self.D - 1)) * 2 - 1
        return new_vec

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
                self.best_agent = self.population[i].copy()
                self.best_cost_history.append(self.best_fitness)

    def optimize(self):
        """
        HEADER: Main Optimization Loop (WRAO + Local Search)
        -----------------------------------------------
        Description: Iteratively executes the algorithm, combining "Shaking" 
        (perturbation to escape local minima) with local intensification.
        
        :return: (best_permutation, best_cost, convergence_history)
        """
        self.initialize()
        
        # Fail-safe to ensure best_agent exists
        if self.best_agent is None:
            self.best_agent = self.population[0].copy()
            self.best_fitness = self.fitness[0]

        while self.f < self.MaxF:
            progress = self.f / self.MaxF
            # Dynamic WRAO parameters
            K = 0.4 * (1 - progress) 
            weight = 2.0 * np.exp(-(progress**2))
            
            for i in range(self.N):
                if self.f >= self.MaxF: break
                
                candidate_pos = self.population[i].copy()
                for j in range(self.D):
                    if np.random.rand() < K:
                        # Elimination phase / Shaking (random noise injection)
                        candidate_pos[j] = self.best_agent[j] + np.random.normal(0, 0.4)
                    else:
                        # Movement towards the leader (WRAO trajectory)
                        candidate_pos[j] = weight * self.best_agent[j] + (1 - weight) * candidate_pos[j]

                # Refine new position with Local Search
                perm_cand = self.rov_mapping(candidate_pos)
                p_ls, f_ls = self._full_2opt(perm_cand)
                
                # Update agent if a better local minimum is discovered
                if f_ls < self.fitness[i]:
                    self.fitness[i] = f_ls
                    self.population[i] = self._map_back(p_ls)
                    
                    if f_ls < self.best_fitness:
                        self.best_fitness = f_ls
                        self.best_agent = self.population[i].copy()
                        self.best_cost_history.append(self.best_fitness)
            
        return self.rov_mapping(self.best_agent), self.best_fitness, self.best_cost_history