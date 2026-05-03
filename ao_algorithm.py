import numpy as np
from data_loader import load_instance, load_solution

class ArtemisininOptimizer:
    """
    Implementacja algorytmu Artemisinin Optimization (AO) 
    dostosowana do Problemu Kwadratowego Zagadnienia Przypisania (QAP).
    """

    def __init__(self, n_dim, flow_matrix, dist_matrix, pop_size=30, max_f=10000):
        self.D = n_dim                  # Wymiar problemu (liczba obiektów)
        self.A = flow_matrix            # Macierz przepływu
        self.B = dist_matrix            # Macierz odległości
        self.N = pop_size               # Rozmiar populacji
        self.MaxF = max_f               # Maksymalna liczba ocen funkcji celu
        self.f = 0                      # Licznik ewaluacji
        
        self.population = np.random.uniform(-1, 1, (self.N, self.D))
        self.fitness = np.zeros(self.N)
        
        self.best_agent = None
        self.best_fitness = float('inf')

    def rov_mapping(self, continuous_vector):
        """
        Mapowanie Random Order Value (ROV).
        Konwertuje wektor ciągły na dyskretną permutację.
        """
        return np.argsort(continuous_vector)

    def calculate_qap_fitness(self, continuous_vector):
        """
        Oblicza koszt QAP dla danego agenta.
        """
        p = self.rov_mapping(continuous_vector)

        cost = np.sum(self.A * self.B[p][:, p])
        return cost

    def initialize(self):
        """
        Faza startowa: ocena początkowej populacji.
        """
        for i in range(self.N):
            self.fitness[i] = self.calculate_qap_fitness(self.population[i])
            self.f += 1
            
            if self.fitness[i] < self.best_fitness:
                self.best_fitness = self.fitness[i]
                self.best_agent = self.population[i].copy()

    def update_position(self, i, j, K, c):
        """
        Logika aktualizacji pozycji oparta na trzech fazach AO (Eq. 7, 8, 11).
        """
        r1 = np.random.rand()
        r2 = np.random.rand()
        
        if r2 < K:
            if r1 < 0.5:
                self.population[i, j] += c * self.population[i, j] * ((-1)**np.random.randint(2))
            else:
                self.population[i, j] += c * self.best_agent[j] * ((-1)**np.random.randint(2))

        else:
            r_indices = np.random.choice([idx for idx in range(self.N) if idx != i], 2, replace=False)
            d = 0.5 
            self.population[i, j] += d * (self.population[r_indices[0], j] - self.population[r_indices[1], j])

        if np.random.rand() < 0.2:
            self.population[i, j] = 0.5 * (self.population[i, j] + self.best_agent[j])

    def optimize(self):
        """
        Główna pętla algorytmu AO.
        """
        self.initialize()
        
        while self.f < self.MaxF:
            progress = self.f / self.MaxF
            K = 0.5 * (1 - progress)  
            c = 2 * np.exp(-progress) * np.abs(np.cos(np.pi * progress))
            
            for i in range(self.N):
                old_position = self.population[i].copy()
                
                for j in range(self.D):
                    self.update_position(i, j, K, c)
                
                self.population[i] = np.clip(self.population[i], -1, 1)
                
                current_fit = self.calculate_qap_fitness(self.population[i])
                self.f += 1
                
                if current_fit < self.fitness[i]:
                    self.fitness[i] = current_fit
                else:
                    if np.random.rand() > 0.1: 
                         self.population[i] = old_position

                if current_fit < self.best_fitness:
                    self.best_fitness = current_fit
                    self.best_agent = self.population[i].copy()

            if self.f % (self.N * 10) == 0 or self.f >= self.MaxF:
                print(f"Ewaluacje: {self.f}/{self.MaxF} | Najlepszy koszt (QAP): {self.best_fitness}")

        return self.rov_mapping(self.best_agent), self.best_fitness



instance = 'Scenarios/Christofides/chr12a.dat'
solution_file = 'Scenarios/Christofides/solution/chr12a.sln'

n, matrix_a, matrix_b = load_instance(instance)
n, optimal, solution = load_solution(solution_file)