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
        
        # Inicjalizacja populacji w przestrzeni ciągłej [-1, 1]
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
        # Efektywne obliczenie kosztu macierzowego: suma(A * B_przestawione)
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
        
        # --- 1. Faza Kompleksowej Eliminacji (Eq. 7) ---
        # Symulacja globalnej eksploracji (duża dawka leku)
        if r2 < K:
            if r1 < 0.5:
                # Eksploracja wokół bieżącej pozycji
                self.population[i, j] += c * self.population[i, j] * ((-1)**np.random.randint(2))
            else:
                # Eksploracja w kierunku najlepszego agenta
                self.population[i, j] += c * self.best_agent[j] * ((-1)**np.random.randint(2))
        
        # --- 2. Faza Lokalnego Oczyszczania (Eq. 8) ---
        # Symulacja precyzyjnego szukania (lokalna eksploatacja)
        else:
            # Wybór dwóch losowych agentów do wyznaczenia kierunku (jak w DE)
            r_indices = np.random.choice([idx for idx in range(self.N) if idx != i], 2, replace=False)
            d = 0.5 # Współczynnik kroku
            self.population[i, j] += d * (self.population[r_indices[0], j] - self.population[r_indices[1], j])

        # --- 3. Faza Konsolidacji Po-terapeutycznej (Eq. 11) ---
        # Mechanizm krzyżowania informacji (Information Crossover)
        # Wywoływany z pewnym prawdopodobieństwem, by uniknąć optimów lokalnych
        if np.random.rand() < 0.2:
            self.population[i, j] = 0.5 * (self.population[i, j] + self.best_agent[j])

    def optimize(self):
        """
        Główna pętla algorytmu AO.
        """
        self.initialize()
        
        while self.f < self.MaxF:
            # Obliczanie parametrów adaptacyjnych K i c
            progress = self.f / self.MaxF
            K = 0.5 * (1 - progress)  # Prawdopodobieństwo maleje z czasem
            # Parametr c symuluje stężenie artemizyny (według wzorów z artykułu)
            c = 2 * np.exp(-progress) * np.abs(np.cos(np.pi * progress))
            
            for i in range(self.N):
                # Kopiujemy starą pozycję, by móc wykonać ewentualny powrót (greedy)
                old_position = self.population[i].copy()
                
                for j in range(self.D):
                    # Aktualizujemy każdy wymiar (liczbę ciągłą)
                    self.update_position(i, j, K, c)
                
                # Utrzymanie agenta w granicach przestrzeni [-1, 1]
                self.population[i] = np.clip(self.population[i], -1, 1)
                
                # Ocena nowej pozycji po zmapowaniu na permutację
                current_fit = self.calculate_qap_fitness(self.population[i])
                self.f += 1
                
                # Aktualizacja populacji (podejście zachłanne - greedy)
                if current_fit < self.fitness[i]:
                    self.fitness[i] = current_fit
                else:
                    # Jeśli nowa pozycja jest gorsza, wracamy (lub AO może pozwolić na ruch)
                    # W tej wersji stosujemy prosty mechanizm poprawy:
                    if np.random.rand() > 0.1: # 90% szans na powrót do lepszej pozycji
                         self.population[i] = old_position
                    
                # Aktualizacja globalnego najlepszego rozwiązania
                if current_fit < self.best_fitness:
                    self.best_fitness = current_fit
                    self.best_agent = self.population[i].copy()
            
            # Logowanie postępu
            if self.f % (self.N * 10) == 0 or self.f >= self.MaxF:
                print(f"Ewaluacje: {self.f}/{self.MaxF} | Najlepszy koszt (QAP): {self.best_fitness}")

        # Zwrócenie najlepszej znalezionej permutacji i jej kosztu
        return self.rov_mapping(self.best_agent), self.best_fitness



instance = 'Scenarios/Christofides/chr12a.dat'
solution_file = 'Scenarios/Christofides/solution/chr12a.sln'

n, matrix_a, matrix_b = load_instance(instance)
n, optimal, solution = load_solution(solution_file)