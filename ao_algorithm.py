import numpy as np
from data_loader import load_instance, load_solution

#do powtarzalności wyników podczas testów
#np.random.seed(seed=10)

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


        # Inicjalizacja populacji w przestrzeni ciągłej [-1, 1] lub [0, 1]

        self.population = np.random.uniform(-1, 1, (self.N, self.D))
        #self.population = np.random.uniform(0, 1, (self.N, self.D))
        self.fitness = np.zeros(self.N)
        
        self.best_agent = None
        self.best_fitness = float('inf')
        self.best_cost_history = []

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
            
        best_idx = np.argmin(self.fitness)
        self.best_fitness = self.fitness[best_idx]
        self.best_agent = self.population[best_idx].copy()

    def update_position(self, i, j, K, c, fit_norm_i):
        """
        Logika aktualizacji pozycji oparta na trzech fazach AO (Eq. 7, 8, 11).
        """
        r1 = np.random.rand()
        r2 = np.random.rand()
        direction = 1 if np.random.rand() < 0.5 else -1
        
        # --- 1. Faza Kompleksowej Eliminacji (Eq. 7) ---
        # Symulacja globalnej eksploracji (duża dawka leku)

        if r1 < K:
            if np.random.rand() < 0.5:
                # Eksploracja wokół bieżącej pozycji
                # ai,j = ai,j + c * ai,j * (-1)^t
                self.population[i, j] = self.population[i, j] + c * self.population[i, j] * direction
            else:
                # Eksploracja w kierunku najlepszego agenta
                # ai,j = ai,j + c * best,j * (-1)^t
                self.population[i, j] = self.population[i, j] + c * self.best_agent[j] * direction


        # --- 2. Faza Lokalnego Oczyszczania (Eq. 8) ---
        # Symulacja precyzyjnego szukania (lokalna eksploatacja)

        if r2 < fit_norm_i:
            if np.random.rand() < fit_norm_i:
                # Wybór 3 różnych losowych agentów (b1, b2, b3)
                idx = [index for index in range(self.N) if index != i]
                b1, b2, b3 = np.random.choice(idx, 3, replace=False)
                d = np.random.uniform(0.1, 0.6)
                # ai = ab3 + d * (ab1 - ab2)
                self.population[i] = self.population[b3] + d * (self.population[b1] - self.population[b2])

        # --- 3. Faza Konsolidacji Po-terapeutycznej (Eq. 11) ---
        r_post = np.random.rand()
        if r_post < 0.05:
            # Dormant form - agent zostaje w miejscu (nie zmieniamy self.population[i,j])
            pass
        elif r_post < 0.2:
            # Wybudzenie - przejęcie cechy od najlepszego
            self.population[i, j] = self.best_agent[j]


    def optimize(self):
        """
        Główna pętla algorytmu AO.
        """
        self.initialize()
        
        while self.f < self.MaxF:
            # Obliczanie parametrów adaptacyjnych K i c
            K = 1 - (self.f**(1/6)) / (self.MaxF**(1/6))
            c = np.exp(-4 * self.f / self.MaxF)

            f_min = np.min(self.fitness)
            f_max = np.max(self.fitness)
            denom = (f_max - f_min) if (f_max - f_min) > 1e-10 else 1.0
            # Fitnorm: wyższe prawdopodobieństwo dla LEPSZYCH osobników (w QAP mniejszy fitness = lepiej)
            # Musimy odwrócić fitnes, bo QAP to minimalizacja
            fit_norm = (f_max - self.fitness) / denom

    
            for i in range(self.N):
                # Kopiujemy starą pozycję, by móc wykonać ewentualny powrót (greedy)
                old_position = self.population[i].copy()
                
                for j in range(self.D):
                    # Aktualizujemy każdy wymiar (liczbę ciągłą)
                    self.update_position(i, j, K, c, fit_norm[i])
                
                # Utrzymanie agenta w granicach przestrzeni [0, 1]
                #self.population[i] = np.clip(self.population[i], -1, 1)

                # Ocena nowej pozycji po zmapowaniu na permutację
                current_fit = self.calculate_qap_fitness(self.population[i])
                self.f += 1
                
                # Aktualizacja populacji (podejście zachłanne - greedy)
                # Greedy selection
                if current_fit < self.fitness[i]:
                    self.fitness[i] = current_fit
                    if current_fit < self.best_fitness:
                        self.best_fitness = current_fit
                        self.best_agent = self.population[i].copy()
                else:
                    self.population[i] = old_position
            
            self.best_cost_history.append(self.best_fitness)

            # Logowanie postępu
            if self.f % (self.N * 1000) == 0 or self.f >= self.MaxF:
                print(f"Ewaluacje: {self.f}/{self.MaxF} | Najlepszy koszt (QAP): {self.best_fitness}")

           
        # Zwrócenie najlepszej znalezionej permutacji i jej kosztu
        return self.rov_mapping(self.best_agent), self.best_fitness, self.best_cost_history