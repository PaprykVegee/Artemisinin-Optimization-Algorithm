# Dokumentacja Techniczna Projektu: Hybrydowe Algorytmy Artemisinin Optimizer (AO) w Problemie QAP

---

## 1. Cel projektu

Głównym celem projektu jest implementacja, adaptacja oraz zaawansowana ewaluacja metaheurystyki zainspirowanej naturą – **Algorytmu Optymalizacji Artemizyniny (Artemisinin Optimizer - AO)** – do rozwiązywania dyskretnych problemów kombinatorycznych, ze szczególnym uwzględnieniem **Kwadratowego Zagadnienia Przydziału (QAP - Quadratic Assignment Problem)**.

Klasyczny algorytm AO został zaprojektowany do optymalizacji globalnej w ciągłych przestrzeniach poszukiwań $\mathbb{R}^n$. Niniejsza implementacja skupia się na przekształceniu mechanizmów matematycznych tego algorytmu do pracy w dziedzinie dyskretnej (permutacyjnej) przy użyciu strategii kodowania i dekodowania pozycji typu **ROV (Rank Order Value)**.

### Cele szczegółowe:
1. **Implementacja trzech wariantów AO:** Standardowego hybrydowego AO, wersji z ważonym przywódcą (WRAO) oraz wersji z iniekcją struktur elity na poziomie genetycznym (PMX).
2. **Hybrydyzacja eksploracji i eksploatacji:** Połączenie globalnych zdolności poszukiwawczych AO z deterministycznym przeszukiwaniem lokalnym typu *2-opt*.
3. **Analiza porównawcza:** Benchmarking stabilności, zbieżności i efektywności czasowej zaimplementowanych algorytmów na bazie trudnych instancji z biblioteki **QAPLIB**.

---

## 2. Definicje zmiennych i aparat matematyczny

W celu zachowania pełnej jednoznaczności, w całej dokumentacji oraz opisach algorytmów stosuje się ujednolicone oznaczenia matematyczne i odpowiadające im zmienne programistyczne:

* $n$ (`n_dim` / `D`) – Wymiar problemu (liczba obiektów oraz liczba dostępnych lokacji).
* $A$ (`flow_matrix`) – Macierz przepływu o wymiarach $n \times n$. Element $A_{i,j}$ definiuje intensywność przepływu między obiektem $i$ a obiektem $j$.
* $B$ (`distance_matrix`) – Macierz odległości o wymiarach $n \times n$. Element $B_{k,l}$ definiuje odległość fizyczną między lokacją $k$ a lokacją $l$.
* $\pi$ (`permutation`) – Dyskretny wektor permutacji (rozwiązanie QAP) o długości $n$, gdzie $\pi[i] = k$ oznacza przypisanie obiektu $i$ do lokacji $k$.
* $X_i$ (`population[i]`) – Ciągły wektor pozycji $i$-tego agenta populacji w przestrzeni poszukiwań. $X_{i,d} \in [-1, 1]$ oznacza współrzędną w wymiarze $d$.
* $N$ (`pop_size`) – Całkowita liczba agentów poszukujących (rozmiar populacji).
* $MaxF$ (`max_f`) – Maksymalny budżet obliczeniowy zdefiniowany jako liczba dozwolonych ewaluacji funkcji celu.
* $f$ (`self.f`) – Aktualny licznik wykonanych ewaluacji funkcji kosztu.
* $t$ – Aktualna iteracja algorytmu.
* $T$ – Maksymalna zakładana liczba iteracji.

---

## 3. Analiza matematyczna i architektura oprogramowania

### 3.1. Uniwersalne funkcje pomocnicze i optymalizacje (Numba Core)
Wszystkie krytyczne operacje matematyczne i algorytmiczne zostały oddelegowane do funkcji kompilowanych w trybie JIT (`@njit(cache=True)`) za pomocą biblioteki Numba. Pozwala to na osiągnięcie wydajności bliskiej językowi C.

#### 1. Mapowanie ciągłe na dyskretne: `rov_mapping_numba(continuous_vector)`
Przekształca wektor ciągły $X_i \in \mathbb{R}^n$ w poprawną permutację kombinatoryczną $\pi$. Przypisuje elementom wektora rangi (indeksy) wynikające z ich posortowanych wartości:
$$\pi = \text{argsort}(X_i)$$

#### 2. Obliczanie funkcji celu QAP: `calculate_qap_fitness_numba(A, B, permutation)`
Wyznacza całkowity koszt przydziału na podstawie iloczynu Frobeniusa macierzy wejściowych:
$$f(\pi) = \sum_{i=1}^{n} \sum_{j=1}^{n} A_{i,j} \cdot B_{\pi[i], \pi[j]}$$

#### 3. Mapowanie zwrotne (Odwrotne): `map_back_numba(permutation)`
Po operacjach optymalizacji lokalnej zachodzi potrzeba synchronizacji pozycji ciągłej $X_i$ z ulepszoną permutacją $\pi$. Transformacja mapuje pozycje elementów na równomiernie rozłożone wartości w przedziale $[-1, 1]$:
$$X_{i, \pi[k]} = -1.0 + \frac{2.0 \cdot k}{n - 1}, \quad \text{dla } k = 0, 1, \dots, n-1$$

#### 4. Algorytm Przeszukiwania Lokalnego: `full_2opt_numba(permutation, A, B, current_f, max_f)`
Algorytm przeszukiwania lokalnego z kryterium pierwszej poprawy (*first-improvement*). Generuje otoczenie rozwiązania poprzez systematyczne zamiany par elementów (pairwise swaps). 

Dla każdej pary $(i, j)$, gdzie $1 \le i < j \le n$, konstruowana jest kandydacka permutacja $\pi'$, w której zamieniono przypisania $\pi[i]$ oraz $\pi[j]$. Zmiana jest akceptowana natychmiast, gdy:
$$f(\pi') < f(\pi)$$
Proces powtarza się cyklicznie do momentu, w którym w pełnym przejściu nie udaje się znaleźć żadnej poprawy (osiągnięto lokalne minimum) lub gdy $current\_f \ge max\_f$.

---

### 3.2. Implementacja 1: Standardowy Artemisinin Optimizer (AO)
*Plik źródłowy: `ao_algorithm.py`*

Algorytm AO opiera się na dynamicznym przełączaniu między fazą wstrząsania (eksploracja globalna) a fazą podążania za liderem (eksploatacja).

#### Dynamiczne parametry adaptacyjne:
W każdej iteracji wyznaczane są dwa kluczowe parametry sterujące trajektorią ruchu:
1. **Prawdopodobieństwo mutacji komponentu ($K$):**
   $$K = 1.0 - \left(\frac{f}{MaxF}\right)^2$$
2. **Współczynnik kroku zbieżności ($c$):**
   $$c = 2.0 \cdot \left(1.0 - \frac{f}{MaxF}\right)$$

#### Matematyczny model aktualizacji pozycji:
Dla każdego agenta $i$ w populacji, losowana jest wartość $r_1 \in [0, 1]$. Wybór strategii ruchu zależy od stanu licznika ewaluacji:

* **Faza 1: Wstrząsanie (Gdy $r_1 > 0.5$):**
    Dla każdego wymiaru $d \in \{1, \dots, n\}$, jeśli losowa wartość $r_2 < K$, pozycja agenta ulega rozproszeniu wokół losowego osobnika z populacji ($X_{rand}$) z użyciem rozkładu normalnego $\mathcal{N}(0,1)$:
    $$X_{i,d}^{t+1} = X_{rand,d}^{t} + \mathcal{N}(0, 1) \cdot \left(X_{rand,d}^{t} - X_{i,d}^{t}\right)$$

* **Faza 2: Ruch w stronę lidera (Gdy $r_1 \le 0.5$):**
    Agent przemieszcza się w kierunku aktualnie najlepszego globalnego rozwiązania ($X_{best}$):
    $$X_{i,d}^{t+1} = X_{i,d}^{t} + c \cdot r_3 \cdot \left(X_{best,d}^{t} - X_{i,d}^{t}\right)$$
    gdzie $r_3 \in [0, 1]$ to losowa zmienna o rozkładzie jednostajnym.

#### Schemat blokowy - Standardowe AO


---

### 3.3. Implementacja 2: Weighted Artemisinin Optimizer (WRAO)
*Plik źródłowy: `wrao_algorithm.py`*

Wariant WRAO modyfikuje fazę drugą. Zamiast jednoznacznego przyciągania do pojedynczego punktu $X_{best}$, agenci poruszają się w kierunku wirtualnego środka ciężkości wyznaczonego przez elitę populacji.

#### Matematyczny model wyznaczania ważonego lidera:
1. Populacja jest sortowana rosnąco według wartości funkcji kosztu.
2. Wybierana jest subpopulacja elity o rozmiarze $M = \lceil \text{ranking\_portion} \cdot N \rceil$.
3. Dla każdego wybranego osobnika $m \in \{1, \dots, M\}$ wyznacza się masę (wagę) $w_m$ na podstawie pozycji w rankingu (im mniejszy koszt, tym wyższa waga):
   $$w_m = M - m + 1$$
4. Wirtualna pozycja ważonego lidera $X_{weighted}$ w każdym wymiarze $d$ obliczana jest jako:
   $$X_{weighted, d} = \frac{\sum_{m=1}^{M} w_m \cdot X_{m, d}}{\sum_{m=1}^{M} w_m}$$

W równaniu ruchu Fazy 2, wektor $X_{best}$ zostaje zastąpiony przez skonsolidowany wektor $X_{weighted}$:
$$X_{i,d}^{t+1} = X_{i,d}^{t} + c \cdot r_3 \cdot \left(X_{weighted,d} - X_{i,d}^{t}\right)$$

#### Schemat blokowy - WRAO


---

### 3.4. Implementacja 3: AO z iniekcją elity (PMX)
*Plik źródłowy: `ao_algorithm_pmx.py`*

Ten wariant przenosi część operacji bezpośrednio na struktury dyskretne. Iniekcja fragmentów kodu genetycznego lidera zachodzi z częstotliwością określoną przez parametr `injection_period`.

#### 1. Odległość Hamminga: `hamming_distance_numba(p1, p2)`
Miara odległości między dwoma dyskretnymi rozwiązaniami (permutacjami):
$$D_H(\pi_1, \pi_2) = \sum_{k=1}^{n} \mathbb{I}(\pi_1[k] \neq \pi_2[k])$$
gdzie $\mathbb{I}$ to funkcja wskaźnikowa (zwraca 1 gdy warunek jest prawdziwy, 0 w przeciwnym razie).

#### 2. Mechanizm iniekcji elity: `elite_injection_numba`
Wielkość iniekcji genetycznej (liczba pozycji do skopiowania) jest funkcją odległości Hamminga od lidera:
$$\text{injection\_size} = \max\left(1, \lfloor \text{injection\_rate} \cdot D_H(\pi_i, \pi_{best}) \rfloor\right)$$

Operator iniekcji buduje nowe rozwiązanie potomne $\pi^{new}$ dla agenta bazowego $\pi_i$ wykorzystując geny lidera $\pi_{best}$:
1. Losowany jest indeks początkowy $start$ wycinanego bloku.
2. Geny z przedziału $[start, start + \text{injection\_size}]$ są bezpośrednio kopiowane z $\pi_{best}$ do $\pi^{new}$:
   $$\pi^{new}[k] = \pi_{best}[k], \quad \text{dla } k \in [start, start + \text{injection\_size}]$$
3. Pozostałe wolne pozycje w $\pi^{new}$ są sukcesywnie uzupełniane elementami z permutacji bazowej $\pi_i$. Jeśli dany element z $\pi_i$ już istnieje w zaalokowanym bloku, pomija się go w celu uniknięcia duplikacji (zapewnienie poprawności permutacji).

#### Schemat blokowy - PMX

## 4. Oprogramowanie i instrukcja użycia

### Struktura modułów projektu
* `main.py` – Koordynator eksperymentu. Zarządza ścieżkami, strukturą katalogów wynikowych oraz sekwencyjnym wywoływaniem testów.
* `benchmark.py` – Moduł statystyczny. Uruchamia dany algorytm zadaną liczbę razy ($30$), kontroluje ziarno generatora losowego, mierzy czasy wykonania procesów i oblicza wskaźniki błędów.
* `data_loader.py` – Parser plików wejściowych. Konwertuje tekstowe pliki `.dat` (macierze QAP) oraz `.sln` (optima globalne) na macierze NumPy.

### Instrukcja wdrożenia i uruchomienia
1. Umieść pliki źródłowe w jednym katalogu roboczym.
2. Utwórz folder `Scenarios/` i wgraj do niego pliki instancji (np. `bur26f.dat` i `bur26f.sln`).
3. Uruchom instalację zależności w terminalu systemowym:
   ```bash
   pip install numpy numba pandas matplotlib
4. Wykonaj skrypt głównt
   ```bash
   python main.py
5. Przejdź do wygenerowanego katalogu `Result/bur26f/`, aby przeanalizować raporty wydajnościowe (`final_report.txt`)


## 5. Testy empiryczne i wyniki eksperymentalne

Jakość dopasowania końcowych rozwiązań jest reprezentowana przez wskaźnik **GAP** (błąd względny do optimum znalezionego w literaturze), wyrażany w procentach:

$$\text{GAP} = \frac{\text{Best\_Score} - \text{Optimum}}{\text{Optimum}} \cdot 100\%$$
