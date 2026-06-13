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


## 6. Results and Experimental Analysis

In this section, the results for the three investigated algorithm variants are presented: the discrete **PMX** operator, the proposed **WRAO** (Weighted Artemisinin Optimizer), and the standard **AO** (Artemisinin Optimization). The tests were conducted on diverse instances of the QAP problem from the QAPLIB library, varying in size (from 25 to 256 locations) and matrix characteristics. 

The **GAP** metric (relative error compared to the known optimum, expressed in percentages) was used to evaluate the quality of the solutions.

### 6.1. Tabular Summary

The following table presents the average and extreme results from 30 independent runs of the algorithm for each instance.

| Instance (Size)<br>*Optimum* | Algorithm | Best GAP [%] | Mean GAP [%] | Worst GAP [%] | Best Score | Optimum Hits | Time [s] |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **tai256c** (256)<br>*Opt: 44,759,294* | PMX<br>**WRAO**<br>AO | 0.2541<br>**0.2286**<br>0.2596 | 0.3172<br>**0.3146**<br>0.3217 | **0.3693**<br>0.4038<br>0.3809 | 44,873,028<br>**44,861,630**<br>44,875,494 | 0/30<br>0/30<br>0/30 | 1840.4<br>1894.0<br>1884.5 |
| **tai150b** (150)<br>*Opt: 498,896,643* | PMX<br>WRAO<br>**AO** | 1.8548<br>1.6239<br>**1.5915** | 2.3226<br>2.1972<br>**2.1526** | 2.7804<br>2.8651<br>**2.6909** | 508,150,294<br>506,998,546<br>**506,837,042** | 0/30<br>0/30<br>0/30 | 948.3<br>922.1<br>820.9 |
| **tai100b** (100)<br>*Opt: 1,185,996,137*| PMX<br>**WRAO**<br>AO | 0.8122<br>**0.5738**<br>0.7517 | 1.4424<br>**1.2860**<br>1.4675 | 2.8036<br>2.4007<br>**2.3132** | 1,195,629,139<br>**1,192,802,106**<br>1,194,911,819| 0/30<br>0/30<br>0/30 | 435.7<br>435.7<br>440.9 |
| **tai100a** (100)<br>*Opt: 21,052,466* | **PMX**<br>WRAO<br>AO | **2.4986**<br>2.5527<br>2.5241 | 2.8550<br>**2.8157**<br>2.8309 | 3.0780<br>**2.9606**<br>3.0172 | **21,578,500**<br>21,589,890<br>21,583,854 | 0/30<br>0/30<br>0/30 | 438.1<br>439.2<br>441.4 |
| **lipa90b** (90)<br>*Opt: 12,490,441* | **PMX**<br>**WRAO**<br>AO | **0.0000**<br>**0.0000**<br>20.5034 | 20.3932<br>**20.3084**<br>21.0463 | 21.2608<br>**21.2606**<br>21.2799 | **12,490,441**<br>**12,490,441**<br>15,051,409 | **1/30**<br>**1/30**<br>0/30 | 369.3<br>370.3<br>371.5 |
| **lipa90a** (90)<br>*Opt: 360,630* | PMX<br>WRAO<br>**AO** | 0.6449<br>0.6358<br>**0.6275** | 0.6839<br>0.6816<br>**0.6803** | **0.7040**<br>0.7057<br>0.7151 | 362,956<br>362,923<br>**362,893** | 0/30<br>0/30<br>0/30 | 368.3<br>368.8<br>369.7 |
| **esc128** (128)<br>*Opt: 64* | **PMX**<br>**WRAO**<br>**AO** | **0.0000**<br>**0.0000**<br>**0.0000** | **0.0000**<br>**0.0000**<br>**0.0000** | **0.0000**<br>**0.0000**<br>**0.0000** | **64**<br>**64**<br>**64** | **30/30**<br>**30/30**<br>**30/30** | 700.4<br>700.0<br>702.6 |
| **chr25a** (25)<br>*Opt: 3,796* | PMX<br>**WRAO**<br>AO | 4.6364<br>**2.0547**<br>4.7945 | 14.6592<br>**12.8257**<br>13.4510 | 22.4446<br>**19.4942**<br>20.6006 | 3,972<br>**3,874**<br>3,978 | 0/30<br>0/30<br>0/30 | 33.8<br>34.3<br>34.2 |
| **bur26h** (26)<br>*Opt: 7,098,658* | **PMX**<br>WRAO<br>AO | **0.0000**<br>**0.0000**<br>**0.0000** | **0.0000**<br>0.0002<br>0.0001 | **0.0000**<br>0.0034<br>0.0034 | **7,098,658**<br>**7,098,658**<br>**7,098,658** | **30/30**<br>28/30<br>29/30 | 33.7<br>33.8<br>33.9 |

### 6.2. Performance Analysis on Large Instances (tai)

For the largest and most complex problems from the Taillard family (`tai256c`, `tai150b`, `tai100b`, `tai100a`), the algorithms behaved in a manner heavily dependent on the structure of the instance itself:
* For the **tai256c** and **tai100b** instances, the **WRAO** algorithm dominated the other approaches, achieving the lowest Best GAP (0.228% and 0.573%, respectively) and the best Mean GAP. This indicates that the Weighted Ranking Leader mechanism, combined with local search, effectively prevents premature convergence in highly multidimensional spaces.
* Conversely, for the **tai150b** instance, the standard **AO** variant performed slightly better, obtaining a Best GAP of 1.59%.
* It is worth noting that the differences in execution times between the algorithms on these instances are minimal (e.g., for `tai100b` all oscillate around 435–440 seconds). This demonstrates that the additional computational overhead of WRAO (e.g., related to continuous-discrete mapping) is effectively offset by faster convergence and parallelization in Numba.

### 6.3. Asymmetric and Specific Instances (lipa, esc)

* **esc128**: This instance proved to be relatively simple for all variants. PMX, WRAO, and AO found the global optimum in each of the 30 runs (Optimum Hits: 30/30), achieving a perfect GAP of 0.0%.
* **lipa90b and lipa90a**: The `lipa` family of instances is characterized by a very specific fitness landscape (strong asymmetry). In the `lipa90b` problem, both PMX and WRAO managed to find the optimal solution exactly 1 time out of 30 attempts, while the base AO version completely stagnated in local minima (Best GAP over 20%). This highlights the advantage of hybridization in challenging solution spaces.

### 6.4. Behavior on Small Instances (chr, bur)

For small-sized problems, interesting differences in exploitation capabilities (intensification) were observed:
* On the **chr25a** instance, a massive advantage of the **WRAO** algorithm was recorded. Its Best GAP was only 2.05%, leaving PMX (4.63%) and AO (4.79%) far behind. WRAO also achieved a much better average and worst result, demonstrating significant stability for this problem.
* On the other hand, on the **bur26h** instance, the discrete PMX operator showed 100% effectiveness (30/30 optimum hits), while the continuous algorithms (WRAO and AO) missed the optimal solution in a few isolated runs (28 and 29 hits out of 30, respectively).

### 6.5. Summary

The conducted experiments prove that the proposed **WRAO** algorithm is a highly competitive optimization method for the QAP. Its ability to fine-tune solutions in medium and large instances (such as `tai100b` or `tai256c`) is particularly impressive, breaking the barrier where the standard PMX operator and base AO fail to improve further. Although the discrete PMX matrix performs slightly more consistently on a few specific small instances, WRAO excels in its overall versatility and its superior capability to minimize the average error (Mean GAP).