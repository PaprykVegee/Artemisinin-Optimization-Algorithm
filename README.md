\# QAP Optimization using Artemisinin Optimization Algorithm (AO \& WRAO)



This repository contains a Python implementation of the \*\*Artemisinin Optimization Algorithm (AO)\*\* and its variant, the \*\*Weighted Ranking Artemisinin Optimization (WRAO)\*\*, specifically adapted for the \*\*Quadratic Assignment Problem (QAP)\*\*.



\## 📌 Project Overview

The objective of this project is to implement and evaluate a metaheuristic approach to solve discrete permutation problems. Since the original AO algorithm is designed for continuous search spaces, this implementation focuses on adapting the algorithm's mechanics to handle permutations effectively.



\### Key Objectives:

\* \*\*Implementation\*\*: Developing both standard \*\*AO\*\* and \*\*WRAO\*\* versions.

\* \*\*Adaptation\*\*: Transforming continuous optimization steps into the permutation domain (e.g., using Random Order Value mapping).

\* \*\*Evaluation\*\*: Benchmarking the performance against the \*\*QAPLIB\*\* dataset.



\---



\## 🧠 The Quadratic Assignment Problem (QAP)

The \*\*Quadratic Assignment Problem\*\* is a fundamental combinatorial optimization problem. It involves assigning $n$ facilities to $n$ locations in a way that minimizes the total cost associated with the flows between facilities and the distances between locations.



\### Mathematical Formulation (Matrix Version)

We aim to find a permutation $p$ that minimizes the following objective function:



$$min \\sum\_{i=1}^{n} \\sum\_{j=1}^{n} a\_{ij} \\cdot b\_{p(i)p(j)}$$



Where:

\* $n$: The size of the instance (number of facilities/locations).

\* $A = \[a\_{ij}]$: The \*\*Flow Matrix\*\*, where $a\_{ij}$ is the flow between facility $i$ and facility $j$.

\* $B = \[b\_{ij}]$: The \*\*Distance Matrix\*\*, where $b\_{ij}$ is the distance between location $i$ and location $j$.

\* $p$: A permutation of the set $\\{1, \\dots, n\\}$, where $p(i)$ represents the location assigned to facility $i$.



The core goal is to ensure that facilities with high interaction (high flow) are placed in locations that are close to each other (low distance).



\---



\## 📂 Data Structure \& Scenarios

The project utilizes instances from the \*\*QAPLIB\*\* database. The repository is structured as follows:



```text

└───Scenarios

&#x20;   ├───Burkard        # Instances from R.E. Burkard

&#x20;   ├───Christofides   # Instances from N. Christofides

&#x20;   ├───Elshafei       # Instances from A.N. Elshafei

&#x20;   ├───Hadley         # Instances from S.W. Hadley

&#x20;   ├───Krarup         # Instances from J. Krarup

&#x20;   └───Lipa           # Instances from P. Lipa



### 📄 Data Formats



\#### Input File Format (`.dat`)

Each scenario file contains the problem data structured as follows:

\* \*\*n\*\*: The size of the instance (number of facilities/locations).

\* \*\*Matrix A\*\*: The first $n \\times n$ matrix (typically representing flows).

\* \*\*Matrix B\*\*: The second $n \\times n$ matrix (typically representing distances).



\#### Solution File Format (`.sln`)

The provided solution files serve as benchmarks to verify the algorithm's accuracy:

\* \*\*n\*\*: The size of the instance.

\* \*\*sol\*\*: The objective function value. This is either the mathematically proven \*\*optimal\*\* value or the \*\*best-known\*\* value found by advanced heuristics.

\* \*\*p\*\*: The corresponding permutation (the specific assignment of facilities to locations).



> \*\*Note on Benchmarks:\*\* The reference solutions in QAPLIB were historically derived using various state-of-the-art heuristics, including:

> \* \*\*Ant Systems (ANT)\*\*

> \* \*\*Genetic Hybrids (GEN)\*\*

> \* \*\*Tabu Search (TS)\*\*

> \* \*\*Simulated Annealing (SIM)\*\*

> \* \*\*GRASP\*\* and \*\*Scatter Search\*\*

