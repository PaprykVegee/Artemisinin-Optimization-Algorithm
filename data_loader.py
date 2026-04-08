import numpy as np

def load_instance(file_path):
    with open(file_path, 'r') as f:
        data = f.read().split()
    
    n = int(data[0])

    matrix_a = np.array(data[1 : n*n + 1], dtype=float).reshape((n, n))
    matrix_b = np.array(data[n*n + 1 : 2*n*n + 1], dtype=float).reshape((n, n))
    
    return n, matrix_a, matrix_b


def load_solution(file_path):
    with open(file_path, 'r') as f:
        data = f.read().split()
    
    n = int(data[0])
    optimal = data[1]
    solution = np.array(data[2 : n + 1], dtype=int)
    
    return n, optimal, solution
