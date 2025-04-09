import numpy as np
from math import sqrt


def сhol_decomposition(A):
    A = np.array(A, dtype=float)
    n = A.shape[0]
    L = np.zeros_like(A)

    L[0][0] = sqrt(A[0][0])
    
    for j in range(1, n):
        L[j][0] = A[j][0] / L[0][0]

    for i in range(1, n):
        sum_diag = sum(L[i][p1] ** 2 for p1 in range(i))
        assert A[i][i] - sum_diag > 0, "Матрица A не положительно определена"
        L[i][i] = sqrt(A[i][i] - sum_diag)
        
        for j in range(i + 1, n):
            sum_off_diag = sum(L[i][p2] * L[j][p2] for p2 in range(i))
            L[j][i] = (A[j][i] - sum_off_diag) / L[i][i]

    return L


def chol_solve(A, b):
    L = сhol_decomposition(A)
    
    # Решаем Ly = b
    y = solve_triangular(L, b)
    
    # Решаем L^Tx = y
    x = solve_triangular(L.T, y, lower=False)
    
    return x

def solve_triangular(L, b, lower=True):
    if not lower:
        return solve_upper_triangular(L, b)

    m, n = L.shape
    b = np.asarray(b, dtype=L.dtype)
    
    x = np.zeros_like(b, dtype=L.dtype)

    for i in range(m):
        x[i] = b[i]
        for j in range(i):
            x[i] -= L[i, j] * x[j]
        x[i] /= L[i, i]

    return x

def solve_upper_triangular(A, y):
    m, n = A.shape
    y = np.asarray(y, dtype=A.dtype)
    x = np.zeros_like(y, dtype=A.dtype)

    for i in range(m - 1, -1, -1):
        x[i] = y[i]
        for j in range(i + 1, m):
            x[i] -= A[i, j] * x[j]
        x[i] /= A[i, i]

    return x


def main():
    A = np.array([[4, 2, 1],
                  [2, 5, 3],
                  [1, 3, 9]])

    b = np.array([1, -2, 0], dtype=np.float32)
    x = chol_solve(A, b)
    # Проверка
    Ax_b = np.dot(A, x) - b

    assert np.all(A == A.T), "Матрица A не симметрична" 
    
    L = сhol_decomposition(A)

    print("Матрица A:\n", A)
    print("Решение x:", x)
    print("Нижнетреугольная L:\n", L)
    print("L@L^T:\n", L @ L.T)
    print("Проверка A*x - b:\n", Ax_b)


if __name__ == "__main__":
    main()