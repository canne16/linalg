import numpy as np


def ldlt_decomposition(A):
    A = np.array(A, dtype=float)
    n = A.shape[0]
    L = np.eye(n, dtype=float)
    D = np.zeros((n, n), dtype=float)

    for j in range(n):
        D[j, j] = A[j, j] - np.sum(L[j, :j]**2 * np.diag(D)[:j])

        for i in range(j + 1, n):
            L[i, j] = (A[i, j] - np.sum(L[i, :j] * L[j, :j] * np.diag(D)[:j])) / D[j, j]

    return L, D


def ldlt_solve(A, b):
    if is_positive_definite(A):
    
        L, D = ldlt_decomposition(A)
        
        # Решаем Ly = b
        y = solve_triangular(L, b)
        
        # Решаем Dz = y
        z = solve_dioganal(D, y)
        
        # Решаем L^Tx = z
        x = solve_triangular(L.T, z, lower=False)
        return x

    else:
        return None

def is_positive_definite(A):
    L, D = ldlt_decomposition(A)
    if L is None:
        return False
    return np.all(np.diag(D) > 0)

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

def solve_dioganal(D, b):
    n = D.shape[0]
    b = np.asarray(b, dtype=D.dtype)
    x = np.zeros_like(b, dtype=D.dtype)

    for i in range(n):
        x[i] = b[i]
        x[i] /= D[i, i]

    return x


def main():
    A = np.array([[4, 2, 0, 0, 0],
                  [2, 7, 3, 0, 0],
                  [0, 3, 6, 0, 0],
                  [0, 0, 0, 9, 5],
                  [0, 0, 0, 5, 11]])

    L, D = ldlt_decomposition(A)
    
    b = np.array([1, -2, 0, 4, 5], dtype=np.float32)
    x = ldlt_solve(A, b)
    # Проверка
    Ax_b = np.dot(A, x) - b

    print("Матрица A:\n", A)
    print("Матрица L:\n", L)
    print("Матрица D:\n", D)
    print("Матрица L.T:\n", L.T)

    print("Решение x:", x)
    print("Проверка A*x - b:\n", Ax_b)
    
    if L is not None:
        print("L*D*L.T:\n", L @ D @ L.T)


if __name__ == "__main__":
    main()
