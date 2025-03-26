import numpy as np


# LU-разложение
def lu(a):
    a1 = np.asarray(a)
    if a1.ndim != 2:
        raise ValueError('The input array must be two-dimensional.')

    m, n = a1.shape
    k = min(m, n)

    a1 = a1.copy(order='C')

    p = np.empty(m, dtype=np.int32)
    u = np.zeros([k, k], dtype=a1.dtype)
    L, U, P = lu_dispatcher(a1, u, p)
    
    Pa = np.zeros([m, m], dtype=np.float32)
    Pa[np.arange(m), P] = 1
    P = Pa

    return (P, L, U)


# Реализация основного алгоритма LU разложения
def lu_dispatcher(a1, u, p):
    n = a1.shape[0]
    p[:] = np.arange(n)
    
    for k in range(n - 1):
        max_row = np.argmax(abs(a1[k:n, k])) + k
        if max_row != k:
            a1[[k, max_row], :] = a1[[max_row, k], :]
            p[[k, max_row]] = p[[max_row, k]]
        
        for i in range(k + 1, n):
            a1[i, k] /= a1[k, k]
            a1[i, k + 1:] -= a1[i, k] * a1[k, k + 1:]
    
    np.copyto(u, np.triu(a1))

    l = np.tril(a1, -1) + np.eye(n)
    
    return l, u, p


# Решение системы с помошью разложения PA = LU
def lu_solve(A, b):
    P, L, U = lu(A)
    
    Pb = np.dot(P, b)
    
    # Решаем Ly = Pb
    y = solve_triangular(L, Pb)
    
    # Решаем Ux = y
    x = solve_triangular(U, y, lower=False)
    
    return x, P, L, U

# Решение треугольной матрицы
def solve_triangular(L, Pb, lower=True):
    if not lower:
        return solve_upper_triangular(L, Pb)

    m, n = L.shape
    Pb = np.asarray(Pb, dtype=L.dtype)
    
    x = np.zeros_like(Pb, dtype=L.dtype)

    for i in range(m):
        x[i] = Pb[i]
        for j in range(i):
            x[i] -= L[i, j] * x[j]
        x[i] /= L[i, i]

    return x

# Решение верхнетреугольной матрицы
def solve_upper_triangular(U, y):
    m, n = U.shape
    y = np.asarray(y, dtype=U.dtype)
    x = np.zeros_like(y, dtype=U.dtype)

    for i in range(m - 1, -1, -1):
        x[i] = y[i]
        for j in range(i + 1, m):
            x[i] -= U[i, j] * x[j]
        x[i] /= U[i, i]

    return x



def main():
    # Пример использования
    A = np.array([[-1, -5,  1], 
                [-1,  1,  1], 
                [-4, -1, -2]], dtype=np.float32)

    b = np.array([1, -2, 0], dtype=np.float32)

    x, P, L, U = lu_solve(A, b)

    # Проверка
    Ax_b = np.dot(A, x) - b

    print("Матрица A:\n", A)
    print("Решение x:", x)
    print("Матрица перестановки P:\n", P)
    print("Нижнетреугольная матрица L:\n", L)
    print("Верхнетреугольная матрица U:\n", U)
    print("Проверка A*x - b:\n", Ax_b)


if __name__ == "__main__":
    main()