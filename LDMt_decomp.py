import numpy as np

def ldmt_decomposition(A):
    A = np.array(A, dtype=float)
    n = A.shape[0]
    L = np.eye(n)
    D = np.eye(n)
    M = np.eye(n)

    for j in range(n):
        v = A[:j+1, j].copy()
        for k in range(j):
            v[k+1:j+1] = v[k+1:j+1] - v[k] * A[k+1:j+1, k]

        for i in range(j):
            A[i, j] = v[i] / A[i, i]

        A[j, j] = v[j]

        if A[j, j] <= 0:
            return None, None, None

        for k in range(j):
            A[j+1:, j] -= v[k] * A[j+1:, k]

        A[j+1:, j] = A[j+1:, j] / v[j]

    for i in range(n):
        for j in range(i):
            L[i, j] = A[i, j]
        D[i, i] = A[i, i]
        for j in range(i + 1, n):
            M[i, j] = A[i, j]

    for i in range(n):
        L[i, i] = 1.0
        M[i, i] = 1.0

    return L, D, M


def is_positive_definite(A):
    L, D, Mt = ldmt_decomposition(A)
    if L is None:
        return False
    return np.all(np.diag(D) > 0)


def main():
    A = np.array([[4, 2, 0],
                  [2, 5, 3],
                  [1, 7, 9]])

    L, D, M_T = ldmt_decomposition(A)

    print("Матрица A:\n", A)
    print("Матрица L:\n", L)
    print("Матрица D:\n", D)
    print("Матрица M.T:\n", M_T)

    if L is not None:
        print("L*D*M^T:\n", L @ D @ M_T)

    print("Матрица A положительно определенная?", "Да\n" if is_positive_definite(A) else "Нет\n")


if __name__ == "__main__":
    main()
