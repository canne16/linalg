import numpy as np


def ldmt_decomposition(A):
    A = np.array(A, dtype=float)
    n = A.shape[0]
    L = np.eye(n)
    D = np.zeros((n, n))

    for i in range(n):
        D[i, i] = A[i, i] - np.sum(L[i, :i]**2 * np.diag(D)[:i])
        for j in range(i+1, n):
            L[j, i] = (A[j, i] - np.sum(L[j, :i] * L[i, :i] * np.diag(D)[:i])) / D[i, i]

    return L, D


def is_positive_definite(A):
    L, D = ldmt_decomposition(A)
    if L is None:
        return False
    return np.all(np.diag(D) > 0)


def main():
    A = np.array([[4, 2, 1],
                  [2, 5, 3],
                  [1, 3, 9]])

    L, D = ldmt_decomposition(A)
    print("Матрица A:\n", A)
    print("Матрица L:\n", L)
    print("Матрица D:\n", D)

    print("L*D*Mt:\n", L@D@L.T)

    print("Матрица A положительно определенная?", "Да" if is_positive_definite(A) else "Нет")


if __name__ == "__main__":
    main()
