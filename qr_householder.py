import numpy as np

# QR - разложение методом Хаусхолдера
def qr_householder(A):
    m, n = A.shape
    Q = np.eye(m)
    R = A.copy().astype(float)
    for k in range(n):
        x = R[k:, k]
        e = np.zeros_like(x)
        e[0] = np.linalg.norm(x)
        u = x - e
        v = u / np.linalg.norm(u)
        Hk = np.eye(m)
        Hk[k:, k:] -= 2.0 * np.outer(v, v)
        R = Hk @ R
        Q = Q @ Hk.T
    return Q, R

# решение задачи наименьших квадратов
def least_squares(A, b):
    Q, R = qr_householder(A)
    y = Q.T @ b
    x = np.linalg.solve(R[:A.shape[1], :], y[:A.shape[1]])
    return x

# поиск собственных векторов методом обратной итерации со сдвигом
def eigenvectors_inverse_power_shift(T, tol=1e-6, max_iter=1000):
    n = T.shape[0]
    eigvecs = np.zeros((n, n), dtype=complex)
    for i in range(n):
        mu = T[i, i]
        M = T - mu * np.eye(n)
        if np.linalg.cond(M) > 1e12:
            M += 1e-8 * np.eye(n)
        x = np.random.rand(n) + 1j * np.random.rand(n)
        x = x / np.linalg.norm(x)
        for _ in range(max_iter):
            y = np.linalg.solve(M, x)
            x_new = y / np.linalg.norm(y)
            if np.linalg.norm(x_new - x) < tol:
                break
            x = x_new
        eigvecs[:, i] = x
    return eigvecs


if __name__ == '__main__':
    # проверка разложения QR и МНК
    A = np.random.rand(6, 3)
    b = np.random.rand(6)
    x_ls = least_squares(A, b)
    print("Ошибка МНК:", np.linalg.norm(A @ x_ls - b))

    # собственные векторы треугольной матрицы с положительными диаг. эл.
    T = np.triu(np.random.rand(4, 4) + np.eye(4) * 5)
    v = eigenvectors_inverse_power_shift(T)
    for i in range(4):
        print("Ошибка СВ {}:".format(i), np.linalg.norm(T @ v[:, i] - T[i, i] * v[:, i]))

    # комплексные диаг. элементы
    Tc = np.triu(np.random.rand(3, 3) + 1j * np.random.rand(3, 3))
    for i in range(3): Tc[i, i] += 2 + 1j
    vc = eigenvectors_inverse_power_shift(Tc)
    for i in range(3):
        print("Ошибка комплексного СВ {}:".format(i), np.linalg.norm(Tc @ vc[:, i] - Tc[i, i] * vc[:, i]))
