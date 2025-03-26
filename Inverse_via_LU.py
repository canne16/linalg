import numpy as np
from PA_LU_decomp import lu, lu_solve

def inverse_via_lu(A):

    n = A.shape[0]
    A_inv = np.zeros_like(A, dtype=float)
    
    for j in range(n):
        e_j = np.zeros(n)
        e_j[j] = 1.0
        
        x_j, *noth= lu_solve(A, e_j)
        
        A_inv[:, j] = x_j
    
    return A_inv


def main():

    A = np.array([[2, 1, 1],
                  [4, -6, 0],
                  [-2, 7, 2]], dtype=float)
    
    print("A:")
    print(A)
    
    # 1. Расчет обратной матрицы
    A_inv = inverse_via_lu(A)
    
    print("\nA_inv:")
    print(A_inv)
    
    # Проверка найденной матрицы
    identity_approx = A @ A_inv
    
    print("\nA * A_inv:")
    print(identity_approx)

if __name__ == "__main__":
    main()