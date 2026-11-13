import numpy as np
from numba import jit
"""
General file for the alternating-direction-implicit method that will be used to solve for field variables
"""

@jit(nopython=True)
def ADI(aP, aE, aW, aN, aS, rhs: np.ndarray, field_of_interest: np.ndarray, Nx, Nz) -> np.ndarray:
        """Implementation of alternating-direction implicit method that will sweep row by row across the computational
        domain to call the _TDMA algorithm on a simplified 1D problem.

        this implementation will go from top to bottom row-wise then sweep left to right across the domain.

        Args:
            aN (np.ndarray): Link coefficient matrix for north neighbors
            b (np.ndarray): _description_

        Returns:
            T (np.ndarray): whole temperature field solved for the provided matrices
        """
        F = field_of_interest.copy()

        j = 0
        # start with top row to bottom row
        #iterating across row so aW is lower diagonal, aE is upper diagonal
        for j in range(Nz):
            
            # to prevent indexing errors
            if j==0:
                aN_rhs = np.zeros_like(aN[j, :])
            else:
                aN_rhs = aN[j, :] * F[j-1, :]

            if j==Nz-1:
                aS_rhs = np.zeros_like(aN[j, :])
            else:
                aS_rhs = aS[j, :] * F[j+1, :]

            a = -aW[j, :].copy()
            b = aP[j, :].copy()
            c = -aE[j, :].copy()
            d = rhs[j, :].copy() + aN_rhs + aS_rhs
            # print("Inputs:", a.shape, b.shape, c.shape, d.shape)
            F[j, :] = _TDMA(a, b, c, d)


        i = 0
        # start with left column and sweep to right column
        #iterating across row so aW is lower diagonal, aE is upper diagonal
        for i in range(Nx):
            
            # to prevent indexing errors
            if i==0:
                aW_rhs = np.zeros_like(aW[:, i])
            else:
                aW_rhs = aW[:, i] * F[:, i-1]

            if i==Nx-1:
                aE_rhs = np.zeros_like(aW[:, i])
            else:
                aE_rhs = aE[:, i] * F[:, i+1]

            a = -aN[:, i].copy()
            b = aP[:, i].copy()
            c = -aS[:, i].copy()
            d = rhs[:, i].copy() + aW_rhs + aE_rhs
            F[:, i] = _TDMA(a, b, c, d)


        # residual = rhs.copy() - aP * F

        # residual[:, :-1] += aE[:, :-1] * F[:, 1:]
        # residual[:, 1:]  += aW[:, 1:]  * F[:, :-1]

        # residual[1:, :]  += aN[1:, :]  * F[:-1, :]
        # residual[:-1, :] += aS[:-1, :] * F[1:, :]
        
        # print(f"ADI Residual = {residual.max}")

        return F

@jit(nopython=True)
def _TDMA(a, b, c, d):
    """Tridiagonal Matrix Algorithm solve for a single row of the computational mesh. This algorithm requires equal
    length of each vector a, b, c, d, x and for the off-main diagonals (a & c) to contain a padded 0 in the appropriate
    direction.

    This TDMA call does not have any considerations of boundary conditions or ghost points for the computational domain
    it only considers the row or column of the mesh passed.

    Current implementation follows https://www.cfd-online.com/Wiki/Tridiagonal_matrix_algorithm_-_TDMA_(Thomas_algorithm)
    Future implementations to consider boundary conditions according to Cassel 2021.

    Args:
        a (np.ndarray): lower diagonal coefficient
        b (np.ndarray): main diagonal coefficients
        c (np.ndarray): upper diagonal coefficients
        d (np.ndarray): RHS vector


    Returns:
        x: solution vector
    """
    x = np.zeros_like(d)
    for i in range(1, len(b)):
        m = a[i] / b[i-1]
        b[i] = b[i] - m * c[i-1]
        d[i] = d[i] - m * d[i-1]
    x[-1] = d[-1] / b[-1]
    for i in range(len(x) -2, -1, -1):
        x[i] = (d[i] - c[i] * x[i+1]) / b[i]
    return x


