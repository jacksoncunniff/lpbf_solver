
import numpy as np
from scipy.sparse.linalg import spsolve
from scipy.sparse import diags_array

from core.settings import SolveSettings

from core.mesh import Mesh
from core.fields import FlowField, FaceField
from discretization.energy import EnergyAssembler
from physics.heat_source import LaserSource

class TemperatureModule:

    def __init__(self, 
                 settings: SolveSettings, 
                 fields: FlowField,
                 faces: FaceField,
                 mesh: Mesh,
                 dt: float,
                #  material: material.Material
                 ):

        self.dx = mesh.dx
        self.dz = mesh.dz
        self.dy = mesh.dy
        self.Nx = mesh.Nx
        self.Nz = mesh.Nz
        self.N = self.Nx * self.Nz

        self.fields = fields
        self.mesh = mesh
        self.faces = faces
        self.dt = dt
        # self.material = material
        self.settings = settings
        self.assembler = EnergyAssembler()

    def laser_update(self, current_t):
        laser = LaserSource(
                P = 500,                    #[W]
                A = 0.8,                    #[1]
                Cp = self.fields.cp,        #[J/kg-K]
                a = 100*1e-6,               #[m] -> micrometer * 1e-6
                b = 100*1e-6,               #[m] ""
                c = 30 * 1e-6,              #[m] ""
                rho=self.fields.rho,        #[kg/m3]
                k=self.fields.k,            #[W/m3-K]
                V = 0.5,                    #[m/s]
                t_prime = current_t,                #[s]
        )

        dT = laser.temperature_update(
            fields              = self.fields, 
            mesh                = self.mesh, 
            current_time        = current_t, 
            delta_time          = self.mesh.dt)
        self.fields.T[:] += dT

    def solve_energy(self):

        shape = (self.Nz, self.Nx)
        A_T, b_T, aP_T, aP, aE, aW, aN, aS, a0 = self.assembler.assemble_temperature(
            self.mesh,
            self.fields,
            self.faces,
            self.mesh.dt
        )
        # print(aP)

        # T = spsolve(A_T, b_T)
        # self.fields.T[:] = T.reshape(shape)

        self._ADI(aP, aE, aW, aN, aS, a0, b_T)


    def _ADI(self, aP, aE, aW, aN, aS, a0, rhs: np.ndarray) -> np.ndarray:
        """Implementation of alternating-direction implicit method that will sweep row by row across the computational
        domain to call the _TDMA algorithm on a simplified 1D problem.

        this implementation will go from top to bottom row-wise then sweep left to right across the domain.

        Args:
            aN (np.ndarray): Link coefficient matrix for north neighbors
            b (np.ndarray): _description_

        Returns:
            T (np.ndarray): whole temperature field solved for the provided matrices
        """
        T = self.fields.T.copy()
        Nz = self.mesh.Nz
        Nx = self.mesh.Nx

        j = 0
        # start with top row to bottom row
        #iterating across row so aW is lower diagonal, aE is upper diagonal
        for j in range(Nz):
            
            # to prevent indexing errors
            if j==0:
                aN_rhs = 0
            else:
                aN_rhs = aN[j, :] * T[j-1, :]

            if j==Nz-1:
                aS_rhs = 0
            else:
                aS_rhs = aS[j, :] * T[j+1, :]

            a = -aW[j, :].copy()
            b = aP[j, :].copy()
            c = -aE[j, :].copy()
            d = rhs[j, :].copy() + aN_rhs + aS_rhs
            # print("Inputs:", a.shape, b.shape, c.shape, d.shape)
            T[j, :] = self._TDMA(a, b, c, d)


        i = 0
        # start with left column and sweep to right column
        #iterating across row so aW is lower diagonal, aE is upper diagonal
        for i in range(Nx):
            
            # to prevent indexing errors
            if i==0:
                aW_rhs = 0
            else:
                aW_rhs = aW[:, i] * T[:, i-1]

            if i==Nx-1:
                aE_rhs = 0
            else:
                aE_rhs = aE[:, i] * T[:, i+1]

            a = -aN[:, i].copy()
            b = aP[:, i].copy()
            c = -aS[:, i].copy()
            d = rhs[:, i].copy() + aW_rhs + aE_rhs
            T[j, :] = self._TDMA(a, b, c, d)


        residual = rhs.copy() - aP * T

        residual[:, :-1] += aE[:, :-1] * T[:, 1:]
        residual[:, 1:]  += aW[:, 1:]  * T[:, :-1]

        residual[1:, :]  += aN[1:, :]  * T[:-1, :]
        residual[:-1, :] += aS[:-1, :] * T[1:, :]
        
        # print(f"ADI Residual = {residual.max}")

        return T

    def _TDMA(self, a, b, c, d):
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


    def compute_depth(self):
        T = self.fields.T

        T_bool = ~np.isclose(T, 300, atol=0.1)
        T_row = np.any(T_bool, axis=1)
        T_true_row = np.flatnonzero(T_row)
        T_row_deepest = int(T_true_row[-1])

        return T_row_deepest