import numpy as np
from dataclasses import dataclass, field
import numpy.typing as npt

from core.mesh import Mesh

FloatArray = npt.NDArray[np.float64]



@dataclass
class FlowField:
    """
    TODO docstring for fields classes
    """

    mesh: Mesh
    uq: np.ndarray
    w: np.ndarray
    p: np.ndarray
    T: np.ndarray
    mu: np.ndarray
    rho: np.ndarray
    liquid_frac: np.ndarray
    dgamma_dT: np.ndarray       # this will be a 1 x Nx array that contains the updated surface tension value 
                                # from the Materials class curve, for top row of cells only. calculated from dT of those cells

    uq_old: np.ndarray
    w_old: np.ndarray

    uq_star: np.ndarray
    w_star: np.ndarray
    p_prime: np.ndarray

    dpdx: np.ndarray
    dpdz: np.ndarray

    dudx: np.ndarray
    dwdx: np.ndarray
    

    dTdx: np.ndarray
    dTdz: np.ndarray

    def calculate_pressure_gradients(self):
        """
        Utilizes center differencing except on boundaries, where it has one sided derivatives
        **might need to increase the order for the boundary conditions
        """
        uq = self.uq
        w = self.w
        dx = self.mesh.dx
        dz = self.mesh.dz

        dpdx = np.zeros_like(uq)
        dpdz = np.zeros_like(uq)

        # Interior: second-order central differences
        dpdx[:, 1:-1] = (uq[:, 2:] - uq[:, :-2]) / (2 * dx)
        dpdz[1:-1, :] = (w[2:, :] - w[:-2, :]) / (2 * dz)

        # TODO - check if order needs to be raised on these
        # Left and right boundary-adjacent cells
        dpdx[:, 0] = (uq[:, 1] - uq[:, 0]) / dx
        dpdx[:, -1] = (uq[:, -1] - uq[:, -2]) / dx

        # Top and bottom boundary-adjacent cells
        dpdz[0, :] = (w[1, :] - w[0, :]) / dz
        dpdz[-1, :] = (w[-1, :] - w[-2, :]) / dz

        self.dpdx = dpdx
        self.dpdz = dpdz

    def calculate_temperature_gradients(self):
        """
        Utilizes center differencing except on boundaries, where it has one sided derivatives
        **might need to increase the order for the boundary conditions
        """
        T = self.T
        dx = self.mesh.dx
        dz = self.mesh.dz

        dTdx = np.zeros_like(T)
        dTdz = np.zeros_like(T)

        # Interior: second-order central differences
        dTdx[:, 1:-1] = (T[:, 2:] - T[:, :-2]) / (2 * dx)
        dTdz[1:-1, :] = (T[2:, :] - T[:-2, :]) / (2 * dz)

        # Left and right boundary-adjacent cells
        dTdx[:, 0] = (T[:, 1] - T[:, 0]) / dx
        dTdz[:, -1] = (T[:, -1] - T[:, -2]) / dx

        # Top and bottom boundary-adjacent cells
        dTdx[0, :] = (T[1, :] - T[0, :]) / dz
        dTdz[-1, :] = (T[-1, :] - T[-2, :]) / dz

        self.dTdx = dTdx
        self.dTdz = dTdz

    def calculate_liquid_frac(self):

        Ts = 900    # TODO - UPDATE PROPERTIES & OR REFERENCE MATERIALS SPECIFICALLY IN SEP CLASS
        Tl = 1500

        mask = self.T > Ts & self.T < Tl
        self.liquid_frac[self.T <= Ts] = 0
        self.liquid_frac[self.T >= Tl] = 1

        self.liquid_frac[mask] = (self.T[mask] - Ts) / (Tl - Ts)

    def _update_dwdx(self):

        ## Interior Cells -> Central Difference
        self.dwdx[:, 1:-1] = (self.w[:,:-2] - self.w[:,2:]) / (2*self.mesh.dx)
    
        ## Left / Right Edge -> 1-sided 2nd order approximation
        self.dwdx[:, 0] = (-3 * self.w[:, 0] + 4 * self.w[:, 1] - self.w[:, 2]) / (2 * self.mesh.dx)
        self.dwdx[:, -1] = (-3 * self.w[:, -1] + 4 * self.w[:, -2] - self.w[:, -3]) / (2 * self.mesh.dx)
        pass

    def _update_dudx(self):

        ## Interior Cells -> Central Difference
        self.dudx[:, 1:-1] = (self.uq[:,:-2] - self.uq[:,2:]) / (2*self.mesh.dx)
    
        ## Left / Right Edge -> 1-sided 2nd order approximation
        self.dudx[:, 0] = (-3 * self.uq[:, 0] + 4 * self.uq[:, 1] - self.uq[:, 2]) / (2 * self.mesh.dx)
        self.dudx[:, -1] = (-3 * self.uq[:, -1] + 4 * self.uq[:, -2] - self.uq[:, -3]) / (2 * self.mesh.dx)
        pass


@dataclass
class FaceField:
    """
    """

    flux_q: FloatArray
    flux_w: FloatArray

