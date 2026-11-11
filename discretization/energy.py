import numpy as np
from scipy.sparse import coo_array
from typing import Literal, Optional

from core.mesh import Mesh
from core.fields import FlowField, FaceField
from core.material import Material
from core.boundary_conditions import BoundaryCondition

class EnergyAssembler:


    def __init__(self):
        # nothing for now i guess
        pass


    def assemble_temperature(self, mesh: Mesh, fields: FlowField, faces: FaceField, dt): 

        lid_case = True
        shape = mesh.Nz, mesh.Nx
        dz = mesh.dz
        dx = mesh.dx

        b_T = np.zeros(shape)
        A_T = np.zeros(shape)

        aE = np.empty(shape)
        aW = np.empty(shape)
        aN = np.empty(shape)
        aS = np.empty(shape)
        aP = np.empty(shape)
        a0 = np.empty(shape)

        # Specific Heat Interpolation
        cp_q = 0.5 * (fields.cp[:, :-1] + fields.cp[:, 1:])
        cp_z = 0.5 * (fields.cp[:-1, :] + fields.cp[1:, :])

        # Thermal Conductivity Interpolation
        k_q = 0.5 * (fields.k[:, :-1] + fields.k[:, 1:])
        k_z = 0.5 * (fields.k[:-1, :] + fields.k[1:, :])

        # Diffusion matrices
        De = np.zeros(shape)
        Dw = np.zeros(shape)
        Ds = np.zeros(shape)
        Dn = np.zeros(shape)

        # Flux matrices
        Fe = cp_q * faces.flux_q[:, 1:]
        Fw = cp_q * faces.flux_q[:, :-1]
        Fs = cp_z * faces.flux_w[1:, :]
        Fn = cp_z * faces.flux_w[1:, :]

        # Calculate intermediate values
        De[:, :-1]  = k_q * dz / dx
        Dw[:, 1:]   = k_q * dz / dx
        Ds[:-1, :]  = k_z * dx / dz
        Dn[1:, :]   = k_z * dx / dz

        # Boundary Diffusion
        De[:, -1]  = 2 * k_q[:, -1] * dz / dx
        Dw[:, 0]   = 2 * k_q[:, 0]  * dz / dx
        Ds[-1, :]  = 2 * k_z[-1, :] * dx / dz

        b_T[:, -1]  = De[:, -1] * 300
        b_T[:, 0]   = Dw[:, 0]  * 300
        b_T[-1, :]  = Ds[0, :]  * 300

        h = 5 # UPDATE THIS LATER TODO
        # R_eff is Dn but with conductive and convection between cell and to ambient condition = 300K
        R_eff = np.zeros_like(b_T)
        R_eff[0, :] = dx / (1.0/h + (dz/2)/fields.k[0, :])
        b_T[0, :]   = R_eff * 300 # convection heat loss

        # Grid Peclet Numbers
        Pe = Fe / De
        Pw = Fw / Dw
        Ps = Fs / Ds
        Pn = Fn / Dn

        def scheme():
            # table 5.2 in patanakar, pg 95
            # can be used for future methods besides upwind
            return 1

        aE = De * scheme() + max(-Fe, 0)
        aW = Dw * scheme() + max(-Fw, 0)
        aS = Ds * scheme() + max(-Fs, 0)
        aN = Dn * scheme() + max(-Fn, 0)
        a0 = fields.rho * fields.cp * (dx * dz) / dt
        aP_T: np.ndarray = aE + aW + aN + aS + (Fe - Fw + Fs - Fn) + a0 + R_eff

        # Zero-out non neighbors
        aE[:, -1] = 0.0         # no right cell neighbors on last col
        aW[:, 0] = 0.0          # no left cell neighbors on first col
        aN[0, :] = 0.0          # no top cell neighbors on top row
        aS[-1, :] = 0.0         # no bottom cell neighbors on bottom row

        # RHS
        b_T += a0 * fields.T
        b_T = b_T.ravel()
        aP_ = aP.ravel()
        aE_ = aE.ravel()
        aW_ = aW.ravel()
        aS_ = aS.ravel()
        aN_ = aN.ravel()

        Nx = mesh.Nx
        Nz = mesh.Nz
        A_T = self._build_sparse_matrix(aE_, aW_, aS_, aN_, aP_, Nx, Nz)
        return A_T, b_T, aP_T, aP, aE, aW, aN, aS, a0


    def _build_sparse_matrix(self, aE, aW, aS, aN, aP, Nx, Nz):
        N = Nx * Nz
        k = np.arange(N)
        east  = k[k % Nx != Nx - 1]
        west  = k[k % Nx != 0]
        north = k[k >= Nx]
        south = k[k < N - Nx]

        rows = np.concatenate([
            k,
            east,
            west,
            north,
            south
        ])

        cols = np.concatenate([
            k,
            east+1,
            west-1,
            north-Nx,
            south+Nx
        ])

        values = np.concatenate([
            aP.ravel(),
            -aE.ravel()[east],
            -aW.ravel()[west],
            -aN.ravel()[north],
            -aS.ravel()[south]
        ])

        A = coo_array((values, (rows, cols)), shape=(N, N)).tocsr()
        return A
