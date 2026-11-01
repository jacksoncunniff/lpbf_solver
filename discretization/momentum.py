import numpy as np

from core.fields import FaceField, FlowField
from core.mesh import Mesh
from core.material import Material
from typing import Literal
MomentumComponent = Literal["uq", "w"]
BoundarySide = Literal["left", "bottom", "top", "right"]
Corners = Literal["NW", "NE", "SW", "SE"]

class MomentumAssembler:
    """
    Assemble the discretized 2D momentum equations.

    Uses cell-centered velocity fields and face-centered
    mass fluxes on a structured collocated FVM mesh.
    """

    def __init__(self, mesh: Mesh, material: Material, boundary_conditions):
        self.mesh = mesh
        self.material = material
        self.bc = boundary_conditions

        self.N = mesh.Nx * mesh.Nz
        self.Au = np.empty((self.N, self.N))
        self.bu = np.empty(self.mesh.Nx * self.mesh.Nz)
        self.Aw = np.empty((self.N, self.N))
        self.bw = np.empty(self.mesh.Nx * self.mesh.Nz)

    def assemble_u(self, fields: FlowField, faces: FaceField, dt):
        
        # Reset coeff matrix that is being returned
        self.Au = np.zeros((self.N, self.N), dtype=np.float64)
        self.bu = np.zeros((self.mesh.Nz, self.mesh.Nx))

        # Intermediate link coeff matrices that will each be determined individually
        shape = (self.mesh.Nz, self.mesh.Nx)
        aE = np.empty(shape)
        aW = np.empty(shape)
        aS = np.empty(shape)
        aN = np.empty(shape)
        aP = np.empty(shape)
        a0 = np.empty(shape)
        b = np.empty(shape)

        # Flux matrices
        Fe = faces.flux_q[:, 1:]    #flux_q is Nx+1 in columns, since it has fluxes stored a face for cells
        Fw = faces.flux_q[:, :-1]   #take flux face values except for final column (would represent east flux for final cell)
        Fs = faces.flux_w[1:, :]    
        Fn = faces.flux_w[:-1, :]

        # Diffusion matrices
        De = np.zeros(shape)
        Dw = np.zeros(shape)
        Ds = np.zeros(shape)
        Dn = np.zeros(shape)

        #interpolate interior mu face values for diffusion coeff.
        mu_q = 0.5 * (fields.mu[:, :-1] + fields.mu[:, 1:])
        mu_z = 0.5 * (fields.mu[:-1, :] + fields.mu[1:, :])

        De[:, :-1]  = 2 * self.mesh.dz / self.mesh.dx * mu_q
        Dw[:, 1:]   = 2 * self.mesh.dz / self.mesh.dx * mu_q
        Ds[:-1, :]  = self.mesh.dx / self.mesh.dz * mu_z
        Dn[1:, :]   = self.mesh.dx / self.mesh.dz * mu_z        

        # Boundary diffusion
        De[:, -1] = 4 * self.mesh.dz / self.mesh.dx * fields.mu[:, -1]          #De for right
        Dw[:, 0] = 4 * self.mesh.dz / self.mesh.dx * fields.mu[:, 0]            #Dw for left
        Ds[-1, :] = 2 * self.mesh.dx / self.mesh.dz * fields.mu[-1, :]      #Ds for bottom row
        Dn[0, :] = 2 * self.mesh.dx / self.mesh.dz * fields.mu[0, :]        #Dn for top surface
        # Dn[0, :] = 0    # contributes 0 with prescribed 

        ## Construct links with upwinding
        aE = De + np.maximum(-Fe, 0)
        aW = Dw + np.maximum(Fw, 0)
        aS = Ds + np.maximum(-Fs, 0)
        aN = Dn + np.maximum(Fn, 0)
        a0 = fields.rho * (self.mesh.dx * self.mesh.dz) / dt


        # This must happen before zero-out block because central coeff. needs the boundary cell diffusion contribution
        # it is then zero-ed out because there is no neighboring cell on the boundary
        aP: np.ndarray = aE + aW + aN + aS + (Fe - Fw + Fs - Fn) + a0

        ## Zero-out non neighbors   ~  should handle corners automatically too!
        aE[:, -1] = 0.0         # no right cell neighbors on last col
        aW[:, 0] = 0.0          # no left cell neighbors on first col
        aN[0, :] = 0.0          # no top cell neighbors on top row
        aS[-1, :] = 0.0         # no bottom cell neighbors on bottom row

        # RHS
        V = self.mesh.dx * self.mesh.dz * 1
        
        # transient contribution from a0 * w at time step
        self.bu += a0 * fields.uq_old - V * fields.dpdx

        # transposed momentum from u velocity
        dwdx_z = 0.5 * (fields.dwdx[:-1, :] + fields.dwdx[1:, :])
        tau_cross = mu_z * dwdx_z

        diffusion_stress = np.zeros_like(self.bu)
        diffusion_stress[:-1, :] += tau_cross * self.mesh.dx  # south
        diffusion_stress[1:, :]  -= tau_cross * self.mesh.dx  # north

        # source momentum from mushy zone
        source_momentum_sink = np.zeros_like(self.bu)
        source_momentum_sink = self._source_momentum(fields.liquid_frac, fields.uq_old) * V

        # # source surface tension from marangoni effect
        surface_tension = np.zeros_like(self.bu)
        surface_tension[0, :] = self._source_surf_tension(fields.dgamma_dT, fields.dTdx)

        self.bu += diffusion_stress + source_momentum_sink + surface_tension
        self.bu = self.bu.ravel()

        # print(f"aS: {aS}")
        aP = aP.ravel()
        aE = aE.ravel()
        aW = aW.ravel()
        aS = aS.ravel()
        aN = aN.ravel()

        Nx = self.mesh.Nx
        Nz = self.mesh.Nz
        k = np.arange(Nx * Nz)

        # Arrays of indices that have the named neighbors
        east  = k[k % Nx != Nx - 1]     # list of cells (k index) that have an east neighbor
        west  = k[k % Nx != 0]
        north = k[k >= Nx]
        south = k[k < Nx * Nz - Nx]

        # print(f"Aw shape = {self.Aw.shape}")
        # print(f"{aE}")
        self.Au[k, k]               = aP
        self.Au[east, east+1]       = -aE[east]
        self.Au[west, west-1]       = -aW[west]
        self.Au[north, north-Nx]    = -aN[north]
        self.Au[south, south+Nx]    = -aS[south]

        return self.Au, self.bu, aP

    def assemble_w(self, fields: FlowField, faces: FaceField, dt):
        """
        Construct A_w and b_w.
        
        This will utilize an array slicing centric approach, that is hopefully
        more compact then assemble_u()
        
        """

        # Reset coeff matrix that is being returned
        self.A_w = np.zeros((self.N, self.N), dtype=np.float64)
        self.bw = np.empty((self.mesh.Nz, self.mesh.Nx))

        # Intermediate link coeff matrices that will each be determined individually
        shape = (self.mesh.Nz, self.mesh.Nx)
        aE = np.empty(shape)
        aW = np.empty(shape)
        aS = np.empty(shape)
        aN = np.empty(shape)
        aP = np.empty(shape)
        a0 = np.empty(shape)
        b = np.empty(shape)

        # Flux matrices
        Fe = faces.flux_q[:, 1:]    #flux_q is Nx+1 in columns, since it has fluxes stored a face for cells
        Fw = faces.flux_q[:, :-1]   #take flux face values except for final column (would represent east flux for final cell)
        Fs = faces.flux_w[1:, :]    
        Fn = faces.flux_w[:-1, :]

        # Diffusion matrices
        De = np.zeros(shape)
        Dw = np.zeros(shape)
        Ds = np.zeros(shape)
        Dn = np.zeros(shape)

        #interpolate interior mu face values for diffusion coeff.
        mu_q = 0.5 * (fields.mu[:, :-1] + fields.mu[:, 1:])
        mu_z = 0.5 * (fields.mu[:-1, :] + fields.mu[1:, :])

        De[:, :-1]  = 2 * self.mesh.dz / self.mesh.dx * mu_q
        Dw[:, 1:]   = 2 * self.mesh.dz / self.mesh.dx * mu_q
        Ds[:-1, :]  = 4 * self.mesh.dx / self.mesh.dz * mu_z
        Dn[1:, :]   = 4 * self.mesh.dx / self.mesh.dz * mu_z        

        # Boundary diffusion
        De[:, -1] = self.mesh.dz / self.mesh.dx * fields.mu[:, -1] * 2          #De for right
        Dw[:, 0] = self.mesh.dz / self.mesh.dx * fields.mu[:, 0] * 2            #Dw for left
        Ds[-1, :] = 2 * self.mesh.dx / self.mesh.dz * fields.mu[-1, :] * 2      #Ds for bottom row
        
        # double check this formulation for the top surface.... TODO
        Dn[0, :] = 2 * self.mesh.dx / self.mesh.dz * fields.mu[0, :] * 2        #Dn for top surface

        ## Construct links with upwinding
        aE = De + np.maximum(-Fe, 0)
        aW = Dw + np.maximum(Fw, 0)
        aS = Ds + np.maximum(-Fs, 0)
        aN = Dn + np.maximum(Fn, 0)
        a0 = fields.rho * (self.mesh.dx * self.mesh.dz) / dt


        # This must happen before zero-out block because central coeff. needs the boundary cell diffusion contribution
        # it is then zero-ed out because there is no neighboring cell on the boundary
        aP: np.ndarray = aE + aW + aN + aS + (Fe - Fw + Fs - Fn) + a0


        ## Zero-out non neighbors   ~  should handle corners automatically too!
        aE[:, -1] = 0.0         # no right cell neighbors on last col
        aW[:, 0] = 0.0          # no left cell neighbors on first col
        aN[0, :] = 0.0          # no top cell neighbors on top row
        aS[-1, :] = 0.0         # no bottom cell neighbors on bottom row


        # RHS
        V = self.mesh.dx * self.mesh.dz * 1
        
        # transient contribution from a0 * w at time step
        self.bw += a0 * fields.w_old - V * fields.dpdz

        # transposed momentum from u velocity
        dudz_x = 0.5 * (fields.dudz[:, :-1] + fields.dudz[:, 1:])
        tau_cross = dudz_x * mu_q
        diffusion_stress = np.zeros_like(self.bw)
        diffusion_stress[:, 1:] -= tau_cross * self.mesh.dz # west contribuion
        diffusion_stress[:, :-1] += tau_cross * self.mesh.dz # east contribution

        # source momentum from mushy zone
        source_momentum_sink = np.zeros_like(self.bw)
        source_momentum_sink = self._source_momentum(fields.liquid_frac, fields.w)

        # # source surface tension from marangoni effect
        # surface_tension = np.zeros_like(self.bw)
        # surface_tension[0, :] = self._source_surf_tension(fields.dgamma_dT, fields.dTdx)

        self.bw += diffusion_stress + source_momentum_sink
        self.bw = self.bw.ravel()

        # print(f"aS: {aS}")
        aP = aP.ravel()
        aE = aE.ravel()
        aW = aW.ravel()
        aS = aS.ravel()
        aN = aN.ravel()

        Nx = self.mesh.Nx
        Nz = self.mesh.Nz
        k = np.arange(Nx * Nz)


        # Arrays of indices that have the named neighbors
        east  = k[k % Nx != Nx - 1]     # list of cells (k index) that have an east neighbor
        west  = k[k % Nx != 0]
        north = k[k >= Nx]
        south = k[k < Nx * Nz - Nx]

        # print(f"Aw shape = {self.Aw.shape}")
        # print(f"{aE}")
        self.Aw[k, k]               = aP
        self.Aw[east, east+1]       = -aE[east]
        self.Aw[west, west-1]       = -aW[west]
        self.Aw[north, north-Nx]    = -aN[north]
        self.Aw[south, south+Nx]    = -aS[south]

        return self.Aw, self.bw, aP

    def _source_momentum(self, e_vector, u_vector):
        # source momentum sink from the mushy zone
        # utilizes full vector operation

        G = 1       # permeability  TODO: UPDATE
        d = 1     # computational constant to avoid div by 0

        return -G * (1-e_vector)**2 / (e_vector**3 + d) * u_vector

    def _source_surf_tension(self, dgamma_dT: np.ndarray, dTdx_all: np.ndarray):

        dTdx_topRow = dTdx_all[0, :]

        return -dgamma_dT * dTdx_topRow * self.mesh.dx