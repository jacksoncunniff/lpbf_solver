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

    def assemble_u_old(self, fields: FlowField, faces: FaceField, dt):
        """
        
        Construct and return A_u and b_u to solve u_mom iteratively elsewhere.
        
        """

        self.Au = np.empty((self.N, self.N))
        self.bu = np.empty(self.mesh.Nx * self.mesh.Nz)

        Nz = self.mesh.Nz
        Nx = self.mesh.Nx
        V = self.mesh.dx * self.mesh.dz
        e_vector = fields.liquid_frac.flatten()
        
        # INTERIOR CELLS ONLY
        for j in range(1, Nz-1):
            for i in range(1, Nx-1):
                aP, aE, aW, aN, aS, a0 = self._face_coefficients_interoir(fields, faces, i, j, "uq", dt=dt)
                k = self._ij_to_k(i, j)
                print(self.Au.shape)
                self.Au[k, k] = aP
                self.Au[k, k+1] = -aE
                self.Au[k, k-1] = -aW
                self.Au[k, k - Nx] = -aN
                self.Au[k, k + Nx] = -aS
                
                # Right-hand side
                mu_b = 0.5 * (fields.mu[j+1, i] + fields.mu[j, i])
                mu_t = 0.5 * (fields.mu[j-1, i] + fields.mu[j, i])

                dwdx_b = 0.5 * (fields.dwdx[j, i] + fields.dwdx[j + 1, i])
                dwdx_t = 0.5 * (fields.dwdx[j, i] + fields.dwdx[j - 1, i])

                diffusion_stresses = mu_b * dwdx_b - mu_t * dwdx_t
                # momentum_source =                       #TODO

                self.bu[k] = (a0 * fields.uq_old[j, i] - V * fields.dpdx[j, i]) + diffusion_stresses * self.mesh.dx

        # BOTTOM CELLS ONLY
        j=Nz-1
        for i in range(1, Nx-1):
            aP, aE, aW, aN, aS, a0 = self._boundary_coefficients(fields, faces, i, j, dt, "uq", boundarySide="bottom")
            k = self._ij_to_k(i, j)
            self.Au[k, k] = aP
            self.Au[k, k+1] = -aE
            self.Au[k, k-1] = -aW
            self.Au[k, k - Nx] = -aN
            # self.Au[k, k + Nx] = -aS
            
            # Right-hand side
            mu_b = 0
            mu_t = 0.5 * (fields.mu[j-1, i] + fields.mu[j, i])

            # utilizes old w velocity values (previous iteration)

            dwdx_b = 0
            dwdx_t = 0.5 * (fields.dwdx[j, i] + fields.dwdx[j-1, i])

            diffusion_stresses = mu_b * dwdx_b - mu_t * dwdx_t
            # momentum_source =                       #TODO

            self.bu[k] = (a0 * fields.uq_old[j, i] - V * fields.dpdx[j, i]) + diffusion_stresses * self.mesh.dx

        # LEFT CELLS ONLY
        i=0
        for j in range(1, Nz-1):
            aP, aE, aW, aN, aS, a0 = self._boundary_coefficients(fields, faces, i, j, dt, "uq", boundarySide="left")
            k = self._ij_to_k(i, j)
            self.Au[k, k] = aP
            self.Au[k, k+1] = -aE
            # self.Au[k, k-1] = -aW
            self.Au[k, k - Nx] = -aN
            self.Au[k, k + Nx] = -aS
            
            # Right-hand side
            mu_b = 0.5 * (fields.mu[j+1, i] + fields.mu[j, i])
            mu_t = 0.5 * (fields.mu[j-1, i] + fields.mu[j, i])

            dwdx_b = 0.5 * (fields.dwdx[j, i] + fields.dwdx[j + 1, i])
            dwdx_t = 0.5 * (fields.dwdx[j, i] + fields.dwdx[j - 1, i])

            diffusion_stresses = mu_b * dwdx_b - mu_t * dwdx_t
            # momentum_source =                       #TODO

            self.bu[k] = (a0 * fields.uq_old[j, i] - V * fields.dpdx[j, i]) + diffusion_stresses * self.mesh.dx

        # RIGHT CELLS ONLY
        i = Nx-1
        for j in range(1, Nz-1):
            aP, aE, aW, aN, aS, a0 = self._boundary_coefficients(fields, faces, i, j, dt, "uq", boundarySide="right")
            k = self._ij_to_k(i, j)
            self.Au[k, k] = aP
            # self.Au[k, k+1] = -aE
            self.Au[k, k-1] = -aW
            self.Au[k, k - Nx] = -aN
            self.Au[k, k + Nx] = -aS
            
            # Right-hand side
            mu_b = 0.5 * (fields.mu[j+1, i] + fields.mu[j, i])
            mu_t = 0.5 * (fields.mu[j-1, i] + fields.mu[j, i])

            # utilizes old w velocity values (previous iteration)

            dwdx_b = 0.5 * (fields.dwdx[j, i] + fields.dwdx[j + 1, i])
            dwdx_t = 0.5 * (fields.dwdx[j, i] + fields.dwdx[j - 1, i])

            diffusion_stresses = mu_b * dwdx_b - mu_t * dwdx_t

            self.bu[k] = (a0 * fields.uq_old[j, i] - V * fields.dpdx[j, i]) + diffusion_stresses * self.mesh.dx


        # TOP CELLS ONLY
        j=0
        for i in range(1, Nx-1):
            aP, aE, aW, aN, aS, a0 = self._boundary_coefficients(fields, faces, i, j, dt, "uq", boundarySide="top")
            k = self._ij_to_k(i, j)
            self.Au[k, k] = aP
            self.Au[k, k+1] = -aE
            self.Au[k, k-1] = -aW
            # self.Au[k, k - Nx] = -aN
            self.Au[k, k + Nx] = -aS
            
            # Right-hand side
            mu_b = 0.5 * (fields.mu[j+1, i] + fields.mu[j, i])
            mu_t = 0

            dwdx_b = 0.5 * (fields.dwdx[j, i] + fields.dwdx[j+1, i]) 
            dwdx_t = 0

            diffusion_stresses = mu_b * dwdx_b - mu_t * dwdx_t

            self.bu[k] = (a0 * fields.uq_old[j, i] - V * fields.dpdx[j, i]) + diffusion_stresses * self.mesh.dx

    # NW CELL (TOP LEFT CORNER)
        j=0
        i=0
        aP, aE, aW, aN, aS, a0 = self._corner_coefficients(fields, faces, i, j, dt, "uq", corner="NW")
        k = self._ij_to_k(i, j)
        self.Au[k, k] = aP
        self.Au[k, k+1] = -aE
        # self.Au[k, k-1] = -aW
        # self.Au[k, k - Nx] = -aN
        self.Au[k, k + Nx] = -aS
        
        # Right-hand side
        mu_b = 0.5 * (fields.mu[j+1, i] + fields.mu[j, i])
        # mu_t = 0.5 * (fields.mu[j-1, i] + fields.mu[j, i])
        mu_t = 0

        dwdx_b = 0.5 * (fields.dwdx[j, i] + fields.dwdx[j+1, i]) 
        # dwdx_t = 0.5 * (fields.dwdx[j, i] + fields.dwdx[j-1, i]) 
        dwdx_t = 0      # boundary condition -> w=0 on top

        diffusion_stresses = mu_b * dwdx_b - mu_t * dwdx_t

        self.bu[k] = (a0 * fields.uq_old[j, i] - V * fields.dpdx[j, i]) + diffusion_stresses * self.mesh.dx


    # NE CELL (TOP RIGHT CORNER)
        j=0
        i=Nx-1
        aP, aE, aW, aN, aS, a0 = self._corner_coefficients(fields, faces, i, j, dt, "uq", corner="NE")
        k = self._ij_to_k(i, j)
        self.Au[k, k] = aP
        # self.Au[k, k+1] = -aE
        self.Au[k, k-1] = -aW
        # self.Au[k, k - Nx] = -aN
        self.Au[k, k + Nx] = -aS
        
        # Right-hand side
        mu_b = 0.5 * (fields.mu[j+1, i] + fields.mu[j, i])
        # mu_t = 0.5 * (fields.mu[j-1, i] + fields.mu[j, i])
        mu_t = 0

        dwdx_b = 0.5 * (fields.dwdx[j, i] + fields.dwdx[j+1, i]) 
        # dwdx_t = 0.5 * (fields.dwdx[j, i] + fields.dwdx[j-1, i]) 
        dwdx_t = 0      # boundary condition -> w=0 on top

        diffusion_stresses = mu_b * dwdx_b - mu_t * dwdx_t

        self.bu[k] = (a0 * fields.uq_old[j, i] - V * fields.dpdx[j, i]) + diffusion_stresses * self.mesh.dx
    
    # SW CELL (BOTTOM LEFT CORNER)
        j=Nz-1
        i=0
        aP, aE, aW, aN, aS, a0 = self._corner_coefficients(fields, faces, i, j, dt, "uq", corner="SW")
        k = self._ij_to_k(i, j)
        self.Au[k, k] = aP
        self.Au[k, k+1] = -aE
        # self.Au[k, k-1] = -aW
        self.Au[k, k - Nx] = -aN
        # self.Au[k, k + Nx] = -aS
        
        # Right-hand side
        mu_b = 0
        mu_t = 0.5 * (fields.mu[j-1, i] + fields.mu[j, i])
        # mu_t = 0

        # dwdx_b = 0.5 * (fields.dwdx[j, i] + fields.dwdx[j+1, i]) 
        dwdx_t = 0.5 * (fields.dwdx[j, i] + fields.dwdx[j-1, i]) 
        dwdx_b = 0      # boundary condition

        diffusion_stresses = mu_b * dwdx_b - mu_t * dwdx_t

        self.bu[k] = (a0 * fields.uq_old[j, i] - V * fields.dpdx[j, i]) + diffusion_stresses * self.mesh.dx
    
    
    # SE CELL (BOTTOM RIGHT CORNER)
        j=Nz-1
        i=Nx-1
        aP, aE, aW, aN, aS, a0 = self._corner_coefficients(fields, faces, i, j, dt, "uq", corner="SE")
        k = self._ij_to_k(i, j)
        self.Au[k, k] = aP
        # self.Au[k, k+1] = -aE
        self.Au[k, k-1] = -aW
        self.Au[k, k - Nx] = -aN
        # self.Au[k, k + Nx] = -aS
        
        # Right-hand side
        mu_b = 0
        mu_t = 0.5 * (fields.mu[j-1, i] + fields.mu[j, i])


        # dwdx_b = 0.5 * (fields.dwdx[j, i] + fields.dwdx[j+1, i]) 
        dwdx_t = 0.5 * (fields.dwdx[j, i] + fields.dwdx[j-1, i]) 
        dwdx_b = 0

        diffusion_stresses = mu_b * dwdx_b - mu_t * dwdx_t

        self.bu[k] = (a0 * fields.uq_old[j, i] - V * fields.dpdx[j, i]) + diffusion_stresses * self.mesh.dx
    

        ## Source Momentum Sink Contribution
        u_old_vector = fields.uq_old.flatten()
        self.bu += self._source_momentum(e_vector, u_old_vector)

        ## Surface Tension Contribution
        self.bu[:self.mesh.Nx] += self._source_surf_tension(fields.dgamma_dT, fields.dTdx)

        return self.Au, self.bu, self.Au.diagonal()

    def assemble_u(self, fields: FlowField, faces: FaceField, dt):
        
        # Reset coeff matrix that is being returned
        self.Au = np.zeros((self.N, self.N), dtype=np.float64)
        self.bu = np.zeros((self.mesh.Nz, self.mesh.Nx))

        # Intermediate link coeff matrices that will each be determined individually
        shape = (self.mesh.Nz, self.mesh.Ny)
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
        De[:, -1] = self.mesh.dz / self.mesh.dx * fields.mu[:, -1] / 2          #De for right
        Dw[:, 0] = self.mesh.dz / self.mesh.dx * fields.mu[:, 0] / 2            #Dw for left
        Ds[-1, :] = 2 * self.mesh.dx / self.mesh.dz * fields.mu[-1, :] / 2      #Ds for bottom row
        Dn[0, :] = 2 * self.mesh.dx / self.mesh.dz * fields.mu[0, :] / 2        #Dn for top surface

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
        dudz_x = 0.5 * (fields.dudz[:, :-1] + fields.dudz[:, 1:])
        tau_cross = dudz_x * mu_q
        diffusion_stress = np.zeros_like(self.bu)
        diffusion_stress[:, 1:] -= tau_cross * self.mesh.dz # west contribuion
        diffusion_stress[:, :-1] += tau_cross * self.mesh.dz # east contribution

        # source momentum from mushy zone
        source_momentum_sink = np.zeros_like(self.bu)
        source_momentum_sink = self._source_momentum(fields.liquid_frac, fields.w)

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
        shape = (self.mesh.Nz, self.mesh.Ny)
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

        De[:, :-1]  = self.mesh.dz / self.mesh.dx * mu_q
        Dw[:, 1:]   = self.mesh.dz / self.mesh.dx * mu_q
        Ds[:-1, :]  = 2 * self.mesh.dx / self.mesh.dz * mu_z
        Dn[1:, :]   = 2 * self.mesh.dx / self.mesh.dz * mu_z        

        # Boundary diffusion
        De[:, -1] = self.mesh.dz / self.mesh.dx * fields.mu[:, -1] / 2          #De for right
        Dw[:, 0] = self.mesh.dz / self.mesh.dx * fields.mu[:, 0] / 2            #Dw for left
        Ds[-1, :] = 2 * self.mesh.dx / self.mesh.dz * fields.mu[-1, :] / 2      #Ds for bottom row
        
        # double check this formulation for the top surface.... TODO
        Dn[0, :] = 2 * self.mesh.dx / self.mesh.dz * fields.mu[0, :] / 2        #Dn for top surface

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

        print(f"aS: {aS}")
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

    def _face_coefficients_interoir(self, fields: FlowField, faces: FaceField, i, j, component: MomentumComponent, dt):
        """
        Idea for this method is to assemble and return the face coefficients (aN, aS, aW, aE)
        when given a specific index pair for a FVM cell. 

        The fields and faces should be the varaibles that hold the flow and flux data respectively.

        This method will shorten the momentum coeff assembly functions dramatically I hope
        """
        rho = fields.rho[j, i]
        mu = fields.mu
        flux_q = faces.flux_q
        flux_w = faces.flux_w
        dx = self.mesh.dx
        dz = self.mesh.dz
        # dt = self.mesh.dt

        Fe: float = float(flux_q[j, i+1])
        Fw: float = float(flux_q[j, i-1])
        Fn: float = float(flux_w[j-1, i])
        Fs: float = float(flux_w[j+1, i])

        # Diffusion conductance
        De: float = float(2 * dz * (1/dx) * 0.5 * (mu[j, i] + mu[j, i+1]))
        Dw: float = float(2 * dz * (1/dx) * 0.5 * (mu[j, i] + mu[j, i-1]))
        Ds: float = float(dx * (1/dz) * 0.5 * (mu[j+1, i] + mu[j, i]))
        Dn: float = float(dx * (1/dz) * 0.5 * (mu[j-1, i] + mu[j, i]))
        
        # link coefficients that are utilized with face velocities and to produce 
        #section 5.3.2 patanakar
        aE: float = De + max(-Fe, 0)
        aW: float = Dw + max(Fw, 0)
        aN: float = Dn + max(Fn, 0)
        aS: float = Ds + max(-Fs, 0)
        a0: float = float(rho * (dx*dz) / dt)
        aP: float = aE + aW + aN + aS + (Fe-Fw) + (Fs-Fn) + a0

        return aP, aE, aW, aN, aS, a0

    def _boundary_coefficients(self, fields: FlowField, faces: FaceField, i, j, dt, component: MomentumComponent, boundarySide: BoundarySide):
        Nx = self.mesh.Nx
        Nz = self.mesh.Nz
        rho = fields.rho[j, i]
        mu = fields.mu
        flux_q = faces.flux_q
        flux_w = faces.flux_w
        dx = self.mesh.dx
        dz = self.mesh.dz
        V = dx*dz
        aE = 0
        aW = 0
        aN = 0
        aS = 0
        a0 = 0
        aP = 0

        if boundarySide == "left":
            # Left Surface
            Fe: float = float(flux_q[j, i+1])
            Fw: float = 0
            Fs: float = float(flux_w[j+1, i])
            Fn: float = float(flux_w[j-1, i]) 
    
            # Diffusion conductance
            De = 2 * dz * (1/dx) * 0.5 * (mu[j, i] + mu[j, i+1])
            Dw = 2 * dz * (1/dx) * 1.0 * (mu[j, i] + 0)          / 2
            Ds = dx * (1/dz) * 0.5 * (mu[j-1, i] + mu[j, i])
            Dn = dx * (1/(dz)) * 0.5 * (mu[j, i])
    
            # link coefficients that are utilized with face velocities
            aE: float = float(De + max(-Fe, 0))
            aW: float = float(Dw + max(Fw, 0))
            aN: float = float(Dn + max(Fn, 0))
            aS: float = float(Ds + max(-Fs, 0))
            a0: float = rho * (dx*dz) / dt
            aP: float = aE + aW + aN + aS + (Fe - Fw + Fs - Fn) + a0

        if boundarySide == "right":
            # Right Surface 
            Fe: float = 0
            Fw: float = float(flux_q[j, i-1])
            Fs: float = float(flux_w[j+1, i])
            Fn: float = float(flux_w[j-1, i]) 
    
            # Diffusion conductance
            De = 2 * dz * (1/dx) * 1.0 * (mu[j, i]) / 2
            Dw = 2 * dz * (1/dx) * 0.5 * (mu[j, i] + mu[j, i-1])
            Ds = dx * (1/dz) * 0.5 * (mu[j-1, i] + mu[j, i])
            Dn = dx * (1/(dz)) * 0.5 * (mu[j, i])
    
            # link coefficients that are utilized with face velocities
            aE: float = float(De + max(-Fe, 0))
            aW: float = float(Dw + max(Fw, 0))
            aN: float = float(Dn + max(Fn, 0))
            aS: float = float(Ds + max(-Fs, 0))
            a0: float = rho * (dx*dz) / dt
            aP: float = aE + aW + aN + aS + (Fe - Fw + Fs - Fn) + a0


        if boundarySide == "top":
            # Top Surface (w = 0) -> No slip condition on w, Fn = 0 & marangoni shear stresses present
            # fluxes through individual face (sum ~ 0)
            Fe: float = float(flux_q[j, i+1])
            Fw: float = float(flux_q[j, i-1])
            Fs: float = float(flux_w[j+1, i])
            Fn: float = 0 
    
            # Diffusion conductance
            De = 2 * dz * (1/dx) * 0.5 * (mu[j, i] + mu[j, i+1])
            Dw = 2 * dz * (1/dx) * 0.5 * (mu[j, i] + mu[j, i-1])
            Ds = dx * (1/dz) * 0.5 * (mu[j-1, i] + mu[j, i])
            Dn = dx * (1/(2*dz)) * 1.0 * (mu[j, i])
    
            # link coefficients that are utilized with face velocities
            aE: float = float(De + max(-Fe, 0))
            aW: float = float(Dw + max(Fw, 0))
            aN: float = float(Dn + max(Fn, 0))
            aS: float = float(Ds + max(-Fs, 0))
            a0: float = rho * (dx*dz) / dt
            aP: float = aE + aW + aN + aS + (Fe - Fw + Fs - Fn) + a0


        if boundarySide == "bottom":
            # Bottom Surface
            # fluxes through individual face (sum ~ 0)
            Fe: float = float(flux_q[j, i+1])
            Fw: float = float(flux_q[j, i-1])
            Fs: float = 0
            Fn: float = float(flux_w[j-1, i]) 
    
            # Diffusion conductance
            De = 2 * dz * (1/dx) * 0.5 * (mu[j, i] + mu[j, i+1])
            Dw = 2 * dz * (1/dx) * 0.5 * (mu[j, i] + mu[j, i-1])
            Ds = dx * (1/dz) * 1.0 * (mu[j, i] + 0) / 2
            Dn = dx * (1/(2*dz)) * 0.5 * (mu[j, i])
    
            # link coefficients that are utilized with face velocities
            aE: float = float(De + max(-Fe, 0))
            aW: float = float(Dw + max(Fw, 0))
            aN: float = float(Dn + max(Fn, 0))
            aS: float = float(Ds + max(-Fs, 0))
            a0: float = rho * (dx*dz) / dt
            aP: float = aE + aW + aN + aS + (Fe - Fw + Fs - Fn) + a0


        print(
        f"Cell ({i},{j}): "
        f"a0={a0:.3e}, "
        f"aE={aE:.3e}, aW={aW:.3e}, "
        f"aN={aN:.3e}, aS={aS:.3e}, "
        f"aP={aP:.3e}"
    )
        
        return aP, aE, aW, aN, aS, a0

    def _corner_coefficients(self, fields: FlowField, faces: FaceField, i, j, dt, component: MomentumComponent, corner: Corners):
        Nx = self.mesh.Nx
        Nz = self.mesh.Nz
        rho = fields.rho[j, i]
        mu = fields.mu
        flux_q = faces.flux_q
        flux_w = faces.flux_w
        dx = self.mesh.dx
        dz = self.mesh.dz
        V = dx*dz

        if corner == "NW":
            # top left corner
            Fe: float = float(flux_q[j, i+1])
            Fw: float = 0
            Fs: float = float(flux_w[j+1, i])
            Fn: float = 0 
    
            # Diffusion conductance
            De = 2 * dz * (1/dx) * 0.5 *   (mu[j, i] + mu[j, i+1])
            Dw = 2 * dz * (1/dx) * 1.0 *   (mu[j, i] + 0)                  / 2
            Ds = dx * (1/dz) * 0.5 *       (mu[j,i] + mu[j-1, i])
            Dn = dx * (1/(dz)) * 1.0 *     (mu[j, i] + 0)                  / 2
    
            # link coefficients that are utilized with face velocities
            aE: float = float(De + max(-Fe, 0))
            aW: float = float(Dw + max(Fw, 0))
            aN: float = float(Dn + max(Fn, 0))
            aS: float = float(Ds + max(-Fs, 0))
            a0: float = rho * (dx*dz) / dt
            aP: float = aE + aW + aN + aS + (Fe - Fw + Fs - Fn) + a0

            return aP, aE, aW, aN, aS, a0

        if corner == "NE":
            # top right corner
            Fe: float = 0
            Fw: float = float(flux_q[j, i-1])
            Fs: float = float(flux_w[j+1, i])
            Fn: float = 0 
    
            # Diffusion conductance
            De = 2 * dz * (1/dx) * 1.0 *   (mu[j, i] + 0)                / 2
            Dw = 2 * dz * (1/dx) * 0.5 *   (mu[j, i] + mu[j, i-1])                
            Ds = dx * (1/dz) * 0.5 *       (mu[j, i] + mu[j+1, i])
            Dn = dx * (1/(dz)) * 1.0 *     (mu[j, i] + 0)                / 2
    
            # link coefficients that are utilized with face velocities
            aE: float = float(De + max(-Fe, 0))
            aW: float = float(Dw + max(Fw, 0))
            aN: float = float(Dn + max(Fn, 0))
            aS: float = float(Ds + max(-Fs, 0))
            a0: float = rho * (dx*dz) / dt
            aP: float = aE + aW + aN + aS + (Fe - Fw + Fs - Fn) + a0

            return aP, aE, aW, aN, aS, a0


        if corner == "SW":
            # bottom left corner
            # Fe: float = float(flux_q[j, i+1])
            # Fw: float = float(flux_q[j, i-1])
            # Fs: float = float(flux_w[j+1, i])
            # Fn: float = float(flux_w[j-1, i]) 
            Fe: float = float(flux_q[j, i+1])
            Fw: float = 0
            Fs: float = 0
            Fn: float = float(flux_w[j-1, i]) 


            # Diffusion conductance
            De = 2 * dz * (1/dx) * 0.5 *   (mu[j, i] + mu[j, i+1])                
            Dw = 2 * dz * (1/dx) * 1.0 *   (mu[j, i] + 0)              / 2        
            Ds = dx * (1/dz) * 1.0 *       (mu[j, i] + 0)              / 2
            Dn = dx * (1/(dz)) * 0.5 *     (mu[j, i] + mu[j-1, i])                  
    
            # link coefficients that are utilized with face velocities
            aE: float = float(De + max(-Fe, 0))
            aW: float = float(Dw + max(Fw, 0))
            aN: float = float(Dn + max(Fn, 0))
            aS: float = float(Ds + max(-Fs, 0))
            a0: float = rho * (dx*dz) / dt
            aP: float = aE + aW + aN + aS + (Fe - Fw + Fs - Fn) + a0

            return aP, aE, aW, aN, aS, a0


        if corner == "SE":
            # bottom right corner
            Fe: float = 0
            Fw: float = float(flux_q[j, i-1])
            Fs: float = 0
            Fn: float = float(flux_w[j-1, i]) 

            # Diffusion conductance
            De = 2 * dz * (1/dx) * 1.0 *   (mu[j, i] + 0)                / 2  
            Dw = 2 * dz * (1/dx) * 0.5 *   (mu[j, i] + mu[j, i-1])             
            Ds = dx * (1/dz) * 1.0 *       (mu[j, i] + 0)                  / 2
            Dn = dx * (1/(dz)) * 0.5 *     (mu[j, i] + mu[j-1, i])                  
    
            # link coefficients that are utilized with face velocities
            aE: float = float(De + max(-Fe, 0))
            aW: float = float(Dw + max(Fw, 0))
            aN: float = float(Dn + max(Fn, 0))
            aS: float = float(Ds + max(-Fs, 0))
            a0: float = rho * (dx*dz) / dt
            aP: float = aE + aW + aN + aS + (Fe - Fw + Fs - Fn) + a0

            return aP, aE, aW, aN, aS, a0

    def _ij_to_k(self, i, j):
        Nz = self.mesh.Nz
        Nx = self.mesh.Nx

        return Nx * j + i 

    def _source_momentum(self, e_vector, u_vector):
        # source momentum sink from the mushy zone
        # utilizes full vector operation

        G = 1       # permeability  TODO: UPDATE
        d = 0.1     # computational constant to avoid div by 0

        return -G * (1-e_vector)**2 / (e_vector**3 + d) * u_vector

    def _source_surf_tension(self, dgamma_dT: np.ndarray, dTdx_all: np.ndarray):

        dTdx_topRow = dTdx_all[0, :]

        return -dgamma_dT * dTdx_topRow * self.mesh.dx