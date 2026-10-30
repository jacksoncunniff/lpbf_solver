import numpy as np
from sympy.crypto import elgamal_private_key

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
        self.Au = np.empty((self.mesh.Nz, self.mesh.Nx))
        self.bu = np.empty(self.mesh.Nx * self.mesh.Nz)
        self.Aw = np.empty((self.mesh.Nz, self.mesh.Nx))
        self.bw = np.empty(self.mesh.Nx * self.mesh.Nz)

    def assemble_u(self, fields: FlowField, faces: FaceField, dt):
        """
        
        Construct and return A_u and b_u to solve u_mom iteratively elsewhere.
        
        """
        Nz = self.mesh.Nz
        Nx = self.mesh.Nx
        V = self.mesh.dx * self.mesh.dz
        e_vector = fields.liquid_frac.flatten()
        # INTERIOR CELLS ONLY
        for j in range(1, Nz-1):
            for i in range(1, Nx-1):
                aP, aE, aW, aN, aS, a0 = self._face_coefficients_interoir(fields, faces, i, j, "uq", dt=dt)
                k = self._ij_to_k(i, j)
                self.Au[k, k] = aP
                self.Au[k, k+1] = -aE
                self.Au[k, k-1] = -aW
                self.Au[k, k - Nx] = -aN
                self.Au[k, k + Nx] = -aS
                
                # Right-hand side
                mu_b = 0.5 * (fields.mu[j+1, i] + fields.mu[j, i])
                mu_t = 0.5 * (fields.mu[j-1, i] + fields.mu[j, i])

                # utilizes old w velocity values (previous iteration)
                w_e = fields.w[j, i+1]
                w_w = fields.w[j, i-1]
                w_ne = fields.w[j-1, i+1]
                w_nw = fields.w[j-1, i-1]
                w_se = fields.w[j+1, i+1]
                w_sw = fields.w[j+1, i-1]

                dwdx_b = (1/(4*self.mesh.dx)) * (w_e - w_w + w_se - w_sw)
                dwdx_t = (1/(4*self.mesh.dx)) * (w_e - w_w + w_ne - w_nw)

                diffusion_stresses = mu_b * dwdx_b - mu_t * dwdx_t
                # momentum_source =                       #TODO

                self.bu[k] = (a0 * fields.uq_old[j, i] - V * fields.dpdx[j, i]) + diffusion_stresses * self.mesh.dx

        # BOTTOM CELLS ONLY
        j=Nz
        for i in range(1, Nx-1):
            aP, aE, aW, aN, aS, a0 = self._boundary_coefficients(fields, faces, i, j, dt, "uq", boundarySide="bottom")
            k = self._ij_to_k(i, j)
            self.Au[k, k] = aP
            self.Au[k, k+1] = -aE
            self.Au[k, k-1] = -aW
            self.Au[k, k - Nx] = -aN
            self.Au[k, k + Nx] = -aS
            
            # Right-hand side
            mu_b = 0
            mu_t = 0.5 * (fields.mu[j-1, i] + fields.mu[j, i])

            # utilizes old w velocity values (previous iteration)
            w_e = fields.w[j, i+1]
            w_w = fields.w[j, i-1]
            w_ne = fields.w[j-1, i+1]
            w_nw = fields.w[j-1, i-1]
            w_se = 0
            w_sw = 0

            dwdx_b = (1/(4*self.mesh.dx)) * (w_e - w_w + w_se - w_sw)
            dwdx_t = (1/(4*self.mesh.dx)) * (w_e - w_w + w_ne - w_nw)

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
            self.Au[k, k-1] = -aW
            self.Au[k, k - Nx] = -aN
            self.Au[k, k + Nx] = -aS
            
            # Right-hand side
            mu_b = 0.5 * (fields.mu[j+1, i] + fields.mu[j, i])
            mu_t = 0.5 * (fields.mu[j-1, i] + fields.mu[j, i])

            # utilizes old w velocity values (previous iteration)
            w_e = fields.w[j, i+1]
            w_w = 0
            w_ne = fields.w[j-1, i+1]
            w_nw = 0
            w_se = fields.w[j+1, i+1]
            w_sw = 0

            dwdx_b = (1/(4*self.mesh.dx)) * (w_e - w_w + w_se - w_sw)
            dwdx_t = (1/(4*self.mesh.dx)) * (w_e - w_w + w_ne - w_nw)

            diffusion_stresses = mu_b * dwdx_b - mu_t * dwdx_t
            # momentum_source =                       #TODO

            self.bu[k] = (a0 * fields.uq_old[j, i] - V * fields.dpdx[j, i]) + diffusion_stresses * self.mesh.dx

        # RIGHT CELLS ONLY
        i = Nx
        for j in range(1, Nz-1):
            aP, aE, aW, aN, aS, a0 = self._boundary_coefficients(fields, faces, i, j, dt, "uq", boundarySide="right")
            k = self._ij_to_k(i, j)
            self.Au[k, k] = aP
            self.Au[k, k+1] = -aE
            self.Au[k, k-1] = -aW
            self.Au[k, k - Nx] = -aN
            self.Au[k, k + Nx] = -aS
            
            # Right-hand side
            mu_b = 0.5 * (fields.mu[j+1, i] + fields.mu[j, i])
            mu_t = 0.5 * (fields.mu[j-1, i] + fields.mu[j, i])

            # utilizes old w velocity values (previous iteration)
            w_e = fields.w[j, i+1]
            w_w = fields.w[j, i-1]
            w_ne = fields.w[j-1, i+1]
            w_nw = fields.w[j-1, i-1]
            w_se = fields.w[j+1, i+1]
            w_sw = fields.w[j+1, i-1]

            dwdx_b = (1/(4*self.mesh.dx)) * (w_e - w_w + w_se - w_sw)
            dwdx_t = (1/(4*self.mesh.dx)) * (w_e - w_w + w_ne - w_nw)

            diffusion_stresses = mu_b * dwdx_b - mu_t * dwdx_t
            # momentum_source =                       #TODO

            self.bu[k] = (a0 * fields.uq_old[j, i] - V * fields.dpdx[j, i]) + diffusion_stresses * self.mesh.dx


        # TOP CELLS ONLY
        j=0
        for i in range(1, Nx-1):
            aP, aE, aW, aN, aS, a0 = self._boundary_coefficients(fields, faces, i, j, dt, "uq", boundarySide="top")
            k = self._ij_to_k(i, j)
            self.Au[k, k] = aP
            self.Au[k, k+1] = -aE
            self.Au[k, k-1] = -aW
            self.Au[k, k - Nx] = -aN
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
        self.Au[k, k-1] = -aW
        self.Au[k, k - Nx] = -aN
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
        i=Nx
        aP, aE, aW, aN, aS, a0 = self._corner_coefficients(fields, faces, i, j, dt, "uq", corner="NE")
        k = self._ij_to_k(i, j)
        self.Au[k, k] = aP
        self.Au[k, k+1] = -aE
        self.Au[k, k-1] = -aW
        self.Au[k, k - Nx] = -aN
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
        j=Nz
        i=0
        aP, aE, aW, aN, aS, a0 = self._corner_coefficients(fields, faces, i, j, dt, "uq", corner="SW")
        k = self._ij_to_k(i, j)
        self.Au[k, k] = aP
        self.Au[k, k+1] = -aE
        self.Au[k, k-1] = -aW
        self.Au[k, k - Nx] = -aN
        self.Au[k, k + Nx] = -aS
        
        # Right-hand side
        mu_b = 0
        mu_t = 0.5 * (fields.mu[j-1, i] + fields.mu[j, i])
        # mu_t = 0

        # dwdx_b = 0.5 * (fields.dwdx[j, i] + fields.dwdx[j-1, i]) 
        dwdx_t = 0.5 * (fields.dwdx[j, i] + fields.dwdx[j+1, i]) 
        dwdx_b = 0      # boundary condition

        diffusion_stresses = mu_b * dwdx_b - mu_t * dwdx_t

        self.bu[k] = (a0 * fields.uq_old[j, i] - V * fields.dpdx[j, i]) + diffusion_stresses * self.mesh.dx
    
    
    # SE CELL (BOTTOM LEFT CORNER)
        j=Nz
        i=Nx
        aP, aE, aW, aN, aS, a0 = self._corner_coefficients(fields, faces, i, j, dt, "uq", corner="SE")
        k = self._ij_to_k(i, j)
        self.Au[k, k] = aP
        self.Au[k, k+1] = -aE
        self.Au[k, k-1] = -aW
        self.Au[k, k - Nx] = -aN
        self.Au[k, k + Nx] = -aS
        
        # Right-hand side
        mu_b = 0
        mu_t = 0.5 * (fields.mu[j-1, i] + fields.mu[j, i])


        # dwdx_b = 0.5 * (fields.dwdx[j, i] + fields.dwdx[j-1, i]) 
        dwdx_t = 0.5 * (fields.dwdx[j, i] + fields.dwdx[j+1, i]) 
        dwdx_b = 0

        diffusion_stresses = mu_b * dwdx_b - mu_t * dwdx_t

        self.bu[k] = (a0 * fields.uq_old[j, i] - V * fields.dpdx[j, i]) + diffusion_stresses * self.mesh.dx
    

        ## Source Momentum Sink Contribution
        u_old_vector = fields.uq_old.flatten()
        self.bu += self._source_momentum(e_vector, u_old_vector)

        ## Surface Tension Contribution
        self.bu += self._source_surf_tension(fields.dgamma_dT, fields.dTdx)



    def assemble_w(self, fields, faces, dt):
        """Construct A_w and b_w."""
        pass


        ## Following three methods only return the link coefficients for assembly methods
        ## to place them into the greater Au or Aw matrix
        ## Aw component option is not enabled yet TODO
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

        if boundarySide == "left":
            # Left Surface
            Fe: float = float(flux_q[j, i+1])
            Fw: float = 0
            Fs: float = float(flux_w[j+1, i])
            Fn: float = float(flux_w[j-1, i]) 
    
            # Diffusion conductance
            De = -2 * dz * (1/dx) * 0.5 * (mu[j, i] + mu[j, i+1])
            Dw = -2 * dz * (1/dx) * 0.5 * (mu[j, i] + mu[j, i-1]) / 2
            Ds = -dx * (1/dz) * 0.5 * (mu[j-1] + mu[j, i])
            Dn = -dx * (1/(dz)) * 0.5 * (mu[j, i])
    
            # link coefficients that are utilized with face velocities
            aE: float = float(De + max(-Fe, 0))
            aW: float = float(Dw + max(Fw, 0))
            aN: float = float(Dn + max(Fn, 0))
            aS: float = float(Ds + max(-Fs, 0))
            a0: float = rho * (dx*dz) / dt
            aP: float = aE + aW + aN + aS
            return aP, aE, aW, aN, aS, a0

        if boundarySide == "right":
            # Right Surface 
            Fe: float = 0
            Fw: float = float(flux_q[j, i-1])
            Fs: float = float(flux_w[j+1, i])
            Fn: float = float(flux_w[j-1, i]) 
    
            # Diffusion conductance
            De = -2 * dz * (1/dx) * 0.5 * (mu[j, i] + mu[j, i+1]) / 2
            Dw = -2 * dz * (1/dx) * 0.5 * (mu[j, i] + mu[j, i-1])
            Ds = -dx * (1/dz) * 0.5 * (mu[j-1] + mu[j, i])
            Dn = -dx * (1/(dz)) * 0.5 * (mu[j, i])
    
            # link coefficients that are utilized with face velocities
            aE: float = float(De + max(-Fe, 0))
            aW: float = float(Dw + max(Fw, 0))
            aN: float = float(Dn + max(Fn, 0))
            aS: float = float(Ds + max(-Fs, 0))
            a0: float = rho * (dx*dz) / dt
            aP: float = aE + aW + aN + aS
            return aP, aE, aW, aN, aS, a0

        if boundarySide == "top":
            # Top Surface (w = 0) -> No slip condition on w, Fn = 0 & marangoni shear stresses present
            # fluxes through individual face (sum ~ 0)
            Fe: float = float(flux_q[j, i+1])
            Fw: float = float(flux_q[j, i-1])
            Fs: float = float(flux_w[j+1, i])
            Fn: float = 0 
    
            # Diffusion conductance
            De = -2 * dz * (1/dx) * 0.5 * (mu[j, i] + mu[j, i+1])
            Dw = -2 * dz * (1/dx) * 0.5 * (mu[j, i] + mu[j, i-1])
            Ds = -dx * (1/dz) * 0.5 * (mu[j-1] + mu[j, i])
            Dn = -dx * (1/(2*dz)) * 0.5 * (mu[j, i])
    
            # link coefficients that are utilized with face velocities
            aE: float = float(De + max(-Fe, 0))
            aW: float = float(Dw + max(Fw, 0))
            aN: float = float(Dn + max(Fn, 0))
            aS: float = float(Ds + max(-Fs, 0))
            a0: float = rho * (dx*dz) / dt
            aP: float = aE + aW + aN + aS
            return aP, aE, aW, aN, aS, a0

        if boundarySide == "bottom":
            # Bottom Surface
            # fluxes through individual face (sum ~ 0)
            Fe: float = float(flux_q[j, i+1])
            Fw: float = float(flux_q[j, i-1])
            Fs: float = 0
            Fn: float = float(flux_w[j+1, i]) 
    
            # Diffusion conductance
            De = -2 * dz * (1/dx) * 0.5 * (mu[j, i] + mu[j, i+1])
            Dw = -2 * dz * (1/dx) * 0.5 * (mu[j, i] + mu[j, i-1])
            Ds = -dx * (1/dz) * 0.5 * (mu[j-1] + mu[j, i]) / 2
            Dn = -dx * (1/(2*dz)) * 0.5 * (mu[j, i])
    
            # link coefficients that are utilized with face velocities
            aE: float = float(De + max(-Fe, 0))
            aW: float = float(Dw + max(Fw, 0))
            aN: float = float(Dn + max(Fn, 0))
            aS: float = float(Ds + max(-Fs, 0))
            a0: float = rho * (dx*dz) / dt
            aP: float = aE + aW + aN + aS
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
            De = -2 * dz * (1/dx) * 0.5 *   (mu[j, i] + mu[j, i+1])
            Dw = -2 * dz * (1/dx) * 0.5 *   (mu[j, i] + 0)                  / 2
            Ds = -dx * (1/dz) * 0.5 *       (mu[j,i] + mu[j-1, i])
            Dn = -dx * (1/(dz)) * 0.5 *     (mu[j, i] + 0)                  / 2
    
            # link coefficients that are utilized with face velocities
            aE: float = float(De + max(-Fe, 0))
            aW: float = float(Dw + max(Fw, 0))
            aN: float = float(Dn + max(Fn, 0))
            aS: float = float(Ds + max(-Fs, 0))
            a0: float = rho * (dx*dz) / dt
            aP: float = aE + aW + aN + aS
            return aP, aE, aW, aN, aS, a0

        if corner == "NE":
            # top right corner
            Fe: float = 0
            Fw: float = float(flux_q[j, i-1])
            Fs: float = float(flux_w[j+1, i])
            Fn: float = 0 
    
            # Diffusion conductance
            De = -2 * dz * (1/dx) * 0.5 *   (mu[j, i] + 0)                / 2
            Dw = -2 * dz * (1/dx) * 0.5 *   (mu[j, i] + mu[j, i-1])                
            Ds = -dx * (1/dz) * 0.5 *       (mu[j, i] + mu[j+1, i])
            Dn = -dx * (1/(dz)) * 0.5 *     (mu[j, i] + 0)                / 2
    
            # link coefficients that are utilized with face velocities
            aE: float = float(De + max(-Fe, 0))
            aW: float = float(Dw + max(Fw, 0))
            aN: float = float(Dn + max(Fn, 0))
            aS: float = float(Ds + max(-Fs, 0))
            a0: float = rho * (dx*dz) / dt
            aP: float = aE + aW + aN + aS
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
            De = -2 * dz * (1/dx) * 0.5 *   (mu[j, i] + mu[j, i+1])                
            Dw = -2 * dz * (1/dx) * 0.5 *   (mu[j, i] + 0)              / 2        
            Ds = -dx * (1/dz) * 0.5 *       (mu[j, i] + 0)              / 2
            Dn = -dx * (1/(dz)) * 0.5 *     (mu[j, i] + mu[j+1, i])                  
    
            # link coefficients that are utilized with face velocities
            aE: float = float(De + max(-Fe, 0))
            aW: float = float(Dw + max(Fw, 0))
            aN: float = float(Dn + max(Fn, 0))
            aS: float = float(Ds + max(-Fs, 0))
            a0: float = rho * (dx*dz) / dt
            aP: float = aE + aW + aN + aS
            return aP, aE, aW, aN, aS, a0


        if corner == "SE":
            # bottom right corner
            Fe: float = 0
            Fw: float = float(flux_q[j, i-1])
            Fs: float = 0
            Fn: float = float(flux_w[j-1, i]) 

            # Diffusion conductance
            De = -2 * dz * (1/dx) * 0.5 *   (mu[j, i] + 0)                / 2  
            Dw = -2 * dz * (1/dx) * 0.5 *   (mu[j, i] + mu[j, i-1])             
            Ds = -dx * (1/dz) * 0.5 *       (mu[j, i] + 0)                  / 2
            Dn = -dx * (1/(dz)) * 0.5 *     (mu[j, i] + mu[j-1, i])                  
    
            # link coefficients that are utilized with face velocities
            aE: float = float(De + max(-Fe, 0))
            aW: float = float(Dw + max(Fw, 0))
            aN: float = float(Dn + max(Fn, 0))
            aS: float = float(Ds + max(-Fs, 0))
            a0: float = rho * (dx*dz) / dt
            aP: float = aE + aW + aN + aS
            return aP, aE, aW, aN, aS, a0

    def _ij_to_k(self, i, j):
        Nz = self.mesh.Nz
        Nx = self.mesh.Nx

        return Nx * j + i 

    def _source_momentum(self, e_vector, u_vector):
        # source momentum sink from the mushy zone
        # utilizes full vector operation

        G = 1       # permeability
        d = 0.1     # computational constant to avoid div by 0

        return -G * (1-e_vector)**2 / (e_vector**3 + d) * u_vector

    def _source_surf_tension(self, dgamma_dT: np.ndarray, dTdx_all: np.ndarray):

        dTdx_topRow = dTdx_all[0, :]

        return dgamma_dT * dTdx_topRow