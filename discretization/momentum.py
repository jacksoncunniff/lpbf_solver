import numpy as np

from core.fields import FaceField, FlowField
from typing import Literal
MomentumComponent = Literal["uq", "w"]

class MomentumAssembler:
    """
    Assemble the discretized 2D momentum equations.

    Uses cell-centered velocity fields and face-centered
    mass fluxes on a structured collocated FVM mesh.
    """

    def __init__(self, mesh, material, boundary_conditions):
        self.mesh = mesh
        self.material = material
        self.bc = boundary_conditions

    def assemble_u(self, fields, faces, dt):
        """Construct A_u and b_u."""
        pass

    def assemble_w(self, fields, faces, dt):
        """Construct A_w and b_w."""
        pass

    def _face_coefficients(self, fields: FlowField, faces: FaceField, i, j, component: MomentumComponent):
        """
        Idea for this method is to assemble and return the face coefficients (aN, aS, aW, aE)
        when given a specific index pair for a FVM cell. 

        The fields and faces should be the varaibles that hold the flow and flux data respectively.

        This method will shorten the momentum coeff assembly functions dramatically I hope
        """
        pass
    

    def _boundary_coefficients(self, fields: FlowField, faces: FaceField, i, j, component: MomentumComponent):
        pass





def assemble_u_mom(fields: dict, dpdx):

    k=0 #index for flattened matrix
    A = np.empty_like(fields["uq"])
    b = np.empty_like(fields["b_uq"])

    
    ## INTERIOR POINTS
    for j in range(1, Nz-1):
        for i in range(1,Nx-1): 
            rho = fields["rho"][j, i]
            
            Fe: float = fields["flux_q"][j, i+1]
            Fw: float = fields["flux_q"][j, i-1]
            Fn: float = fields["flux_z"][j-1, i]
            Fs: float = fields["flux_z"][j+1, i]

            # Diffusion conductance
            De = 2 * dz * (1/dx) * 0.5 * (fields["mu"][j, i] + fields["mu"][j, i+1])
            Dw = 2 * dz * (1/dx) * 0.5 * (fields["mu"][j, i] + fields["mu"][j, i-1])
            Ds = dx * (1/dz) * 0.5 * (fields["mu"][j+1, i] + fields["mu"][j, i])
            Dn = dx * (1/dz) * 0.5 * (fields["mu"][j-1, i] + fields["mu"][j, i])

            # link coefficients that are utilized with face velocities and to produce 
            #section 5.3.2 patanakar
            aE: float = De + max(-Fe, 0)
            aW: float = Dw + max(Fw, 0)
            aN: float = Dn + max(Fn, 0)
            aS: float = Ds + max(-Fs, 0)
            a0: float = rho * (dx*dz) / dt
            aP: float = aE + aW + aN + aS + (Fe-Fw) + (Fs-Fn) + a0

            # Populate A matrix that is Nx*Ny square in size -> each row represents one equation
            # with grid point P located along the diagonal

            k = ij_to_k(i, j, Nx, Nz)
            A[k, k] = aP
            A[k, k+1] = -aE
            A[k, k-1] = -aW
            A[k, k - Nx] = -aN
            A[k, k + Nx] = -aS

            # Right-hand side
            mu_b = 0.5 * (fields["mu"][j+1, i] + fields["mu"][j, i])
            mu_t = 0.5 * (fields["mu"][j-1, i] + fields["mu"][j, i])

            # utilizes old w velocity values (previous iteration)
            w_e = fields["w"][j, i+1]
            w_w = fields["w"][j, i-1]
            w_ne = fields["w"][j-1, i+1]
            w_nw = fields["w"][j-1, i-1]
            w_se = fields["w"][j+1, i+1]
            w_sw = fields["w"][j+1, i-1]

            dwdx_b = (1/(4*dx)) * (w_e - w_w + w_se - w_sw)
            dwdx_t = (1/(4*dx)) * (w_e - w_w + w_ne - w_nw)

            diffusion_stresses = mu_b * dwdx_b - mu_t * dwdx_t

            b[k] = (a0 * fields["uq_old"][j, i] - V * dpdx[j, i]) + diffusion_stresses * dx

        pass

    # Boundary Treatment
    # Top Surface (W = 0) -> No slip condition on w, Fn = 0 & marangoni shear stresses present
    j = 0 
    for i in range(1, Nx):
        
        # fluxes through individual face (sum ~ 0)
        Fe: float = fields["flux_q"][j, i+1]
        Fw: float = fields["flux_q"][j, i-1]
        Fs: float = fields["flux_q"][j+1, i]
        Fn: float = 0 

        # Diffusion conductance --> marangoni stresses propagate here
        De = -2 * dz * (1/dx) * 0.5 * (fields["mu"][j, i] + fields["mu"][j, i+1])
        Dw = -2 * dz * (1/dx) * 0.5 * (fields["mu"][j, i] + fields["mu"][j, i-1])
        Ds = -dx * (1/dz) * 0.5 * (fields["mu"][j-1] + fields["mu"][j, i])
        
        Dn = -dx * (1/(2*dz)) * 0.5 * (fields["mu"][j, i])

        # link coefficients that are utilized with face velocities and to produce 
        aE: float = De + max(-Fe, 0)
        aW: float = Dw + max(Fw, 0)
        aN: float = Dn + max(Fn, 0)
        aS: float = Ds + max(-Fs, 0)
        aP: float = aE + aW + aN + aS

        # Populate A matrix that is Nx*Ny square in size -> each row represents one equation
        k = ij_to_k(i, j, Nx, Nz)
        A[k, k] = aP
        A[k, k+1] = aE
        A[k, k-1] = aW
        A[k, k + Nz] = aS

        # Right-hand side
        mu_b = 0.5 * (fields["mu"][j+1, i] + fields["mu"][j, i])
        mu_t = 0.5 * (fields["mu"][j-1, i] + fields["mu"][j, i])

        # utilizes old w velocity values (previous iteration)
        w_e = fields["w"][j, i+1]
        w_w = fields["w"][j, i-1]
        # w_ne = fields["w"][j-1, i+1]
        # w_nw = fields["w"][j-1, i-1]
        w_se = fields["w"][j+1, i+1]
        w_sw = fields["w"][j+1, i-1]

        dwdx_b = (1/(4*dx)) * (w_e - w_w + w_se - w_sw)
        dwdx_t = (1/(4*dx)) * (w_e - w_w + w_ne - w_se)
        diffusion_stresses = mu_b * dwdx_b - mu_t * dwdx_t 

        b[k] = (a0 * fields["uq_old"][j, i] - V * dpdx[j, i]) + diffusion_stresses * dx
    
    # Boundary Treatment
    # Bottom Surface (Z = Nz) -> No slip condition, Fs = 0
    j = Nz 
    for i in range(1, Nx):
        # fluxes through individual face (sum ~ 0)
        Fe: float = fields["flux_q"][j, i+1]
        Fw: float = fields["flux_q"][j, i-1]
        Fn: float = fields["flux_q"][j-1, i]
        Fs: float = 0


        # Diffusion conductance
        De = -2 * dz * (1/dx) * 0.5 * (fields["mu"][j, i] + fields["mu"][j, i+1])
        Dw = -2 * dz * (1/dx) * 0.5 * (fields["mu"][j, i] + fields["mu"][j, i-1])
        Dn = -dx * (1/dz) * 0.5 * (fields["mu"][j+1] + fields["mu"][j, i])
        Ds = -dx * (1/(2*dz)) * 0.5 * (fields["mu"][j, i])

        # link coefficients that are utilized with face velocities and to produce 
        aE: float = De + max(-Fe, 0)
        aW: float = Dw + max(Fw, 0)
        aN: float = Dn + max(Fn, 0)
        aS: float = Ds + max(-Fs, 0)
        aP: float = aE + aW + aN + aS

        k = ij_to_k(i, j, Nx, Nz)
        A[k, k] = aP
        A[k, k + 1] = -aE
        A[k, k - 1] = -aW       
        A[k, k - Nz] = aN


        # Right-hand side
        mu_b = 0.5 * (fields["mu"][j+1, i] + fields["mu"][j, i])
        mu_t = 0.5 * (fields["mu"][j-1, i] + fields["mu"][j, i])

        # utilizes old w velocity values (previous iteration)
        w_e = fields["w"][j, i+1]
        w_w = fields["w"][j, i-1]
        w_ne = fields["w"][j-1, i+1]
        w_nw = fields["w"][j-1, i-1]
        w_se = fields["w"][j+1, i+1]
        w_sw = fields["w"][j+1, i-1]

        dwdx_b = (1/(4*dx)) * (w_e - w_w + w_se - w_sw)
        dwdx_t = (1/(4*dx)) * (w_e - w_w + w_ne - w_se)
        diffusion_stresses = mu_b * dwdx_b - mu_t * dwdx_t 

        b[k] = (a0 * fields["uq_old"][j, i] - V * dpdx[j, i]) + diffusion_stresses * dx



    # Boundary Treatment
    # Left Surface (X = 0) -> No slip condition, Fw = 0
    i = 0 
    for j in range(1,Nz):
        
        # fluxes through individual face (sum ~ 0)
        Fe: float = fields["flux_q"][j, i+1]
        Fw: float = 0
        Fs: float = fields["flux_z"][j+1, i]
        Fn: float = fields["flux_z"][j-1, i] 

        # Diffusion conductance
        De = -2 * dz * (1/dx) * 0.5 * (fields["mu"][j, i] + fields["mu"][j, i+1])
        Dw = -2 * dz * (1/dx) * 0.5 * (fields["mu"][j, i] + fields["mu"][j, i-1])
        Ds = -dx * (1/dz) * 0.5 * (fields["mu"][j-1] + fields["mu"][j, i])
        
        Dn = -dx * (1/(2*dz)) * 0.5 * (fields["mu"][j, i])

        # link coefficients that are utilized with face velocities and to produce 
        aE: float = De + max(-Fe, 0)
        aW: float = Dw + max(Fw, 0)
        aN: float = Dn + max(Fn, 0)
        aS: float = Ds + max(-Fs, 0)
        aP: float = aE + aW + aN + aS

        k = ij_to_k(i, j, Nx, Nz)
        A[k, k] = aP
        A[k, k+1] = -aE
        A[k, k + Nz] = -aS
        A[k, k - Nz] = -aN

        # Right-hand side
        mu_b = 0.5 * (fields["mu"][j+1, i] + fields["mu"][j, i])
        mu_t = 0.5 * (fields["mu"][j-1, i] + fields["mu"][j, i])

        # utilizes old w velocity values (previous iteration)
        w_e = fields["w"][j, i+1]
        # w_w = fields["w"][j, i-1]
        # w_ne = fields["w"][j-1, i+1]
        # w_nw = fields["w"][j-1, i-1]
        w_se = fields["w"][j+1, i+1]
        w_sw = fields["w"][j+1, i-1]

        dwdx_b = (1/(4*dx)) * (w_e - w_w + w_se - w_sw)
        dwdx_t = (1/(4*dx)) * (w_e - w_w + w_ne - w_se)
        diffusion_stresses = mu_b * dwdx_b - mu_t * dwdx_t 

        b[k] = (a0 * fields["uq_old"][j, i] - V * dpdx[j, i]) + diffusion_stresses * dx


    # Top Left Corner
    i = 0
    j = 0

    Fe: float = fields["flux_q"][j, i+1]
    Fw: float = 0
    Fn: float = 0
    Fs: float = fields["flux_q"][j+1, i]

    De = -2 * dz * (1/dx) * 0.5 * (fields["mu"][j, i] + fields["mu"][j, i+1])
    Dw = -2 * dz * (1/(2*dx)) * 0.5 * (fields["mu"][j, i])
    Ds = -dx * (1/dz) * 0.5 * (fields["mu"][j-1] + fields["mu"][j, i])
    Dn = -dx * (1/(2*dz)) * 0.5 * (fields["mu"][j, i])

    aE: float = De + max(-Fe, 0)
    aW: float = Dw + max(Fw, 0)
    aN: float = Dn + max(Fn, 0)
    aS: float = Ds + max(-Fs, 0)
    aP: float = aE + aW + aN + aS

    k = ij_to_k(i, j, Nx, Nz)
    A[k, k] = aP
    A[k, k+1] = -aE
    A[k, k + Nz] = aS
    # A[k, k-1] = aW
    # A[k, k - Nz] = aN


    # Top Right Corner
    i = Nx
    j = 0

    Fe: float = 0
    Fn: float = 0
    Fw: float = fields["flux_q"][j, i-1]
    Fs: float = fields["flux_q"][j+1, i]

    De = -2 * dz * (1/dx) * 0.5 * (fields["mu"][j, i] + fields["mu"][j, i+1])
    Dw = -2 * dz * (1/(2*dx)) * 0.5 * (fields["mu"][j, i])
    Ds = -dx * (1/dz) * 0.5 * (fields["mu"][j-1] + fields["mu"][j, i])
    Dn = -dx * (1/(2*dz)) * 0.5 * (fields["mu"][j, i])

    aE: float = De + max(-Fe, 0)
    aW: float = Dw + max(Fw, 0)
    aN: float = Dn + max(Fn, 0)
    aS: float = Ds + max(-Fs, 0)
    aP: float = aE + aW + aN + aS

    k = ij_to_k(i, j, Nx, Nz)
    A[k, k] = aP
    A[k, k-1] = -aW
    A[k, k + Nz] = aS
    # A[k, k - Nz] = aN
    # A[k, k+1] = aE


    # Bottom Left
    i = 0
    j = Nz
    Fe: float = fields["flux_q"][j, i+1]
    Fn: float = fields["flux_q"][j-1, i]
    Fw: float = 0
    Fs: float = 0

    # Diffusion conductance --> marangoni stresses propagate here
    De = -2 * dz * (1/dx) * 0.5 * (fields["mu"][j, i] + fields["mu"][j, i+1])
    Dw = -2 * dz * (1/dx) * 0.5 * (fields["mu"][j, i] + fields["mu"][j, i-1])
    Ds = -dx * (1/dz) * 0.5 * (fields["mu"][j-1] + fields["mu"][j, i])
    Dn = -dx * (1/dz) * 0.5 * (fields["mu"][j+1] + fields["mu"][j, i])

    aE: float = De + max(-Fe, 0)
    aW: float = Dw + max(Fw, 0)
    aN: float = Dn + max(Fn, 0)
    aS: float = Ds + max(-Fs, 0)
    aP: float = aE + aW + aN + aS

    k = ij_to_k(i, j, Nx, Nz)
    A[k, k] = aP
    A[k, k+1] = -aE
    A[k, k - Nz] = aN
    # A[k, k-1] = aW
    # A[k, k + Nz] = aS



    # Bottom Right
    i = Nx
    j = Nz
    Fw: float = fields["flux_q"][j, i-1]
    Fn: float = fields["flux_q"][j-1, i]
    Fe: float = 0
    Fs: float = 0


    # Diffusion conductance
    De = -2 * dz * (1/dx) * 0.5 * (fields["mu"][j, i] + fields["mu"][j, i+1])
    Dw = -2 * dz * (1/dx) * 0.5 * (fields["mu"][j, i] + fields["mu"][j, i-1])
    Ds = -dx * (1/dz) * 0.5 * (fields["mu"][j-1] + fields["mu"][j, i])
    Dn = -dx * (1/dz) * 0.5 * (fields["mu"][j+1] + fields["mu"][j, i])

    # link coefficients that are utilized with face velocities and to produce 
    #section 5.3.2 patanakar
    aE: float = De + max(-Fe, 0)
    aW: float = Dw + max(Fw, 0)
    aN: float = Dn + max(Fn, 0)
    aS: float = Ds + max(-Fs, 0)
    aP: float = aE + aW + aN + aS

    k = ij_to_k(i, j, Nx, Nz)
    A[k, k] = aP
    A[k, k-1] = -aW
    A[k, k - Nz] = aN
    # A[k, k+1] = aE
    # A[k, k + Nz] = aS

    A_u = A
    
    return A_u, b_u, A_p