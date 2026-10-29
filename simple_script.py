import numpy as np
from scipy.sparse import diags
from utils import *

rho = 1

# bed size [m]
Lx = 1e-3
Lz = 2.5e-2
Ly = 2.5e-2

# mesh quantity
Nx = 100
Ny = 100
Nz = 100

# cell spaces
dx = Lx/Nx
dy = Ly/Ny
dz = Lz/Nz

# origin
x_min = 0
z_min = 0
y_min = -Ly/2

# Cell-center coordinates
x_coords = x_min + (np.arange(Nx) + 0.5) * dx
y_coords = y_min + (np.arange(Ny) + 0.5) * dy
z_coords = z_min + (np.arange(Nz) + 0.5) * dz


T0 = 300 # Kelvin
Tb = T0 # Base plate temperature




# Pre-allocate solution arrays
def allocate_fields(nq, Nz):
    shape = (Nz, nq)

    return {
        # Primary variables     **this floats as the initial guess for u, w, p
        "uq": np.zeros(shape),
        "w":  np.zeros(shape),
        "p":  np.zeros(shape),
        "T": np.ones(shape)*T0, #temperature field for analytical temp eqn, start at ambient
        "mu":  np.zeros(shape),
        "rho":  np.zeros(shape),

        # Previous timestep
        "uq_old": np.zeros(shape),
        "w_old":  np.zeros(shape),

        # SIMPLE predictor/corrector
        "uq_star": np.zeros(shape),
        "w_star":  np.zeros(shape),
        "p_prime": np.zeros(shape),

        # Momentum diagonal coefficients
        "aP_uq": np.zeros(shape),
        "aP_w":  np.zeros(shape),

        # Pressure gradients for PWIM
        "dpdq": np.zeros(shape),
        "dpdz": np.zeros(shape),

        # Face mass fluxes
        "flux_q": np.zeros((Nz, nq + 1)),
        "flux_z": np.zeros((Nz + 1, nq)),

        # Linear-system RHS vectors
        "b_uq": np.zeros(nq * Nz),
        "b_w":  np.zeros(nq * Nz),
        "b_p":  np.zeros(nq * Nz),
    }

xz = allocate_fields(Nx, Nz)  
yz = allocate_fields(Ny, Nz) 




def calculate_pressure_gradients(fields):
    uq = fields["uq"]
    w = fields["w"]
    dpdx = np.zeros_like(uq)
    dpdz = np.zeros_like(uq)

    # Interior: second-order central differences
    dpdx[:, 1:-1] = (uq[:, 2:] - uq[:, :-2]) / (2 * dx)
    dpdz[1:-1, :] = (w[2:, :] - w[:-2, :]) / (2 * dz)

    # Left and right boundary-adjacent cells
    dpdx[:, 0] = (uq[:, 1] - uq[:, 0]) / dx
    dpdx[:, -1] = (uq[:, -1] - uq[:, -2]) / dx

    # Top and bottom boundary-adjacent cells
    dpdz[0, :] = (w[1, :] - w[0, :]) / dz
    dpdz[-1, :] = (w[-1, :] - w[-2, :]) / dz

    return dpdx, dpdz   
    
#upwind method on convection -> page 84 patanakar or page 87 ferziger
#CDS on diffusion-> page 88 ferziger
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


def assemble_w_mom(fields: dict, dpdz):



    Fe: float = fields["flux_q"][j, i+1]
    Fw: float = fields["flux_q"][j, i-1]
    Fn: float = fields["flux_q"][j-1, i]
    Fs: float = fields["flux_q"][j+1, i]

    aE: float
    aW: float
    aN: float
    aS: float

    aP: float = 




    pass

## Solver variables



def update_material_properties(fields):
    
    
    pass

def energy_conservation(fields):
    pass

def compute_melt_pool_depth(fields):
    pass

def face_fluxes(fields):
    ## Implementing Rhie-Chow interpolation and saving to the u fluxes and w fluxes variables in fields arrays
    pass

def solve_u_star(A, b, mesh):

    u_star = spsolve


    pass


def solve_w_star():


    pass


def velocity_correction(velocity_star, link_coeff_diag_vector, p_prime_matrix):
    # Cell centered velocity correction
    # patanakar pg 123

    # Interoir points
    for j in range(1, Nz-1):
        for i in range(1, Nx-1):


    # Left boundary
    i = 0
    for j in range(1, Ny-1)


    return velocity_array
