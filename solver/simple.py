import numpy as np
from dataclasses import dataclass, field

from core.mesh import Mesh
from core.fields import FlowField, FaceField
from core.boundary_conditions import BoundaryConditions

from discretization.momentum import u_link_coefficients, w_link_coefficients
from discretization.pressure import pressure_correction
from discretization.interpolation import face_fluxes

from core.settings import SolveSettings



class SIMPLEcfd:
    """
    Take settings from main execution -> set up domain, mesh, etc.

    run loops until convergence
    follow method on paper

    """

    def __init__(self, 
                 settings: SolveSettings, 
                 
                 ):
        















depth_convergence = 0.0001 # [m]
depth_change = 1
t = 0 # initial time [s]
m = 0 # initial iteration count
dt = 0.001 # [s]

## timesteps for X-Z plane



while depth_change > depth_convergence:
    t = (m+1)*dt

    # temperature module
    update_material_properties(xz)  ## TODO

    heating_module(xz["T"], t)

    update_material_properties(xz)

    simple_loop(1000)

    energy_conservation(xz)  ## TODO
    depth_change = compute_melt_pool_depth(xz)  ## TODO

## simple loop?
def simple_loop(max_iters):
    for iteration in range(max_iters):

        xz["dpdx"], xz["dpdz"] = calculate_pressure_gradients(xz["p"])

        # X-momentum RHS uses dpdx
        A_u, b_u = assemble_u_mom(xz, xz["dpdx"])
        xz = solve_u_star(A_u, b_u)

        # Z-momentum RHS uses dpdz
        A_w, b_w = assemble_w_mom(xz, xz["dpdz"])
        xz = solve_w_star(A_w, b_w)

        # Rhie-Chow, and pressure correction...
        xz["flux_q"], xz["flux_z"] = face_fluxes(xz)
        

        # guess pressure field p_star
        # solve momentum equations to find 
            # u_star
            # w_star
        # solve p' equation
        # calculate p from p' + p_star
        # calculate u, w from correction formulas
        # solve for other transport properties
        # use new pressure as next pressure guess p_star