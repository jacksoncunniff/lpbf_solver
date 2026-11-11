import numpy as np
from typing import Tuple
from scipy.sparse import coo_array

from core.mesh import Mesh
from core.fields import FaceField, FlowField

def face_velocity_interpolation(
        fields: FlowField,
        faces: FaceField,
        mesh: Mesh,
        Apu: np.ndarray,
        Apw: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    dx = mesh.dx
    dz = mesh.dz
    Nx = mesh.Nx
    Nz = mesh.Nz
    V = dz*dx
    uface: np.ndarray
    wface: np.ndarray

    Apu = Apu.reshape(Nz, Nx)
    Apw = Apw.reshape(Nz, Nx)

    du = V / Apu
    dw = V / Apw

    du_bar = 0.5*(du[:, :-1] + du[:, 1:])
    dw_bar = 0.5*(dw[:-1, :] + dw[1:, :])
    rho_bar_q = 0.5*(fields.rho[:, :-1] + fields.rho[:, 1:])  #face interpolated density
    rho_bar_w = 0.5*(fields.rho[:-1, :] + fields.rho[1:, :])  #face interpolated density

    Cq_internal = rho_bar_q * dz / dx * du_bar
    Cw_internal = rho_bar_w * dx / dz * dw_bar

    dpdx_bar = 0.5*(fields.dpdx[:, :-1] + fields.dpdx[:, 1:])
    px_face = (fields.p[:, 1:] - fields.p[:, :-1]) / dx

    dpdz_bar = 0.5*(fields.dpdz[:-1, :] + fields.dpdz[1:, :])
    pz_face = (fields.p[1:, :] - fields.p[:-1, :]) / dz

    ubar = 0.5*(fields.uq_star[:, :-1] + fields.uq_star[:, 1:]) # velocity interpolated to interior faces i & i+1
    wbar = 0.5*(fields.w_star[:-1, :] + fields.w_star[1:, :])


    uface = ubar - du_bar*(px_face - dpdx_bar)
    wface = wbar - dw_bar*(pz_face - dpdz_bar)

    ### FLUX CORRECTION TO FIELD VARS.

    faces.flux_q_star[:, 1:-1] = rho_bar_q * uface * dz # fluxes in x-dir excluding boundaries
    faces.flux_w_star[1:-1, :] = rho_bar_w * wface * dx


    return uface, wface, Cq_internal, Cw_internal


def assemble_pressure_corrector(
    mesh,
    flux_q_star,
    flux_w_star,
    Cq,
    Cw,
):


    Cq_full = np.zeros((mesh.Nz, mesh.Nx+1))
    Cw_full = np.zeros((mesh.Nz+1, mesh.Nx))

    Cq_full[:, 1:-1] = Cq
    Cw_full[1:-1, :] = Cw

    Ce = Cq_full[:, 1:]
    Cwe = Cq_full[:, :-1]
    Cs = Cw_full[1:, :]
    Cn = Cw_full[:-1, :]

    aP_pressure = Ce + Cwe + Cn + Cs
    aE_pressure = Ce
    aW_pressure = Cwe
    aN_pressure = Cn
    aS_pressure = Cs

    mass_imbalance = (
    flux_q_star[:, 1:] - flux_q_star[:, :-1]
    + flux_w_star[1:, :] - flux_w_star[:-1, :])


    Ap = _build_sparse_matrix(aE_pressure, aW_pressure, aS_pressure, aN_pressure, aP_pressure, mesh.Nx, mesh.Nz)
    bp = -mass_imbalance
    App = aP_pressure.copy()

    return Ap, bp, Cq_full, Cw_full


def scalar_pressure_gradient(pressure_field, dx, dz):

    dpdz, dpdx = np.gradient(
        pressure_field,
        dz,
        dx,
        edge_order=1,
    )
    return dpdx, dpdz

def _build_sparse_matrix(aE, aW, aS, aN, aP, Nx, Nz):
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
