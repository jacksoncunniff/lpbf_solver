"""

This file is intended to run and test output of individual classes.
In particular, the momentum assembly is being tested for small meshes to 
valildate coefficients for SIMPLE.


"""


# packages
import numpy as np
import argparse
import os
from scipy.sparse import issparse
from scipy.sparse.linalg import spsolve


# Local classes
from core.material import Material
from core.mesh import Mesh
from core.fields import FlowField, FaceField
from core.settings import SolveSettings, PlotSettings

from discretization.momentum import MomentumAssembler
# from solver.simple import SIMPLEcfd
# from physics.heat_source import LaserSource



def main():
    parser = argparse.ArgumentParser(

    )

    parser.add_argument("--nx", type=int, default=100)
    parser.add_argument("--nz", type=int, default=50)
    parser.add_argument("--dt", type=float, default=1e-6)
    parser.add_argument("--steps", type=int, default=1000)

    args = parser.parse_args()
    solverSettings = SolveSettings(
                    convergence_depth_delta=0,
                    simple_iterations_max=0,
                    mode="test"
    )

    Nx = args.nx
    Nz = args.nz
    dt = args.dt
    steps = args.steps

    mesh = Mesh(1, 1, 1, Nx, Nx, Nz, dt)

    fields = FlowField(mesh=mesh, Nq=Nx, Nz=Nz)

    # Initialize a simple, uniform test case.
    fields.uq[:] = 5.0
    fields.w[:] = 5
    fields.p[:] = 0.0
    fields.uq_old[:] = 4.0

    fields.rho[:] = 4000.0
    fields.mu[:] = 0.0004

    # Zero initial convection fluxes: diffusion-only test.
    faces = FaceField(
        flux_q=np.zeros((Nz, Nx + 1)),
        flux_w=np.zeros((Nz + 1, Nx))
    )


    material = Material()       # empty material definition just for testing

    momentum = MomentumAssembler(
        mesh=mesh,
        material=material,
        boundary_conditions=None
    )

    A_u, b_u, A_diag = momentum.assemble_u(
        fields=fields,
        faces=faces,
        dt=dt
    )

    # --------------------------------------------------
    # 1. Check matrix dimensions
    # --------------------------------------------------
    N = Nx * Nz

    assert A_u.shape == (N, N), (
        f"Incorrect matrix shape: {A_u.shape}"
    )

    assert b_u.shape == (N,), (
        f"Incorrect RHS shape: {b_u.shape}"
    )

    # A_u = A_u.tocsr() if issparse(A_u) else A_u

    # --------------------------------------------------
    # 2. Inspect matrix properties
    # --------------------------------------------------
    diagonal = A_u.diagonal()

    print("\n--- X-MOMENTUM ASSEMBLY TEST ---")
    print(f"Grid:             {Nx} x {Nz}")
    print(f"Total cells:      {N}")
    print(f"Matrix shape:     {A_u.shape}")
    # print(f"Matrix nonzeros:  {np.count_nonzero(A_u) if not issparse(A_u) else A_u.nnz}")
    print(f"Matrix nonzeros:  {np.count_nonzero(A_u)}")
    print(f"Minimum diagonal: {diagonal.min():.6e}")
    print(f"Maximum diagonal: {diagonal.max():.6e}")
    print(f"RHS minimum:      {b_u.min():.6e}")
    print(f"RHS maximum:      {b_u.max():.6e}")

    # --------------------------------------------------
    # 3. Basic consistency checks
    # --------------------------------------------------
    assert np.all(np.isfinite(diagonal))
    assert np.all(np.isfinite(b_u))

    negative_cells = np.where(diagonal <= 0)[0]

    print("\nCells with nonpositive diagonal:")

    for k in negative_cells:
        j, i = divmod(k, Nx)

        print(
            f"k={k}, i={i}, j={j}, "
            f"aP={diagonal[k]:.6e}"
        )

    assert np.all(diagonal > 0), (
        "X-momentum matrix contains nonpositive diagonal entries."
    )

    # Check an interior cell
    i = Nx // 2
    j = Nz // 2
    k = j * Nx + i

    print(f"\nInterior cell: (i={i}, j={j}), k={k}")
    print(f"aP = {A_u[k, k]:.6e}")
    print(f"aE = {-A_u[k, k+1]:.6e}")
    print(f"aW = {-A_u[k, k-1]:.6e}")
    print(f"aN = {-A_u[k, k-Nx]:.6e}")
    print(f"aS = {-A_u[k, k+Nx]:.6e}")
    print(f"bP = {b_u[k]:.6e}")

    # --------------------------------------------------
    # 4. Solve the assembled equation
    # --------------------------------------------------
    u_flat = np.asarray(spsolve(A_u, b_u))

    u_test = u_flat.reshape((Nz, Nx))

    residual = A_u @ u_flat - b_u
    residual_norm = np.linalg.norm(residual)

    print("\n--- LINEAR SOLVE ---")
    print(f"Minimum u:       {u_test.min():.6e}")
    print(f"Maximum u:       {u_test.max():.6e}")
    print(f"Residual norm:   {residual_norm:.6e}")

    print("\nAssembly test completed.")




if __name__ == "__main__":
    main()