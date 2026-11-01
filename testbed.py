"""
This file is intended to run and test output of individual classes.
In particular, the momentum assembly is being tested for small meshes to 
valildate coefficients for SIMPLE.
"""

# packages
import os
import timeit
import argparse
import warnings
import numpy as np

from statistics import median
from scipy.sparse import issparse
from scipy.sparse.linalg import spsolve
from scipy.sparse.linalg import MatrixRankWarning

# Local classes
from core.material import Material
from core.mesh import Mesh
from core.fields import FlowField, FaceField
from core.settings import SolveSettings, PlotSettings
from discretization.momentum import MomentumAssembler

from solver.simple import SIMPLEcfd


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
    fields.w[:] = 5.0
    fields.p[:] = 0.0
    fields.uq_old[:] = 4.0
    fields.w_old[:] = 4.0
    fields.liquid_frac[:] = 0
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

    A_u2, b_u2, A_diag2 = momentum.assemble_u(
        fields=fields,
        faces=faces,
        dt=dt
    )

    A_w, b_w, Aw_diag = momentum.assemble_w(
        fields=fields,
        faces=faces,
        dt=dt)

#     benchmark_u_assembly(
#     momentum=momentum,
#     fields=fields,
#     faces=faces,
#     dt=dt,
# )

    compare_assembly(
        A_old = A_u,
        b_old = b_u,
        A_new = A_u2,
        b_new = b_u2,
        name ="X - New"
    )

    check_assembly = False
    if check_assembly:
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

        print("\n--- W-MOMENTUM ASSEMBLY TEST ---")
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

def compare_assembly(
    A_old: np.ndarray,
    b_old: np.ndarray,
    A_new: np.ndarray,
    b_new: np.ndarray,
    name: str
) -> None:

    # Convert to dense NumPy arrays if needed
    A_old = np.asarray(A_old)
    A_new = np.asarray(A_new)

    b_old = np.asarray(b_old)
    b_new = np.asarray(b_new)

    # Ensure dimensions match
    assert A_old.shape == A_new.shape
    assert b_old.shape == b_new.shape

    # Calculate numerical differences
    A_error = np.abs(A_old - A_new)
    b_error = np.abs(b_old - b_new)

    print(f"\n--- {name} ASSEMBLY COMPARISON ---")
    print(f"Maximum A difference: {A_error.max():.6e}")
    print(f"Maximum b difference: {b_error.max():.6e}")

    # Compare with numerical tolerances
    np.testing.assert_allclose(
        A_old, A_new,
        rtol=1e-10,
        atol=1e-7
    )

    np.testing.assert_allclose(
        b_old, b_new,
        rtol=1e-10,
        atol=1e-12
    )

    print("PASS: Assemblies agree")

def benchmark_u_assembly(
    momentum, fields, faces, dt, repeats=7, calls_per_repeat=20
):
    methods = {
        "Loop": momentum.assemble_u_old,
        "Vectorized": momentum.assemble_u,
    }
    timings = {}

    for name, method in methods.items():
        def assemble(method=method):
            return method(fields=fields, faces=faces, dt=dt)

        # Warm up before collecting timings.
        for _ in range(3):
            assemble()

        batch_times = timeit.repeat(
            stmt=assemble,
            repeat=repeats,
            number=calls_per_repeat,
        )

        timings[name] = [
            elapsed / calls_per_repeat
            for elapsed in batch_times
        ]

    print(
        f"\nU-momentum assembly: "
        f"{momentum.mesh.Nx} × {momentum.mesh.Nz}"
    )
    print(f"{repeats} batches of {calls_per_repeat} calls per method")

    for name, samples in timings.items():
        print(
            f"{name:12s} "
            f"median = {median(samples) * 1e3:.4f} ms/call, "
            f"minimum = {min(samples) * 1e3:.4f} ms/call"
        )

    speedup = median(timings["Loop"]) / median(timings["Vectorized"])
    print(f"Loop time / vectorized time: {speedup:.2f}×")


def test_simple_closed_box(perturb_pressure=False):
    Nx, Nz = 8, 12
    dt = 0.01

    mesh = Mesh(
        Lx=1.0, Ly=1.0, Lz=1.0,
        Nx=Nx, Ny=Nx, Nz=Nz,
        dt=dt,
    )

    fields = FlowField(mesh=mesh, Nq=Nx, Nz=Nz)
    faces = FaceField(mesh=mesh)

    # Rest state with no thermal or surface forcing.
    for name in (
        "uq", "w", "uq_old", "w_old",
        "uq_star", "w_star", "p", "p_prime",
        "dpdx", "dpdz", "dudx", "dwdx", "dudz",
        "dTdx", "dTdz", "dgamma_dT",
    ):
        getattr(fields, name).fill(0.0)

    fields.rho.fill(1.0)
    fields.mu.fill(0.01)
    fields.liquid_frac.fill(1.0)

    if perturb_pressure:
        j, i = np.indices((Nz, Nx))
        fields.p[:] = 1e-3 * (-1.0) ** (i + j)
        fields.p[:] -= fields.p.mean()

    settings = SolveSettings(
        convergence_depth_delta=0.0,
        simple_iterations_max=100,
        mode="test",
        relaxation_factor=0.5,
    )

    solver = SIMPLEcfd(
        settings=settings,
        fields=fields,
        faces=faces,
        mesh=mesh,
        dt=dt,
        material=Material(),
    )

    # Fail immediately on singular pressure systems or invalid arithmetic.
    with warnings.catch_warnings():
        warnings.simplefilter("error", MatrixRankWarning)

        with np.errstate(over="raise", divide="raise", invalid="raise"):
            solver.run(maxIters=50)

    for name, array in (
        ("uq", fields.uq),
        ("w", fields.w),
        ("p", fields.p),
        ("flux_q", faces.flux_q),
        ("flux_w", faces.flux_w),
    ):
        assert np.isfinite(array).all(), f"{name} is nonfinite"

    continuity = (
        faces.flux_q[:, 1:] - faces.flux_q[:, :-1]
        + faces.flux_w[1:, :] - faces.flux_w[:-1, :]
    )

    # Suggested tolerances for this simple nondimensional test.
    assert np.max(np.abs(continuity)) < 1e-10
    assert np.max(np.abs(fields.uq)) < 1e-6
    assert np.max(np.abs(fields.w)) < 1e-6

    # Pressure should become spatially constant; its offset is arbitrary.
    assert np.ptp(fields.p) < 1e-6

    np.testing.assert_allclose(
        faces.flux_q[:, [0, -1]], 0.0, rtol=0, atol=1e-14
    )
    np.testing.assert_allclose(
        faces.flux_w[[0, -1], :], 0.0, rtol=0, atol=1e-14
    )

    # SIMPLE must not advance the previous-time fields.
    assert np.all(fields.uq_old == 0.0)
    assert np.all(fields.w_old == 0.0)

    print(f"PASS: closed box, pressure perturbation={perturb_pressure}")

if __name__ == "__main__":
    test_simple_closed_box()
    test_simple_closed_box(True)
    