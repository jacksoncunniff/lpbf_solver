import numpy as np
from scipy.sparse.linalg import spsolve

from core.settings import SolveSettings

from core.mesh import Mesh
from solver.ADI import ADI
from core.fields import FlowField, FaceField
from core.boundary_conditions import BoundaryCondition

from discretization.momentum import MomentumAssembler
from discretization.pressure import assemble_pressure_corrector, face_velocity_interpolation, scalar_pressure_gradient

from typing import Literal, Optional
RunModes = Literal["ADI", "SPSOLVE"]

class SIMPLEcfd:
    """
    Take settings from main execution -> set up domain, mesh, etc.

    run loops until convergence
    follow method on paper

    """

    def __init__(self, 
                 settings: SolveSettings, 
                 fields: FlowField,
                 faces: FaceField,
                 mesh: Mesh,
                 dt: float,
                 mode: Optional[RunModes] = "SPSOLVE"
                 ):

        self.dx = mesh.dx
        self.dz = mesh.dz
        self.dy = mesh.dy
        self.Nx = mesh.Nx
        self.Nz = mesh.Nz
        self.N = self.Nx * self.Nz
        self.mode = mode
        self.fields = fields
        self.mesh = mesh
        self.faces = faces
        self.dt = dt
        self.settings = settings
        self.assembler = MomentumAssembler(self.mesh)

    def _check_system(self, name, A, rhs):
        A = A.tocsr()
        diagonal = A.diagonal()
        rhs = np.asarray(rhs).ravel()

        print(f"\n{name}")
        print("Matrix finite:", np.all(np.isfinite(A.data)))
        print("RHS finite:", np.all(np.isfinite(rhs)))
        print("Minimum diagonal:", np.min(diagonal))
        print("Zero diagonals:", np.count_nonzero(diagonal == 0))
        print("Empty rows:", np.count_nonzero(np.diff(A.indptr) == 0))

        assert np.all(np.isfinite(A.data))
        assert np.all(np.isfinite(rhs))
        assert np.all(diagonal != 0)    

    def _solve_momentum(self):
        shape = (self.Nz, self.Nx)
        Au, bu, aPu, aE_u, aW_u, aS_u, aN_u, aPu_, bu_ = self.assembler.assemble_u(self.fields, self.faces, self.dt)
        Aw, bw, aPw, aE_w, aW_w, aS_w, aN_w, aPw_, bw_ = self.assembler.assemble_w(self.fields, self.faces, self.dt)

        if self.mode == "SPSOLVE":
            # self._check_system("uq-star", Au, bu)
            # self._check_system("w-star", Aw, bw)
            uq_star = spsolve(Au, bu)
            w_star = spsolve(Aw, bw)
            self.fields.uq_star[:] = uq_star.reshape(shape)
            self.fields.w_star[:] = w_star.reshape(shape)

        else:
            uq_star = ADI(aP = aPu_, 
                aE = aE_u, 
                aW = aW_u,
                aS = aS_u,
                aN = aN_u,
                rhs = bu_,
                field_of_interest=self.fields.uq,
                Nx = self.mesh.Nx,
                Nz = self.mesh.Nz
                )

            w_star = ADI(aP = aPw_, 
                        aE = aE_w, 
                        aW = aW_w,
                        aS = aS_w,
                        aN = aN_w,
                        rhs = bw_,
                        field_of_interest=self.fields.w,
                        Nx = self.mesh.Nx,
                        Nz = self.mesh.Nz
                        )

            self.fields.uq_star[:] = uq_star
            self.fields.w_star[:] = w_star

        # residual_q = bu - Au @ uq_star
        # residual_w = bw - Aw @ w_star
        # return aPu, aPw, residual_q, residual_w
        return aPu, aPw

    def _refresh_gradients(self):
        f = self.fields

        f.dpdx[:], f.dpdz[:] = scalar_pressure_gradient(
            f.p, self.dx, self.dz
        )

        f.dudx[:], f.dudz[:] = scalar_pressure_gradient(
            f.uq, self.dx, self.dz
        )

        dwdx, _ = scalar_pressure_gradient(
            f.w, self.dx, self.dz
        )
        f.dwdx[:] = dwdx

    def _solve_pressure_correction(self, Cp, Cw):
        """Routine to solve pressure corrections after finding rhie chow face velocities

        Returns:
            _type_: _description_
        """

        Ap, bp, Cq_full, Cw_full, aP_pressure, aE_pressure, aW_pressure, aN_pressure, aS_pressure, bp = assemble_pressure_corrector(
            self.mesh,
            self.faces.flux_q_star,
            self.faces.flux_w_star,
            Cp,
            Cw,
        )
        Nz = self.mesh.Nz
        Nx = self.mesh.Nx
        k = (Nz-1)*Nx + 1
        Ap = Ap.tolil(copy=True)
        Ap[:, k] = 0
        Ap[k, :] = 0
        Ap[k, k] = 1
        bp.ravel()[k] = 0
        Ap = Ap.tocsr(copy=True)

        if self.mode == "SPSOLVE":
            pprime_flat = spsolve(Ap, bp.ravel())
            pprime = pprime_flat.reshape((self.mesh.Nz, self.mesh.Nx))
            # numerical_residual = bp - Ap @ pprime_flat
        else:
            pprime = ADI(aP_pressure, aE_pressure, aW_pressure, aN_pressure, aS_pressure, bp, field_of_interest=np.zeros_like(self.fields.p), Nx = self.mesh.Nx, Nz = self.mesh.Nz)
            # pprime_flat = spsolve(Ap, bp.ravel())
            # pprime = pprime_flat.reshape((self.mesh.Nz, self.mesh.Nx))
            
            # pprime = pprime_flat.reshape((self.mesh.Nz, self.mesh.Nx))            


        return pprime, Cq_full, Cw_full

    def _correct_fluxes(self, pprime, Cq_full, Cw_full):

        self.faces.flux_q[:] = self.faces.flux_q_star
        self.faces.flux_w[:] = self.faces.flux_w_star

        # increment internal horizontal faces.
        self.faces.flux_q[:, 1:-1] -= Cq_full[:, 1:-1] * (
            pprime[:, 1:] - pprime[:, :-1]
        )

        # " " internal vertical faces. MUST KEEP EXTERNAL FLUXES 0
        # BOUNDARY CONDITIONS ENFORCE NO NORMAL FLUX
        self.faces.flux_w[1:-1, :] -= Cw_full[1:-1, :] * (
            pprime[1:, :] - pprime[:-1, :]
        )

        pass

    def _correct_cell_velocity(self, aPu, aPw, pprime):
        V = self.mesh.dx * self.mesh.dz
        du = V / aPu.reshape((self.mesh.Nz, self.mesh.Nx))
        dw = V / aPw.reshape((self.mesh.Nz, self.mesh.Nx))

        dPp_dx, dPp_dz = scalar_pressure_gradient(pprime, self.mesh.dx, self.mesh.dz)

        self.fields.uq[:] = self.fields.uq_star - du * dPp_dx
        self.fields.w[:] = self.fields.w_star - dw * dPp_dz

    def _correct_pressure(self, relaxation_factor, pprime):
        
        self.fields.p += relaxation_factor * pprime

        pass

    def _check_convergence(self,criteria = 1e-4, iteration = 0) -> bool:
    
        if iteration < 3:
            return False
        
        V = self.mesh.dx * self.mesh.dz


        continuity = self._continuity_residual()
        continuity_max_norm = np.max(np.abs(continuity))
        continuity_L2_norm = np.linalg.norm(continuity)

        # conditions to non-dimensionalize the residuals
        rho_ref = 1.0
        u_ref = 1.0 
        L_ref = 1.0
        mu_ref = 0.01

        momentum_scaling = (rho_ref*u_ref**2/L_ref + mu_ref*u_ref/L_ref**2)

        Ru, Rw = self._momentum_residual()
        Ru_max = np.max(np.abs(Ru/V))
        Ru_L2_norm = np.linalg.norm(Ru/V)
        Rw_max = np.max(Rw/V)
        Rw_L2_norm = np.linalg.norm(Rw/V)

        momentum_errors = max(Ru_max/momentum_scaling, Rw_max/momentum_scaling)
        continuity_errors = continuity_max_norm / (V*rho_ref) * L_ref / u_ref
        errors = np.array([momentum_errors, continuity_errors])

        # print(f"SIMPLE Iter Convergence: {iteration} | {errors.max()}")

        if np.all(errors < criteria):
            return bool(np.isfinite(errors).all() and np.all(errors < criteria))
        else:
            return False

    def _continuity_residual(self):

        flux_e = self.faces.flux_q[:, 1:]
        flux_w = self.faces.flux_q[:, :-1]
        flux_s = self.faces.flux_w[1:, :]
        flux_n = self.faces.flux_w[:-1, :]
        r = (flux_e - flux_w) + (flux_s - flux_n)

        return r

    def _momentum_residual(self):
        shape = (self.mesh.Nz, self.mesh.Nx)

        Au, bu, aPu, aE_u, aW_u, aS_u, aN_u, aPu_, bu_ = self.assembler.assemble_u(self.fields, self.faces, self.dt)
        Aw, bw, aPw, aE_w, aW_w, aS_w, aN_w, aPw_, bw_ = self.assembler.assemble_w(self.fields, self.faces, self.dt)

        Ru = (
            np.asarray(bu).ravel()
            - Au @ self.fields.uq.ravel()
        ).reshape(shape)

        Rw = (
            np.asarray(bw).ravel()
            - Aw @ self.fields.w.ravel()
        ).reshape(shape)

        return Ru, Rw

    def run(self, maxIters):
        """Executes SIMPLE loop at one time step
        """
        converged = False
        iteration = 0
        while not converged:

            self._refresh_gradients()
            aPu, aPw = self._solve_momentum()
            uface, wface, Cq, Cw = face_velocity_interpolation(
                self.fields,
                self.faces,
                self.mesh,
                aPu,    
                aPw
            )

            pprime, Cq_full, Cw_full = self._solve_pressure_correction(Cq, Cw)
            self._correct_fluxes(pprime, Cq_full, Cw_full)

            self._correct_cell_velocity(aPu, aPw, pprime)
            self._correct_pressure(self.settings.relaxation_factor, pprime)
            self._refresh_gradients()

            converged = self._check_convergence(iteration=iteration)
            iteration += 1
            if iteration > maxIters:
                break

