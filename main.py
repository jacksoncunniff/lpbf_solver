# packages
import numpy as np
import os
import argparse

from sympy import solveset


# Local classes
from core.mesh import Mesh
from core.fields import FlowField, FaceField
from core.material import Material
from core.boundary_conditions import BoundaryConditions
from core.settings import SolveSettings, PlotSettings

from discretization.momentum import MomentumAssembler
from discretization.pressure import pressure_correction
from discretization.interpolation import face_fluxes

from solver.simple import SIMPLEcfd

from physics.heat_source import LaserSource
from physics.material_properties import update_material_properties


def setup():
    parser = argparse.ArgumentParser(

    )

    parser.add_argument("--nx", type=int, default=100)
    parser.add_argument("--nz", type=int, default=50)
    parser.add_argument("--dt", type=float, default=1e-6)
    parser.add_argument("--steps", type=int, default=1000)

    args = parser.parse_args()

    mesh: Mesh = Mesh(1.0, 1.0, 1.0, args.nx, 1, args.nz, args.dt)
    material = Material()
    solverSettings = SolveSettings()
    plotSettings = PlotSettings()

    return solverSettings, plotSettings, mesh, material
    # take args and throw them into the solver settings, mesh, etc.


def main(solveSettings: SolveSettings, plotSettings: PlotSettings, mesh: Mesh, material: Material):

    Nq = mesh.Nx
    Nz = mesh.Nz
    fields = FlowField(mesh, Nq, Nz)
    faces = FaceField(mesh)

    simpleSolve =  SIMPLEcfd(solveSettings, fields, faces, mesh, mesh.dt, material)    
    
    simpleSolve.run()




if __name__ == "__main__":
    main()