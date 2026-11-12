# packages
import os
import argparse

# Local classes
from core import mesh
from core.mesh import Mesh
from core.fields import FlowField, FaceField
from core.material import Material
from core.boundary_conditions import BoundaryCondition
from core.settings import SolveSettings, PlotSettings

from solver.energy import TemperatureModule
from solver.simple import SIMPLEcfd

from postprocessing.visualize import *

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
    solverSettings = SolveSettings(0.25, 50)
    plotSettings = PlotSettings()

    return solverSettings, plotSettings, mesh, material
    # take args and throw them into the solver settings, mesh, etc.


def main(solveSettings: SolveSettings, plotSettings: PlotSettings, mesh: Mesh, material: Material):
    Nq = mesh.Nx
    Nz = mesh.Nz
    fields = FlowField(mesh, Nq, Nz)
    faces = FaceField(mesh)
    energy = TemperatureModule(solveSettings, fields, faces, mesh, mesh.dt, material)
    simpleSolve =  SIMPLEcfd(solveSettings, fields, faces, mesh, mesh.dt, material)    

    t = 0.0
    steady_state = False
    depth = -1
    while not steady_state:
        print(f"Entered main loop, t = {t}")
        fields.update_material_properties()
        energy.laser_update(current_t = t)
        fields.update_material_properties()
        simpleSolve.run(50)
        energy.solve_energy()
        depthNew = energy.compute_depth()
        t += mesh.dt

        if depthNew == depth:
            steady_state = True
        else:
            depth = depthNew


if __name__ == "__main__":
    
    
    solverSettings, plotSettings, mesh, material = setup()

    main(
        solveSettings= solverSettings,
        plotSettings= plotSettings,
        mesh = mesh,
        material = material
    )
    # plot_fields()