# packages
import os
import argparse

# Local classes
from core import mesh
from core.mesh import Mesh
from core.fields import FlowField, FaceField
# from core.material import Material
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
    x = 0.9*1e-3# in [m]
    y = 0.5*1e-3
    z = 0.5*1e-3
    mesh: Mesh = Mesh(x, y, z, args.nx, 1, args.nz, args.dt)
    # material = Material()
    solverSettings = SolveSettings(0.25, 50)
    plotSettings = PlotSettings()

    return solverSettings, plotSettings, mesh
    # take args and throw them into the solver settings, mesh, etc.


def main(solveSettings: SolveSettings, plotSettings: PlotSettings, mesh: Mesh):
    Nq = mesh.Nx
    Nz = mesh.Nz
    fields = FlowField(mesh, Nq, Nz)            # initialize fields
    faces = FaceField(mesh)
    energy = TemperatureModule(solveSettings, fields, faces, mesh, mesh.dt)
    simpleSolve =  SIMPLEcfd(solveSettings, fields, faces, mesh, mesh.dt)    

    t = 0
    steady_state = False
    depth = -1
    i = 0
    n = 100
    while not steady_state:
        print(f"Entered main loop, t = {t}")

        fields.update_material_properties()
        energy.laser_update(current_t = t)        
        fields.update_material_properties()

        fields.calculate_temperature_gradients()

        simpleSolve.run(100)
        energy.solve_energy()

        depthNew = energy.compute_depth()
        # print(depthNew)
        t += mesh.dt

        # if depthNew == depth:
        #     steady_state = True
        # else:
        #     depth = depthNew
        
        if i > n: steady_state = True
        i += 1

    return fields


if __name__ == "__main__":
    
    
    solverSettings, plotSettings, mesh = setup()

    F = main(
        solveSettings= solverSettings,
        plotSettings= plotSettings,
        mesh = mesh,
        # material = material
    )
    plot_temperature(F)