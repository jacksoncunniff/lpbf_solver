# packages
import os
import argparse
import copy
import time
from typing import Optional

# Local classes
from core import mesh
from core.mesh import Mesh
from core.fields import FlowField, FaceField
# from core.material import Material
from core.boundary_conditions import BoundaryCondition
from core.settings import SolveSettings, PlotSettings

from solver.energy import TemperatureModule
from solver.simple import SIMPLEcfd
from solver.simple import RunModes

from postprocessing.visualize import *

def setup():
    parser = argparse.ArgumentParser(

    )
    parser.add_argument("--nx", type=int, default=100)
    parser.add_argument("--nz", type=int, default=50)
    parser.add_argument("--dt", type=float, default=1e-6)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--mode", type=str, default="SPSOLVE")

    args = parser.parse_args()
    x = 0.9*1e-3# in [m]
    y = 0.5*1e-3
    z = 0.5*1e-3
    steps = args.steps
    mesh: Mesh = Mesh(x, y, z, args.nx, 1, args.nz, args.dt)
    # material = Material()
    solverSettings = SolveSettings(0.25, 50)
    plotSettings = PlotSettings()

    return solverSettings, plotSettings, mesh, steps, args.mode

def main(solveSettings: SolveSettings, plotSettings: PlotSettings, mesh: Mesh, steps: int, mode: Optional[RunModes] = None):
    # Initialize variables and fields
    Nq = mesh.Nx
    Nz = mesh.Nz
    fields = FlowField(mesh, Nq, Nz)
    faces = FaceField(mesh)

    # Setup solver projects
    energy = TemperatureModule(solveSettings, fields, faces, mesh, mesh.dt)
    
    simpleSolve =  SIMPLEcfd(solveSettings, fields, faces, mesh, mesh.dt, mode=mode)  

    # Run time variables
    t = 0
    steady_state = False
    i = 0
    n = steps

    # Main CFD execution loop
   
    while not steady_state:
        print(f"Step: {i+1}\nTime: {t}\n")

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
        
        i += 1
        if i > n: steady_state = True

    return fields

def time_mode(solveSettings, plotSettings, mesh, steps, mode: RunModes):
    start = time.perf_counter()
    main(
        solveSettings= solveSettings,
        plotSettings= plotSettings,
        mesh = mesh,
        steps=steps,
        mode="SPSOLVE"
    )
    end = time.perf_counter()
    print(f"Run: {mode} in {end-start:.4f}")

if __name__ == "__main__":
    

    solverSettings, plotSettings, mesh, steps, mode = setup()

    # time_mode(
    #     solveSettings= solverSettings,
    #     plotSettings= plotSettings,
    #     mesh = mesh,
    #     steps=steps,
    #     mode="SPSOLVE"
    # )

    # time_mode(
    #     solveSettings= solverSettings,
    #     plotSettings= plotSettings,
    #     mesh = mesh,
    #     steps=steps,
    #     mode="ADI"
    # )


    F = main(
        solveSettings= solverSettings,
        plotSettings= plotSettings,
        mesh = mesh,
        steps=steps,
        mode=mode
    )


    fig, ax = plot_temperature(F)
    plt.show()

    fig, ax = plot_velocity(F)
    plt.show()
