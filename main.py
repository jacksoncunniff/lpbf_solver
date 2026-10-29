# packages
import numpy as np
import os
import argparse


# Local classes
from core.mesh import Mesh
from core.fields import FlowField, FaceField
from core.material import Material
from core.boundary_conditions import BoundaryConditions
from core.settings import SolverSettings

from discretization.momentum import u_link_coefficients, w_link_coefficients
from discretization.pressure import pressure_correction
from discretization.interpolation import face_fluxes

from solver.simple import SIMPLEcfd

from physics.heat_source import LaserSource
from physics.material_properties import update_material_properties


def main():
    parser = argparse.ArgumentParser(

    )

    parser.add_argument("--nx", type=int, default=100)
    parser.add_argument("--nz", type=int, default=50)
    parser.add_argument("--dt", type=float, default=1e-6)
    parser.add_argument("--steps", type=int, default=1000)

    args = parser.parse_args()


if __name__ == "__main__":
    main()