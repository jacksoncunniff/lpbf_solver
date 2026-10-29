import numpy as np
from dataclasses import dataclass, field

@dataclass
class Mesh:
    # dimensions
    Lx: float
    Ly: float
    Lz: float

    # cell counts per direction
    Nx: int
    Ny: int
    Nz: int

    # cell spacing
    dx: float = field(init=False)
    dy: float = field(init=False)
    dz: float = field(init=False)

    # Coordinate arrays
    x: np.ndarray = field(init=False)
    y: np.ndarray = field(init=False)
    z: np.ndarray = field(init=False)

    def __post_init__(self):
        self.dx = self.Lx / self.Nx
        self.dy = self.Ly / self.Ny
        self.dz = self.Lz / self.Nz

        self.x = (np.arange(self.Nx) + 0.5) * self.dx
        self.y = (np.arange(self.Ny) + 0.5) * self.dy
        self.z = (np.arange(self.Nz) + 0.5) * self.dz