import numpy as np
import matplotlib.pyplot as plt

from core.mesh import Mesh
from core.fields import FlowField
from core.settings import SolveSettings, PlotSettings

# def plot_temperature(fields: FlowField):

#     fig, ax = plt.subplots()
#     ax.pcolormesh(fields.T, cmap='inferno')
#     plt.show()


def plot_temperature(fields: FlowField):
    mesh = fields.mesh

    x_edges = np.arange(mesh.Nx + 1) * mesh.dx * 1e6
    z_edges = np.arange(mesh.Nz + 1) * mesh.dz * 1e6

    fig, ax = plt.subplots(figsize=(8, 5), layout="constrained")

    image = ax.pcolormesh(
        x_edges,
        z_edges,
        fields.T,
        shading="flat",
        cmap="inferno",
    )

    ax.set(
        xlabel="x [μm]",
        ylabel="Depth [μm]",
        title="Temperature field",
    )
    ax.set_aspect("equal")
    ax.invert_yaxis()

    fig.colorbar(image, ax=ax, label="Temperature [K]")

    return fig, ax