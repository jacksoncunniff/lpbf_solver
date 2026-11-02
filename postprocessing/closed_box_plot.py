"""Plot diagnostics from test_simple_closed_box without changing solver fields.

Dependencies: numpy, matplotlib.
Suggested destination in lpbf_solver: postprocessing/closed_box.py.

Usage inside test_simple_closed_box, after solver.run(...):

    from postprocessing.closed_box import plot_closed_box
    fig = plot_closed_box(
        mesh, fields, faces,
        output_path="results/closed_box.png",
        show=True,
    )

Coordinates are in metres, velocity in m/s, and physical pressure in Pa.
Mass-flux units follow the solver's out-of-plane depth convention. With
unit-depth 2D fluxes, interpret the continuity residual per unit depth.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from matplotlib.figure import Figure

__all__ = ["continuity_residual", "plot_closed_box"]


def _checked_array(name, values, shape=None):
    array = np.asarray(values, dtype=float)
    if shape is not None and array.shape != shape:
        raise ValueError(f"{name}: expected shape {shape}, got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains NaN or infinity")
    return array


def continuity_residual(flux_q, flux_w):
    """Return signed net outward mass flux for every cell, shape (Nz, Nx)."""
    q = _checked_array("flux_q", flux_q)
    if q.ndim != 2 or q.shape[0] < 1 or q.shape[1] < 2:
        raise ValueError("flux_q must have shape (Nz, Nx + 1), with Nx, Nz >= 1")
    nz, nx = q.shape[0], q.shape[1] - 1
    w = _checked_array("flux_w", flux_w, (nz + 1, nx))
    return q[:, 1:] - q[:, :-1] + w[1:, :] - w[:-1, :]


def plot_closed_box(
    mesh,
    fields,
    faces,
    *,
    output_path: str | Path | None = None,
    show: bool = False,
    title: str = "SIMPLE closed-box diagnostics",
    dpi: int = 180,
) -> Figure:
    """Return a six-panel Matplotlib figure; optionally save and/or display it.

    Required inputs:
        mesh: Nx, Nz, dx, dz (uniform cell-centred mesh, origin x=z=0).
        fields: p, uq, w, each shape (Nz, Nx).
        faces: flux_q shape (Nz, Nx+1), flux_w shape (Nz+1, Nx).

    If both predicted flux arrays exist, they appear alongside the corrected
    continuity residual. They refer to the most recent predictor, not the
    initial condition. Missing predicted arrays are clearly labelled.

    Positive z and w point downwards. Velocity arrows are scaled for visibility;
    colorbars and panel maxima show their actual magnitudes. The two continuity
    panels share one color scale. No input array is modified, no solver methods
    are called, and no global plotting backend is selected here.
    """
    import matplotlib.pyplot as plt

    nx, nz = int(mesh.Nx), int(mesh.Nz)
    dx, dz = float(mesh.dx), float(mesh.dz)
    if nx < 1 or nz < 1 or not np.isfinite([dx, dz]).all() or min(dx, dz) <= 0:
        raise ValueError("Mesh counts and spacings must be positive and finite")

    shape = (nz, nx)
    p = _checked_array("fields.p", fields.p, shape)
    u = _checked_array("fields.uq", fields.uq, shape)
    w = _checked_array("fields.w", fields.w, shape)
    q = _checked_array("faces.flux_q", faces.flux_q, (nz, nx + 1))
    zflux = _checked_array("faces.flux_w", faces.flux_w, (nz + 1, nx))
    corrected = continuity_residual(q, zflux)

    predicted = None
    qstar = getattr(faces, "flux_q_star", None)
    wstar = getattr(faces, "flux_w_star", None)
    if (qstar is None) != (wstar is None):
        raise ValueError("Provide both predicted face-flux arrays, or neither")
    if qstar is not None:
        predicted = continuity_residual(
            _checked_array("faces.flux_q_star", qstar, (nz, nx + 1)),
            _checked_array("faces.flux_w_star", wstar, (nz + 1, nx)),
        )

    x_edges = np.arange(nx + 1) * dx
    z_edges = np.arange(nz + 1) * dz
    x = (np.arange(nx) + 0.5) * dx
    z = (np.arange(nz) + 0.5) * dz
    pressure = p - np.mean(p)
    speed = np.hypot(u, w)
    velocity_limit = max(float(np.max(np.abs(u))), float(np.max(np.abs(w))))
    residual_limit = float(np.max(np.abs(corrected)))
    if predicted is not None:
        residual_limit = max(residual_limit, float(np.max(np.abs(predicted))))

    with plt.rc_context({"font.size": 10, "axes.titlesize": 11}):
        fig, axes = plt.subplots(2, 3, figsize=(13.8, 8.6), layout="constrained")
        fig.suptitle(f"{title} | {nx} x {nz} cells", fontsize=16)

        def panel(ax, data, label, unit, *, limit=None, positive=False):
            magnitude = float(np.max(np.abs(data)))
            bound = magnitude if limit is None else limit
            # Avoid a degenerate color range for an exactly zero rest state.
            bound = bound if bound > 0 else 1.0
            artist = ax.pcolormesh(
                x_edges, z_edges, data,
                shading="flat", cmap="viridis" if positive else "RdBu_r",
                vmin=0.0 if positive else -bound, vmax=bound,
                rasterized=True,
            )
            ax.set_title(f"{label}\nmax |value| = {magnitude:.3e}")
            ax.set_xlabel("x [m]")
            ax.set_ylabel("z [m], positive down")
            ax.set_xlim(x_edges[0], x_edges[-1])
            ax.set_ylim(z_edges[-1], z_edges[0])
            ax.set_aspect("equal", adjustable="box")
            colorbar = fig.colorbar(artist, ax=ax, shrink=0.85, pad=0.025)
            colorbar.set_label(unit)
            colorbar.formatter.set_powerlimits((-2, 2))
            colorbar.update_ticks()

        panel(axes[0, 0], pressure, "Pressure minus spatial mean", "Pa")
        panel(axes[0, 1], u, "Horizontal velocity u", "m/s", limit=velocity_limit)
        panel(axes[0, 2], w, "Vertical velocity w", "m/s", limit=velocity_limit)
        panel(axes[1, 0], speed, "Speed (arrows scaled for visibility)", "m/s", positive=True)

        max_speed = float(np.max(speed))
        if max_speed > 0:
            stride = max(1, int(np.ceil(max(nx, nz) / 16)))
            X, Z = np.meshgrid(x, z)
            selection = (slice(None, None, stride), slice(None, None, stride))
            axes[1, 0].quiver(
                X[selection], Z[selection], u[selection], w[selection],
                angles="xy", scale_units="xy",
                scale=max_speed / (0.075 * min(nx * dx, nz * dz)),
                color="white", edgecolor="black", linewidth=0.25, width=0.004,
            )
        else:
            axes[1, 0].text(
                0.5, 0.5, "Exactly zero velocity",
                transform=axes[1, 0].transAxes, ha="center", va="center",
                color="white", bbox={"facecolor": "black", "alpha": 0.5},
            )

        if predicted is None:
            axes[1, 1].set_axis_off()
            axes[1, 1].text(
                0.5, 0.5, "Predicted fluxes were not provided",
                transform=axes[1, 1].transAxes, ha="center", va="center",
            )
        else:
            panel(axes[1, 1], predicted, "Predicted continuity imbalance",
                  "Net outward mass flux (solver units)", limit=residual_limit)
        panel(axes[1, 2], corrected, "Corrected continuity imbalance",
              "Net outward mass flux (solver units)", limit=residual_limit)

        if output_path is not None:
            destination = Path(output_path).expanduser()
            destination.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(destination, dpi=dpi, facecolor="white")
        if show:
            plt.show()
        return fig
