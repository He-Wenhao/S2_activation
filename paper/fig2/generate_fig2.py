from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from numpy.polynomial.legendre import leggauss


# ============================================================
# Quadrature node layouts and surface-attached weight disks
# on S^2 for lmax = 4.
#
# This version:
#   - keeps only the "back disks hidden" version,
#   - hides the back-side sphere grid as well,
#   - draws each weight as a small tangent disk attached to S^2,
#   - uses disk area proportional to quadrature weight.
#
# Notes:
#   - Driscoll-Healy weights here are schematic latitude-dependent
#     area weights, not the exact Fourier-form DH weights.
#   - The Lebedev panel uses a Fibonacci-sphere proxy for visualizing
#     non-tensor, roughly uniform nodes; it is not a true Lebedev rule.
# ============================================================

lmax = 4

# Viewing angle
ELEV = 22
AZIM = 35


def view_direction(elev_deg=ELEV, azim_deg=AZIM):
    """
    Approximate viewing direction in data coordinates for Matplotlib 3D.

    Points with normal p satisfying p dot VIEW > cutoff are treated as visible.
    """
    elev = np.deg2rad(elev_deg)
    azim = np.deg2rad(azim_deg)
    v = np.array([
        np.cos(elev) * np.cos(azim),
        np.cos(elev) * np.sin(azim),
        np.sin(elev),
    ])
    return v / np.linalg.norm(v)


VIEW = view_direction()


def is_front(xyz, cutoff=-0.05):
    """
    Decide whether unit-sphere points are on the visible/front hemisphere.

    A slightly negative cutoff keeps near-limb features visible.
    """
    return xyz @ VIEW > cutoff


def sph_to_xyz(theta, phi):
    """
    Convert spherical coordinates to Cartesian coordinates on the unit sphere.

    theta: colatitude in [0, pi]
    phi: longitude in [0, 2pi)
    """
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    return np.column_stack([x, y, z])


def style_sphere(ax, title, elev=ELEV, azim=AZIM):
    ax.set_title(title, fontsize=11)
    ax.set_box_aspect([1, 1, 1])
    ax.set_axis_off()
    ax.view_init(elev=elev, azim=azim)

    lim = 1.18
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(-lim, lim)


def plot_sphere_line_segments(ax, pts, visible, **plot_kwargs):
    """
    Plot only contiguous visible segments of a 3D polyline.
    """
    pts = np.asarray(pts)
    visible = np.asarray(visible, dtype=bool)

    if not np.any(visible):
        return

    idx = np.where(visible)[0]
    breaks = np.where(np.diff(idx) > 1)[0] + 1
    groups = np.split(idx, breaks)

    # Merge wrapped visible segments for closed latitude rings.
    if len(groups) > 1 and groups[0][0] == 0 and groups[-1][-1] == len(visible) - 1:
        merged = np.concatenate([groups[-1], groups[0]])
        groups = [merged] + groups[1:-1]

    for group in groups:
        if len(group) < 2:
            continue
        seg = pts[group]
        ax.plot(seg[:, 0], seg[:, 1], seg[:, 2], **plot_kwargs)


def plot_front_sphere_grid(ax, n_lat=17, n_lon=36, cutoff=-0.05):
    """
    Draw only the visible/front-side latitude-longitude grid of the unit sphere.
    """
    theta_vals = np.linspace(0.08 * np.pi, 0.92 * np.pi, n_lat)
    phi_curve = np.linspace(0.0, 2 * np.pi, 360)

    for theta in theta_vals:
        theta_curve = np.full_like(phi_curve, theta)
        pts = sph_to_xyz(theta_curve, phi_curve)
        visible = is_front(pts, cutoff=cutoff)
        plot_sphere_line_segments(ax, pts, visible, linewidth=0.25, alpha=0.18)

    phi_vals = np.linspace(0.0, 2 * np.pi, n_lon, endpoint=False)
    theta_curve = np.linspace(0.0, np.pi, 360)

    for phi in phi_vals:
        phi_curve = np.full_like(theta_curve, phi)
        pts = sph_to_xyz(theta_curve, phi_curve)
        visible = is_front(pts, cutoff=cutoff)
        plot_sphere_line_segments(ax, pts, visible, linewidth=0.25, alpha=0.18)


def tangent_basis(n):
    """
    Return two orthonormal tangent vectors at unit normal n.
    """
    n = np.asarray(n, dtype=float)

    if abs(n[2]) < 0.9:
        anchor = np.array([0.0, 0.0, 1.0])
    else:
        anchor = np.array([1.0, 0.0, 0.0])

    e1 = np.cross(n, anchor)
    e1 /= np.linalg.norm(e1)

    e2 = np.cross(n, e1)
    e2 /= np.linalg.norm(e2)

    return e1, e2


def scale_radii(weights, r_min=0.025, r_max=0.105):
    """
    Convert quadrature weights to tangent-disk radii.

    Radius scales as sqrt(weight), so disk area is proportional to weight.
    """
    w = np.asarray(weights, dtype=float)

    if np.allclose(w.max(), w.min()):
        return np.full_like(w, 0.07, dtype=float)

    s = (w - w.min()) / (w.max() - w.min())
    return r_min + np.sqrt(s) * (r_max - r_min)


def tangent_disk(center, radius, nseg=32, lift=0.008):
    """
    Create a small polygonal disk tangent to the sphere at center.

    lift moves the disk slightly outward to avoid z-fighting with the grid.
    """
    n = center / np.linalg.norm(center)
    e1, e2 = tangent_basis(n)

    angles = np.linspace(0.0, 2 * np.pi, nseg, endpoint=False)
    c = center + lift * n

    verts = [
        c + radius * (np.cos(angle) * e1 + np.sin(angle) * e2)
        for angle in angles
    ]
    return verts


def add_disk_collection(ax, xyz, radii, alpha=0.72, linewidth=0.35):
    """
    Add tangent disks to a 3D axis.
    """
    if len(xyz) == 0:
        return

    polys = [tangent_disk(p, r) for p, r in zip(xyz, radii)]

    coll = Poly3DCollection(
        polys,
        alpha=alpha,
        linewidths=linewidth,
        edgecolors="black",
    )

    ax.add_collection3d(coll)


def plot_weight_disks_front_only(ax, xyz, weights, title):
    """
    Plot visible/front hemisphere grid and only front-side weight disks.
    """
    plot_front_sphere_grid(ax)

    radii = scale_radii(weights)
    front = is_front(xyz)

    add_disk_collection(
        ax,
        xyz[front],
        radii[front],
        alpha=0.72,
        linewidth=0.35,
    )

    style_sphere(ax, title)


def build_dh_panel():
    n_theta_dh = 2 * (lmax + 1)
    n_phi_dh = 2 * n_theta_dh

    theta_dh = np.linspace(
        0.5 * np.pi / n_theta_dh,
        np.pi - 0.5 * np.pi / n_theta_dh,
        n_theta_dh,
    )
    phi_dh = np.linspace(0.0, 2 * np.pi, n_phi_dh, endpoint=False)

    th_dh, ph_dh = np.meshgrid(theta_dh, phi_dh, indexing="ij")
    xyz_dh = sph_to_xyz(th_dh.ravel(), ph_dh.ravel())

    ring_w_dh = np.sin(theta_dh)
    ring_w_dh = ring_w_dh / ring_w_dh.sum()

    weights_dh = np.repeat(ring_w_dh / n_phi_dh, n_phi_dh)
    weights_dh = weights_dh / weights_dh.sum() * (4 * np.pi)
    return xyz_dh, weights_dh


def build_gl_panel():
    n_theta_gl = lmax + 1
    mu_gl, w_mu_gl = leggauss(n_theta_gl)

    order = np.argsort(-mu_gl)
    mu_gl = mu_gl[order]
    w_mu_gl = w_mu_gl[order]

    theta_gl = np.arccos(mu_gl)
    n_phi_gl = 2 * (lmax + 1)
    phi_gl = np.linspace(0.0, 2 * np.pi, n_phi_gl, endpoint=False)

    th_gl, ph_gl = np.meshgrid(theta_gl, phi_gl, indexing="ij")
    xyz_gl = sph_to_xyz(th_gl.ravel(), ph_gl.ravel())

    weights_gl = np.repeat(w_mu_gl * (2 * np.pi / n_phi_gl), n_phi_gl)
    weights_gl = weights_gl / weights_gl.sum() * (4 * np.pi)
    return xyz_gl, weights_gl


def build_lebedev_proxy_panel(n_leb):
    idx = np.arange(n_leb)
    golden_angle = np.pi * (3 - np.sqrt(5))

    z = 1 - 2 * (idx + 0.5) / n_leb
    r = np.sqrt(1 - z * z)
    phi_leb = idx * golden_angle

    xyz_leb = np.column_stack([
        r * np.cos(phi_leb),
        r * np.sin(phi_leb),
        z,
    ])

    weights_leb = np.full(n_leb, 4 * np.pi / n_leb)
    return xyz_leb, weights_leb


def main():
    xyz_dh, weights_dh = build_dh_panel()
    xyz_gl, weights_gl = build_gl_panel()
    xyz_leb, weights_leb = build_lebedev_proxy_panel(xyz_gl.shape[0])

    fig = plt.figure(figsize=(12.5, 4.2), dpi=180)

    panels = [
        (
            "Driscoll--Healy weights\nfront surface only",
            xyz_dh,
            weights_dh,
        ),
        (
            "Gauss--Legendre weights\nfront surface only",
            xyz_gl,
            weights_gl,
        ),
        (
            "Lebedev-style proxy weights\nfront surface only",
            xyz_leb,
            weights_leb,
        ),
    ]

    for idx, (title, xyz, weights) in enumerate(panels, start=1):
        ax = fig.add_subplot(1, 3, idx, projection="3d")
        plot_weight_disks_front_only(ax, xyz, weights, title)

    fig.suptitle(
        r"Surface-attached quadrature weight disks on $S^2$ for $\ell_{\max}=4$",
        fontsize=14,
    )

    fig.text(
        0.5,
        0.02,
        "Only front-side disks and front-side sphere grid are shown. "
        "Disk area is proportional to quadrature weight. "
        "Driscoll--Healy weights are schematic; Lebedev uses a "
        "Fibonacci-sphere proxy.",
        ha="center",
        fontsize=9,
    )

    fig.tight_layout(rect=[0, 0.06, 1, 0.92])

    out_dir = Path(__file__).resolve().parent
    out_png = out_dir / "fig_quadratures.png"
    out_pdf = out_dir / "fig_quadratures.pdf"

    fig.savefig(out_png, bbox_inches="tight", dpi=220)
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)

    print(out_png)
    print(out_pdf)


if __name__ == "__main__":
    main()
