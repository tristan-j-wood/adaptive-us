import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle
from mpl_toolkits.mplot3d import art3d


def _is_minimum(values, idx) -> bool:
    """Checks if idx is at the edge of values and checks if it is a minimum"""
    if idx != 0 or idx != len(values) - 1:
        if (values[idx - 1] and values[idx + 1]) > values[idx]:
            return True
    else:
        return False


def _get_minima_idxs(y_vals) -> list:
    """Find the indexes of the minima from a set of data"""
    grad = np.gradient(y_vals)
    idxs = np.where(np.diff(np.sign(grad)) != 0)[0] + 1

    minima_idxs = []
    for idx in idxs:
        if _is_minimum(y_vals, idx):
            minima_idxs.append(idx)

    return minima_idxs


def _add_point(ax, x, y, z, fc='red', ec='red', radius=0.02) -> None:
    """Add a point onto a 3D surface for plotting"""
    xy_len, z_len = ax.get_figure().get_size_inches()

    axis_length = [x[1] - x[0] for x in
                   [ax.get_xbound(), ax.get_ybound(), ax.get_zbound()]]

    axis_rotation = {
        'y': ((x, z, y),
              axis_length[2] / axis_length[0] * xy_len / z_len)}

    for a, ((x0, y0, z0), ratio) in axis_rotation.items():
        p = Circle((x0, y0), radius, fc=fc, ec=ec, linewidth=0.1)
        ax.add_patch(p)
        art3d.pathpatch_2d_to_3d(p, z=z0, zdir=a)

    return None


def _plot_2d_total_energy(x, y, z, ref) -> None:
    """
    Plots a 2D surface of the bias energy + potential energy for a given
    reference value. Annotes the minima on the surface
    """
    plt.close()
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    cmap = plt.cm.get_cmap('plasma')

    x_grid, y_grid = np.meshgrid(x, y)
    ax.plot_surface(x_grid, y_grid, z, cmap=cmap, linewidth=0,
                    antialiased=False)

    for i in range(len(y_grid)):

        minima_idxs = _get_minima_idxs(z[i])
        for idx in minima_idxs:
            _add_point(ax, x[idx], y_grid[i][0], z[i][idx])

        reference_idx = (np.abs(x - ref)).argmin()
        _add_point(ax, ref, y_grid[i][0], z[i][reference_idx], fc='w',
                   ec='w')

    ax.set_xlabel('Reaction coordinate / Å')
    ax.set_ylabel(r'$\kappa~/~eV Å^{-2}$')
    ax.set_zlabel('Total energy / eV')

    plt.savefig('2d_total_energy.pdf')
    plt.close()

    return None


def _plot_1d_total_energy(x, bias_e, pot_e, total_e, ref, k_idx) -> None:
    """
    Plots the bias energy, potential energy and total energy for a given
    kappa and reference value. Annotes the minima on the curve
    """
    plt.close()
    cmap = plt.cm.get_cmap('plasma')

    minima_idxs = _get_minima_idxs(total_e[k_idx])

    plt.plot(x, pot_e, label='Potential energy', color=cmap(0.1))
    plt.plot(x, bias_e[k_idx], label='Bias energy', color=cmap(0.3))
    plt.plot(x, total_e[k_idx], label='Total energy', color=cmap(0.7))

    for idx in minima_idxs:
        plt.plot(x[idx], total_e[k_idx][idx], label='Minima', c='None',
                 marker='o', markerfacecolor='None', markeredgecolor='red',
                 markeredgewidth=1)

    reference_idx = (np.abs(x - ref)).argmin()

    plt.plot(ref, total_e[k_idx][reference_idx], label='Reference',
             c='None', marker='o', markerfacecolor='None',
             markeredgecolor='k', markeredgewidth=1)

    plt.xlabel('Reaction coordinate / Å')
    plt.ylabel('Energy / eV')
    plt.legend()

    plt.savefig('1d_total_energy.pdf')
    plt.close()

    return None
