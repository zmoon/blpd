"""
Create plots of LPD model results.

The functions here all take `xr.Dataset`s as their first argument.
"""
from itertools import cycle

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection as _LineCollection
from mpl_toolkits.mplot3d import Axes3D as _Axes3D  # noqa: F401 unused import

from . import utils  # TODO: move all calls to namespaced form since I am using a lot now
from .utils import _add_snippets

# ^ could add `as _{}` to all of these
# to indicate that these are not intended to be public parts of the module name space
# since __all__ is not respected by linters or autocompleters

__all__ = (
    "conc_2d",
    "conc_scatter",
    "conc_xline",
    "final_pos_hist",
    "final_pos_hist2d",
    "final_pos_scatter",
    "trajectories",
    "ws_hist",
)

# TODO: create some base classes for plots to reduce repeating of code
#       - allow passing fig and ax kwargs in __init__
#       - stars to mark sources; check_fig_num, labeling, etc.
#       - alpha calculation based on number of particles & spread?

# TODO: really all plots could have the auto-bounds stuff. and (optionally?) print message about it?


_SOURCE_MARKER_PROPS = dict(marker="*", c="gold", ms=11, mec="0.35", mew=1.0)

utils._SNIPPETS.update(
    spc_params="""
spc : str
    ASCII identifier of a chemical species. For example, `'apinene'`.
    """.strip(),
    ds_lpd_params="""
ds : xr.Dataset
    Dataset of Lagrangian particle positions, e.g., from `blpd.model.Model.to_xr`.
    """.strip(),
    ds_chem_params="""
ds : xr.Dataset
    Dataset of Lagrangian particle positions + chemistry, e.g., from
    `blpd.chem.calc_relative_levels_fix_oxidants`.
    """.strip(),
    log_cnorm_params="""
log_cnorm : bool
    Whether to use a logarithmic colormap normalization.
    """.strip(),
)


@_add_snippets
def final_pos_scatter(ds, sdim="xy"):
    """Scatter plot of particle end positions.

    Parameters
    ----------
    %(ds_lpd_params)s
    %(sdim_params)s
    """
    p = utils.load_p(ds)

    dims = utils.dims_from_sdim(sdim)
    if len(dims) == 3:
        sdim = "xyz"
        x = ds.x.values
        y = ds.y.values
        z = ds.z.values
        subplot_kw = {"projection": "3d"}
        coords = (x, y, z)
        plot_kw = dict(alpha=0.5, mew=0, ms=7)
    elif len(dims) == 2:
        x = ds[dims[0]].values
        y = ds[dims[1]].values
        subplot_kw = {}
        coords = (x, y)
        plot_kw = dict(alpha=0.5, mfc="none", mew=0.8, ms=5)
    else:
        raise ValueError("invalid `sdim`")

    num = utils.check_fig_num(f"final-pos-scatter-{sdim}")
    fig, ax = plt.subplots(num=num, subplot_kw=subplot_kw)

    ax.plot(*coords, "o", **plot_kw)

    ax.set_xlabel(f"${dims[0]}$")
    ax.set_ylabel(f"${dims[1]}$")
    if len(dims) == 3:
        ax.set_zlabel(f"${dims[2]}$")
    ax.set_title(utils.s_t_info(p), loc="left")
    ax.set_title(f"$N_p = {p['Np_tot']}$", loc="right")

    for xs, ys in p["source_positions"]:
        sp = dict(x=xs, y=ys, z=p["release_height"])
        if len(dims) == 3:
            ax.plot([sp[dims[0]]], [sp[dims[1]]], [sp[dims[2]]], "*", c="gold", ms=10)
        else:
            ax.plot(sp[dims[0]], sp[dims[1]], "*", c="gold", ms=10)

    fig.set_tight_layout(True)


# TODO: add option to do trajectories for continuous release runs, colored by time out
# TODO: trajectories for hist run in 3-D?


@_add_snippets
def trajectories(ds, *, smooth=False, smooth_window_size=None, color_sources=False):
    """Plot particle trajectories.

    Parameters
    ----------
    %(ds_lpd_params)s
    smooth : bool
        Whether to smooth the trajectories.
    smooth_window_size : int, optional
        How many points to include in the moving average window.
    color_sources : bool
        Whether to color particle trajectories by their sources
        to help in differentiating them.
    """
    if "t" not in ds.dims:
        raise ValueError("time 't' must be a dimension")

    p = utils.load_p(ds)
    pos = np.stack((ds.x.values, ds.y.values, ds.z.values), axis=-1)  # same shape as hist["pos"]

    t_tot = p["t_tot"]
    dt = p["dt_out"]  # note output timestep not that of the integration
    # N_t = p["N_t"]
    Np = p["Np_tot"]
    # N = Np * N_t
    assert pos.shape[0] == Np

    ltitle = utils.s_t_info(p)
    rtitle = utils.s_sample_size(p)

    # allow specifying smooth_window_size only
    if smooth_window_size is not None and not smooth:
        smooth = True
    if smooth:
        if smooth_window_size is None:
            n = 100
        else:
            n = int(smooth_window_size)

        if n * dt > 0.5 * t_tot:
            raise ValueError(
                "can't do any smoothing with the requested window size (not enough points)"
            )
        pos0 = pos[:, 0, :][:, np.newaxis, :]
        pos = np.swapaxes(utils.moving_average(np.swapaxes(pos, 0, 1), n=n, axis=0), 0, 1)
        # ^ `np.swapaxes` should return views, not create new arrays (ver >= 1.10.0)

        # preserve starting point
        pos = np.concatenate((pos0, pos), axis=1)  # axis=1 => basically `hstack`

        ltitle = f"$N_{{smooth}} = {n}$ ({n*dt:} s)\n{ltitle}"  # add smoothing info to left title

    num = utils.check_fig_num("trajectories")
    fig, ax = plt.subplots(num=num)

    if color_sources:
        if isinstance(color_sources, (list, np.ndarray)):
            colors = color_sources  # assume is a list of colors (TODO: should check)
        else:
            colors = plt.get_cmap("Dark2").colors
            # Dark2 is a ListedColormap with 8 colors; `plt.cm.Dark2` same but pylint complains plt.cm `no-member` Dark2

        N_sources = p["N_sources"]
        for i_source, color in zip(range(N_sources), cycle(colors)):
            segs = [pos[i, :, :2] for i in range(i_source, Np, N_sources)]

            lc = _LineCollection(
                segs,
                linewidths=0.5,
                colors=color,
                linestyles="solid",
                alpha=0.3,
            )
            ax.add_collection(lc)

    else:
        segs = [pos[i, :, :2] for i in range(Np)]

        lc = _LineCollection(
            segs,
            linewidths=0.5,
            colors="0.6",
            linestyles="solid",
            alpha=0.5,
        )
        ax.add_collection(lc)

    for x, y in p["source_positions"]:
        ax.plot(x, y, **_SOURCE_MARKER_PROPS)

    ax.autoscale()  # `ax.add_collection` won't do this automatically
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    ax.set_title(ltitle, loc="left")
    ax.set_title(rtitle, loc="right")

    fig.set_tight_layout(True)


@_add_snippets
def conc_scatter(
    ds,
    spc="apinene",
    sdim="xy",
    *,
    cmap="gnuplot",
    log_cnorm=False,
    vmax=100,
    vmin=None,
    ax=None,
):
    """Plot species relative level in particle as a scatter (2- or 3-d).

    Parameters
    ----------
    %(ds_chem_params)s
    %(spc_params)s
    %(sdim_params)s
    %(log_cnorm_params)s
    """
    p = utils.load_p(ds)

    dims = utils.dims_from_sdim(sdim)
    if len(dims) == 3:
        sdim = "xyz"
        x = ds.x.values
        y = ds.y.values
        z = ds.z.values
        subplot_kw = {"projection": "3d"}
        coords = (x, y, z)
        # plot_kw = dict(alpha=0.5, mew=0, ms=7)
    elif len(dims) == 2:
        x = ds[dims[0]].values
        y = ds[dims[1]].values
        subplot_kw = {}
        coords = (x, y)
        # plot_kw = dict(alpha=0.5, mfc="none", mew=0.8, ms=5)
    else:
        raise ValueError("invalid `sdim`")

    conc = ds.f_r.sel(spc=spc).values
    spc_display_name = str(ds.display_name.sel(spc=spc).values)

    num = utils.check_fig_num(f"horizontal-end-positions-with-conc_{spc}_scatter_{sdim}")
    fig, ax = utils.maybe_new_figure(num, ax=ax, subplot_kw=subplot_kw)

    norm, _ = utils.maybe_log_cnorm(log_cnorm=log_cnorm, levels=None, vmin=vmin, vmax=vmax)

    im = ax.scatter(
        *coords,
        c=conc,
        s=7,
        marker="o",
        alpha=0.4,
        linewidths=0,
        cmap=cmap,
        norm=norm,
    )
    cb = fig.colorbar(im, ax=ax, drawedges=False)
    cb.set_label(f"{spc_display_name} relative conc. (%)")

    for xs, ys in p["source_positions"]:
        sp = dict(x=xs, y=ys, z=p["release_height"])
        ax.plot(*tuple(sp[dim] for dim in dims), **_SOURCE_MARKER_PROPS)

    ax.set_title(utils.s_t_info(p), loc="left")
    ax.set(
        xlabel=f"{ds[dims[0]].attrs['long_name']} [{ds[dims[0]].attrs['units']}]",
        ylabel=f"{ds[dims[1]].attrs['long_name']} [{ds[dims[1]].attrs['units']}]",
    )
    if len(dims) == 3:
        ax.set_zlabel(f"{ds[dims[2]].attrs['long_name']} [{ds[dims[2]].attrs['units']}]")
    fig.set_tight_layout(True)


@_add_snippets
def conc_2d(
    ds,
    spc="apinene",
    *,
    bins=(20, 10),
    plot_type="pcolor",
    levels=30,  # for contourf
    cmap="gnuplot",
    log_cnorm=False,
    vmax=100,
    vmin=None,
    ax=None,
):
    """Plot species 2-d (*xy*) binned average species relative level.

    Parameters
    ----------
    %(ds_chem_params)s
    %(spc_params)s
    %(bins_params)s
    plot_type : {'pcolor', 'contourf'}
    levels : int
        Only used for plot type `'contourf'`.
    %(log_cnorm_params)s
    """
    p = utils.load_p(ds)
    X = ds.x.values
    Y = ds.y.values
    conc = ds.f_r.sel(spc=spc).values
    spc_display_name = str(ds.display_name.sel(spc=spc).values)

    if "x" not in ds.dims:
        # We were passed the particles dataset, need to do the binning
        binned = utils.bin_values_xy(X, Y, conc, bins=bins)
    else:
        raise NotImplementedError("already binned?")

    fig, ax = utils.maybe_new_figure(f"horizontal-end-positions-with-conc_{spc}_{plot_type}", ax=ax)

    norm, locator = utils.maybe_log_cnorm(
        log_cnorm=log_cnorm,
        levels=levels if plot_type == "contourf" else None,
        vmin=vmin,
        vmax=vmax,
    )

    if plot_type == "pcolor":
        im = ax.pcolormesh(binned.xe, binned.ye, binned.v, cmap=cmap, norm=norm)
    elif plot_type == "contourf":
        im = ax.contourf(
            binned.x, binned.y, binned.v, levels, cmap=cmap, norm=norm, locator=locator
        )
    else:
        raise ValueError("`plot_type` should be 'pcolor' or 'contourf'")

    cb = fig.colorbar(im, ax=ax, drawedges=False)
    cb.set_label(f"{spc_display_name} relative conc. (%)")

    for x, y in p["source_positions"]:
        ax.plot(x, y, **_SOURCE_MARKER_PROPS)

    ax.autoscale(enable=True, axis="both", tight=True)
    ax.set_title(utils.s_t_info(p), loc="left")
    ax.set(
        xlabel=f"{ds.x.attrs['long_name']} [{ds.x.attrs['units']}]",
        ylabel=f"{ds.y.attrs['long_name']} [{ds.y.attrs['units']}]",
    )
    fig.set_tight_layout(True)


@_add_snippets
def conc_xline(
    ds,
    spc="apinene",
    y=0.0,
    *,
    dy=1.0,
    ax=None,  # TODO: select z as well?
    legend=True,
    label=None,
    legend_title="Chemical species",
):
    r"""Plot species average relative level in the *x*-direction at a certain approximate y value.
    `spc` can be ``'all'`` (all will fit in one plot since we are plotting lines).

    Parameters
    ----------
    %(ds_chem_params)s
    %(spc_params)s
    y : float
        *y* value of our "line".
    dy : float
        We will include Lagrangian particles within $[-\delta y / 2, \delta y / 2]$.
    """
    p = utils.load_p(ds)
    X = ds.x.values
    Y = ds.y.values

    fig, ax = utils.maybe_new_figure(f"horizontal-end-positions-with-conc_{spc}_line", ax=ax)

    spc_to_plot = ds.spc.values if spc == "all" else [spc]

    # Define bins (edges)
    xe = utils._auto_bins_1d(X, nx_max=100, std_mult=2.0, method="sqrt")
    ye = np.r_[y - 0.5 * dy, y + 0.5 * dy]
    # ze = np.r_[z - 0.5*dz, z + 0.5*dz]
    bins = [xe, ye]

    for spc in spc_to_plot:
        spc_display_name = str(ds.display_name.sel(spc=spc).values)
        conc = ds.f_r.sel(spc=spc).values
        binned = utils.bin_values_xy(X, Y, conc, bins=bins)
        ax.plot(binned.x, binned.v.squeeze(), label=spc_display_name if label is None else label)

    for xs, ys in p["source_positions"]:
        if abs(ys - y) <= 5:
            ax.plot(xs, np.nanmin(binned.v), **_SOURCE_MARKER_PROPS)

    if legend:
        fig.legend(
            ncol=2,
            fontsize="small",
            title=legend_title,
        )
    ax.set_title(utils.s_t_info(p), loc="left")
    ax.set(
        xlabel=f"{ds.x.attrs['long_name']} [{ds.x.attrs['units']}]",
        ylabel="Relative concentration",
    )
    fig.set_tight_layout(True)


# TODO: x-y (or u-v) hists for different height (z) bins


@_add_snippets
def ws_hist(
    ds,
    *,
    bounds=None,
):
    """Plot histograms of particle wind speed components (one subplot for each).
    For a single- or continuous-release run.

    Parameters
    ----------
    %(ds_lpd_params)s
    bounds : 2-tuple(float), optional
        If provided, used as wind speed limits for all histograms.
    """
    p = utils.load_p(ds)
    u_all = np.ravel(ds.u.values)  # or can do `.ravel()`
    v_all = np.ravel(ds.v.values)
    w_all = np.ravel(ds.w.values)

    num = utils.check_fig_num("ws-hist-all")
    fig, axs = plt.subplots(3, 1, num=num, sharex=True)

    if bounds is None:
        bins = 100
    else:
        bins = np.linspace(bounds[0], bounds[1], 100)

    labels = ["$u$", "$v$", "$w$"]
    for i, (ui, ax) in enumerate(zip([u_all, v_all, w_all], axs.flat)):
        ax.hist(ui, bins)
        ax.text(0.01, 0.98, labels[i], va="top", ha="left", fontsize=13, transform=ax.transAxes)

    if bounds:
        axs[0].set_xlim(bounds)

    axs[0].set_title(utils.s_t_info(p), loc="left")
    axs[0].set_title(utils.s_sample_size(p), loc="right")
    axs[0].set_xlabel(f"[{ds.u.attrs['units']}]")
    fig.set_tight_layout(True)


@_add_snippets
def final_pos_hist(
    ds,
    *,
    bounds=None,
):
    """Plot histograms of final position components (*x*, *y*, and *z*).

    Parameters
    ----------
    %(ds_lpd_params)s
    bounds : 2-tuple(float), optional
        If provided, used as position component limits for all histograms.
    """
    p = utils.load_p(ds)
    if "t" in ds.dims:
        xf = ds.x.isel(t=-1).values
        yf = ds.y.isel(t=-1).values
        zf = ds.z.isel(t=-1).values
    else:
        xf = ds.x.values
        yf = ds.y.values
        zf = ds.z.values

    num = utils.check_fig_num("final-pos-hist")
    fig, axs = plt.subplots(3, 1, num=num, sharex=True)

    if bounds is None:
        bins = 100
    else:
        bins = np.linspace(bounds[0], bounds[1], 100)

    labels = ["$x$", "$y$", "$z$"]
    for i, (xi, ax) in enumerate(zip([xf, yf, zf], axs.flat)):
        ax.hist(xi, bins)
        ax.text(0.01, 0.98, labels[i], va="top", ha="left", fontsize=13, transform=ax.transAxes)

    if bounds:
        axs[0].set_xlim(bounds)

    axs[0].set_title(utils.s_t_info(p), loc="left")
    axs[0].set_title(utils.s_sample_size(p, N_p_only=True), loc="right")
    axs[-1].set_xlabel(f"[{ds.x.attrs['units']}]")
    fig.set_tight_layout(True)


@_add_snippets
def final_pos_hist2d(
    ds,
    *,
    sdim="xy",
    bins=50,
    plot_type="pcolor",
    log_cnorm=False,
    vmax=None,
):
    """Plot 2-d histogram of selected final position components.

    Parameters
    ----------
    %(ds_lpd_params)s
    %(sdim_params)s
    %(bins_params)s
    %(log_cnorm_params)s
    """
    p = utils.load_p(ds)

    dims = utils.dims_from_sdim(sdim)
    if "t" in ds.dims:
        x = ds[dims[0]].isel(t=-1).values
        y = ds[dims[1]].isel(t=-1).values
    else:
        x = ds[dims[0]].values
        y = ds[dims[1]].values

    num = utils.check_fig_num(f"final-pos-hist-{sdim}")
    fig, ax = plt.subplots(num=num)

    if bins == "auto":
        bins = utils.auto_bins_xy([x, y])

    if log_cnorm:
        norm = mpl.colors.LogNorm(vmin=1.0, vmax=vmax)
    else:
        norm = mpl.colors.Normalize(vmin=1.0, vmax=vmax)

    _, _, _, im = ax.hist2d(x, y, bins=bins, norm=norm)
    # ^ returns h (nx, ny), xedges, yedges, image

    cb = fig.colorbar(im, ax=ax)
    cb.set_label("Particle count")

    for x, y in p["source_positions"]:
        ax.plot(x, y, "*", c="gold", ms=11, mec="0.35", mew=1.0)

    ax.set_xlabel(f"${dims[0]}$ [{ds.x.attrs['units']}]")
    ax.set_ylabel(f"${dims[1]}$ [{ds.x.attrs['units']}]")
    ax.autoscale(enable=True, axis="both", tight=True)
    ax.set_title(utils.s_t_info(p), loc="left")
    ax.set_title(utils.s_sample_size(p, N_p_only=True), loc="right")
    fig.set_tight_layout(True)
