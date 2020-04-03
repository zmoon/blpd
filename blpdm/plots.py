"""
Create plots of lpdm results.
"""

__all__ = (
    "conc",
    "trajectories",
    "final_pos_hist",
    "final_pos_hist2d",
    "ws_hist_all",
    "final_pos_scatter",
)

from itertools import cycle

import matplotlib as mpl
from matplotlib.collections import LineCollection as _LineCollection
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import numpy as np
from scipy import stats

# ^ could add `as _{}` to all of these
# to indicate that these are not intended to be public parts of the module name space
# since __all__ is not respected by linters or autocompleters

from .main import chemical_species_data
from .utils import check_fig_num, to_sci_not, sec_to_str, moving_average, s_t_info


# TODO: create some base classes for plots to reduce repeating of code
#       - allow passing fig and ax kwargs in __init__
#       - stars to mark sources; check_fig_num, labeling, etc.
#       - alpha calculation based on number of particles & spread?

# TODO: really all plots could have the auto-bounds stuff. and (optionally?) print message about it?

# TODO: probably should pass model object to the plotting functions, not separate state and p?


def final_pos_scatter(state, p, sdim="xy"):
    """Scatter plot of particle end positions."""
    xpath = state["xp"]
    ypath = state["yp"]
    # zpath = state["zp"]

    Np_tot = p["Np_tot"]
    assert xpath.size == Np_tot

    if sdim in ("xyz", "3d", "3-D"):
        sdim = "xyz"
        x = state["xp"]
        y = state["yp"]
        z = state["zp"]
        subplot_kw = {"projection": "3d"}
        coords = (x, y, z)
        plot_kw = dict(alpha=0.5, mew=0, ms=7)
    elif len(sdim) == 2 and all(sdim_ in ("x", "y", "z") for sdim_ in sdim):
        x = state[f"{sdim[0]}p"]
        y = state[f"{sdim[1]}p"]
        subplot_kw = {}
        coords = (x, y)
        plot_kw = dict(alpha=0.5, mfc="none", mew=0.8, ms=5)
    else:
        raise ValueError("invalid choice of `sdim`")

    dim = list(sdim)

    num = check_fig_num(f"final-pos-scatter-{sdim}")
    fig, ax = plt.subplots(num=num, subplot_kw=subplot_kw)

    ax.plot(*coords, "o", **plot_kw)

    ax.set_xlabel(f"${dim[0]}$")
    ax.set_ylabel(f"${dim[1]}$")
    if subplot_kw:
        ax.set_zlabel(f"${dim[2]}$")
    ax.set_title(s_t_info(p), loc="left")
    ax.set_title(f"$N_p = {Np_tot}$", loc="right")

    # TODO: should make fn for this
    for (xs, ys) in p["source_positions"]:
        sp = dict(x=xs, y=ys, z=p["release_height"])
        if subplot_kw:  # hack for now
            ax.plot([sp[dim[0]]], [sp[dim[1]]], [sp[dim[2]]], "*", c="gold", ms=10)
        else:
            ax.plot(sp[dim[0]], sp[dim[1]], "*", c="gold", ms=10)

    fig.tight_layout()


# TODO: add option to do trajectories for continuous release runs, colored by time out
# TODO: trajectories for hist run in 3-D?


def trajectories(hist, p, *, smooth=False, smooth_window_size=None, color_sources=False):
    """Particle trajectories.

    note: intended to be used for a single-release run
    """
    pos = hist["pos"]

    t_tot = p["t_tot"]
    # dt = p["dt"]
    dt = p["dt_out"]  # use dt from hist, not model integration; TODO: indicate this in the plot?
    N_t = p["N_t"]
    Np = p["Np_tot"]
    N = Np * N_t
    assert pos.shape[0] == Np

    ltitle = s_t_info(p)
    rtitle = f"$N_p = {to_sci_not(Np)}$\n$N = {to_sci_not(N)}$"

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
        pos = np.swapaxes(moving_average(np.swapaxes(pos, 0, 1), n=n, axis=0), 0, 1)
        # ^ `np.swapaxes` should return views, not create new arrays (ver >= 1.10.0)

        # preserve starting point
        pos = np.concatenate((pos0, pos), axis=1)  # axis=1 => basically `hstack`

        ltitle = f"$N_{{smooth}} = {n}$ ({n*dt:} s)\n{ltitle}"  # add smoothing info to left title

    num = check_fig_num("trajectories")
    fig, ax = plt.subplots(num=num)

    if color_sources:
        if isinstance(color_sources, (list, np.ndarray)):
            colors = color_sources  # assume is a list of colors (TODO: should check)
        else:
            colors = plt.get_cmap("Dark2").colors
            # Dark2 is a ListedColormap with 8 colors; `plt.cm.Dark2` same but pylint complains plt.cm `no-member` Dark2

        N_sources = p["N_sources"]
        for j, color in zip(range(N_sources), cycle(colors)):
            segs = [pos[i, :, :2] for i in range(j, Np, N_sources) for j in range(N_sources)]

            lc = _LineCollection(segs, linewidths=0.5, colors=color, linestyles="solid", alpha=0.3,)
            ax.add_collection(lc)

    else:
        segs = [pos[i, :, :2] for i in range(Np)]

        lc = _LineCollection(segs, linewidths=0.5, colors="0.6", linestyles="solid", alpha=0.5,)
        ax.add_collection(lc)

    for (x, y) in p["source_positions"]:
        ax.plot(x, y, "*", c="gold", ms=10)

    ax.autoscale()  # `ax.add_collection` won't do this automatically

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(ltitle, loc="left")
    ax.set_title(rtitle, loc="right")

    fig.tight_layout()


# TODO: much final_pos_hist2d code is repeated here in conc


def conc(
    state,
    p,
    spc="bocimene",  # species to plot: dict key, not display name
    *,
    plot_type="scatter",
    bins=(20, 10),
    levels=30,
    cmap="gnuplot",
    log_cnorm=False,  # change to `log_scale` to make more sense with centerline?
    vmax=100,
    vmin=None,  # allow fair comparison with other plots
    centerline_dy=10,  # width of y bin for centerline plots
):
    """Scatter plot of particle end positions colored by concentration 
    for continuous release runs

    INPUTS
    ------
    spc : str
        species dict key (ASCII format), e.g., 'bocimene'
        or
        'all' (only for `plot_type='centerline'`)

    INPUTS (optional)
    ------
    plot_type : str {'scatter', 'pcolor', 'contourf', 'centerline'}

    """
    xpath = state["xp"]
    ypath = state["yp"]
    zpath = state["zp"]

    X = xpath
    Y = ypath
    Z = zpath

    if plot_type in ("scatter", "pcolor", "contourf"):
        conc = state["conc"][spc]
        spc_display_name = chemical_species_data[spc]["display_name"]

    num = check_fig_num(f"horizontal-end-positions-with-conc_{spc}_{plot_type}")
    fig, ax = plt.subplots(num=num)

    if plot_type == "scatter":
        im = ax.scatter(
            X, Y, c=conc, s=7, marker="o", alpha=0.4, linewidths=0, cmap=cmap, vmin=vmin, vmax=vmax
        )
        # default `s` is 6**2 (default lines.markersize squared)
        # TODO: marker size should be calculated dynamically but also allowed to pass!
    elif plot_type in ("pcolor", "contourf"):
        # binned conc. of floral volatiles depends on both the particle concentration (the passive scalars)
        # and chemical destruction due to oxidation

        # TODO: copied this from the hist2d fn for now. should make a fn to do this...
        if not bins:
            bins = 50
        elif bins == "auto":
            Np = p["Np_tot"]
            xbar, xstd = X.mean(), X.std()
            ybar, ystd = Y.mean(), Y.std()
            mult = 2.0
            nx = min(np.sqrt(Np).astype(int), 100)
            ny = nx
            x_edges = np.linspace(xbar - mult * xstd, xbar + mult * xstd, nx + 1)
            y_edges = np.linspace(ybar - mult * ystd, ybar + mult * ystd, ny + 1)
            bins = [x_edges, y_edges]
            # TODO: fix so that for z we don't go below zero (or just a bit)
        # else:
        # bins = np.linspace(bounds[0], bounds[1], 50)

        # 1. concentration of lpd particles
        H, xedges, yedges = np.histogram2d(X, Y, bins=bins)  # H is binned particle count
        conc_p_rel = (H / H.max()).T  # TODO: really should divide by level at source (closest bin?)

        # 2. chemistry
        ret = stats.binned_statistic_2d(X, Y, conc, statistic="mean", bins=bins)
        conc_c = ret.statistic.T  # it is returned with dim (nx, ny), we need y to be rows (dim 0)
        x = ret.x_edge
        y = ret.y_edge
        # ^ these are cell edges
        xc = x[:-1] + 0.5 * np.diff(x)
        yc = y[:-1] + 0.5 * np.diff(y)
        # ^ these are cell centers

        assert np.allclose(x, xedges)
        assert np.allclose(y, yedges)
        # TODO: find a way to not hist by x,y more than once (here we have done it 2x)

        z = conc_p_rel * conc_c

        # copied from hist2d
        if log_cnorm:
            norm = mpl.colors.LogNorm(vmin=vmin, vmax=vmax)

            # https://matplotlib.org/3.1.3/gallery/images_contours_and_fields/contourf_log.html
            # https://matplotlib.org/3.1.3/api/ticker_api.html#matplotlib.ticker.LogLocator
            # locator = mpl.ticker.LogLocator(subs=(0.25, 0.5, 1.0))  # another way to get more levels in between powers of 10
            nlevels = levels if isinstance(levels, int) else np.asarray(levels).size
            locator = mpl.ticker.LogLocator(subs="all", numticks=nlevels)
            # TODO: although this^ works, the ticks are not all getting labeled. need to fix.

        else:
            norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
            locator = None

        if plot_type == "pcolor":
            im = ax.pcolormesh(x, y, z, cmap=cmap, norm=norm)
        elif plot_type == "contourf":
            im = ax.contourf(xc, yc, z, levels, cmap=cmap, norm=norm, locator=locator)

        ax.set_xlim((x[0], x[-1]))
        ax.set_ylim((y[0], y[-1]))

    elif plot_type == "centerline":
        # raise NotImplementedError("Yo")
        if spc == "all":
            spc_to_plot = state["conc"].keys()
            n_sp = len(p["source_positions"])
            plt.close(fig)
            fig, axs = plt.subplots(n_sp, 1, num=num)
            ax = axs.flat[0]
        else:
            spc_to_plot = [spc]

        # add subplots if necessary (one for each source pos)
        # axs = []
        # for i, source_pos in enumerate(p["source_positions"]):
        #     if i == 0:
        #         ax_ = ax
        #     else:
        #         ax_ = fig.add_subplot(len(p["source_positions"]), 1, i+1, sharex=ax)
        #     axs.append(ax_)

        for spc in spc_to_plot:
            conc = state["conc"][spc]

            for i, source_pos in enumerate(p["source_positions"]):

                ax_ = axs.flat[i]

                x0_source, y0_source = source_pos

                # only one bin in y
                dy = centerline_dy
                y_edges = np.r_[y0_source - 0.5 * dy, y0_source + 0.5 * dy]

                # TODO: binning and such copied from 2d plot part. needs DRYing

                # x bins same as the 2d plots
                # if bins == "auto":
                Np = p["Np_tot"]
                xbar, xstd = X.mean(), X.std()
                mult = 2.0
                nx = min(np.sqrt(Np).astype(int), 100)
                x_edges = np.linspace(xbar - mult * xstd, xbar + mult * xstd, nx + 1)
                # elif isinstance(bins, int):

                bins = [x_edges, y_edges]

                # 1. concentration of lpd particles
                H, xedges, yedges = np.histogram2d(X, Y, bins=bins)  # H is binned particle count
                conc_p_rel = (H / H.max()).T

                # 2. chemistry
                ret = stats.binned_statistic_2d(X, Y, conc, statistic="mean", bins=bins)
                conc_c = ret.statistic.T
                x = ret.x_edge
                y = ret.y_edge
                # ^ these are cell edges
                xc = x[:-1] + 0.5 * np.diff(x)
                yc = y[:-1] + 0.5 * np.diff(y)
                # ^ these are cell centers

                # seems to be dominated by particle dispersion
                # investigating here:
                # z = (conc_p_rel).squeeze()
                # z = (conc_c).squeeze()
                z = (conc_p_rel * conc_c).squeeze()

                # hack for now
                # label = spc if i == 0 else None
                label = chemical_species_data[spc]["display_name"] if i == 0 else None

                ax_.plot(xc, z, "-", label=label)

                if log_cnorm:
                    ax_.set_yscale("log")

                ax_.set_xlabel("x")
                ax_.set_ylabel("y")
                ax_.set_title(f"y = {y0_source}")

    else:
        raise ValueError("invalid `plot_type`")

    if plot_type in ("scatter", "pcolor", "contourf"):
        cb = fig.colorbar(im, drawedges=False)
        cb.set_label(f"{spc_display_name} relative conc. (%)")

        for (x, y) in p["source_positions"]:
            ax.plot(x, y, "*", c="gold", ms=11, mec="0.35", mew=1.0)

        ax.set_xlabel("x")
        ax.set_ylabel("y")

    elif plot_type == "centerline":
        # pass
        # ax.legend()
        fig.legend(ncol=2, fontsize="small")

    ax.set_title(s_t_info(p), loc="left")

    fig.set_tight_layout(True)
    # fig.tight_layout()


# TODO: x-y (or u-v) hists for different height (z) bins


def ws_hist_all(
    hist, p, *, bounds=False,
):
    """Histograms of particle wind speed components 
    from a single-release run.
    """

    ws = hist["ws"]
    u_all = np.ravel(ws[:, :, 0])
    v_all = np.ravel(ws[:, :, 1])
    w_all = np.ravel(ws[:, :, 2])

    num = check_fig_num("ws-hist-all")
    fig, axs = plt.subplots(3, 1, num=num, sharex=True)

    if not bounds:
        bins = 100
    else:
        bins = np.linspace(bounds[0], bounds[1], 100)

    labels = ["$u$", "$v$", "$w$"]
    for i, (ui, ax) in enumerate(zip([u_all, v_all, w_all], axs.flat)):
        ax.hist(ui, bins)
        ax.text(0.01, 0.98, labels[i], va="top", ha="left", fontsize=13, transform=ax.transAxes)

    if bounds:
        axs[0].set_xlim(bounds)

    axs[0].set_title(s_t_info(p), loc="left")
    Np, Nt = p["Np_tot"], p["N_t"]
    N = Np * Nt
    axs[0].set_title(f"$N_p = {to_sci_not(Np)}$\n$N = {to_sci_not(N)}$", loc="right")

    fig.tight_layout()

    # return


def final_pos_hist(
    state, p, *, bounds=False,
):
    """Histograms of final position components."""

    xf = state["xp"]
    yf = state["yp"]
    zf = state["zp"]

    num = check_fig_num("final-pos-hist")
    fig, axs = plt.subplots(3, 1, num=num, sharex=True)

    if not bounds:
        bins = 100
    else:
        bins = np.linspace(bounds[0], bounds[1], 100)

    labels = ["$x$", "$y$", "$z$"]
    for i, (xi, ax) in enumerate(zip([xf, yf, zf], axs.flat)):
        ax.hist(xi, bins)
        ax.text(0.01, 0.98, labels[i], va="top", ha="left", fontsize=13, transform=ax.transAxes)

    if bounds:
        axs[0].set_xlim(bounds)

    axs[0].set_title(s_t_info(p), loc="left")

    fig.tight_layout()


def final_pos_hist2d(
    state, p, *, dim=("x", "y"), bounds=False, create_contourf=False, log_cnorm=False,
):
    """2-D histogram of selected final position components."""

    x = state[f"{dim[0]}p"]
    y = state[f"{dim[1]}p"]

    Np = x.size

    if len(dim) != 2 or any(dim_ not in ("x", "y", "z") for dim_ in dim):
        raise ValueError
    sdim = "-".join(dim)

    # TODO: match style of final_pos_scatter, like 'xy', not 'x-y'
    num = check_fig_num(f"final-pos-hist-{sdim}")
    fig, ax = plt.subplots(num=num)

    if not bounds:
        bins = 50
    elif bounds == "auto":
        xbar, xstd = x.mean(), x.std()
        ybar, ystd = y.mean(), y.std()
        mult = 2.0
        nx = min(np.sqrt(Np).astype(int), 100)
        ny = nx
        x_edges = np.linspace(xbar - mult * xstd, xbar + mult * xstd, nx + 1)
        y_edges = np.linspace(ybar - mult * ystd, ybar + mult * ystd, ny + 1)
        bins = [x_edges, y_edges]
        # TODO: fix so that for z we don't go below zero (or just a bit)
    else:
        bins = np.linspace(bounds[0], bounds[1], 50)

    # H, xedges, yedges = np.histogram2d(x, y, bins=bins)

    if log_cnorm:
        norm = mpl.colors.LogNorm(vmin=1.0)
    else:
        norm = mpl.colors.Normalize(vmin=1.0)

    H, xedges, yedges, im = ax.hist2d(x, y, bins=bins, norm=norm)
    # ^ returns h (nx, ny), xedges, yedges, image

    cb = plt.colorbar(im)

    ax.set_xlabel(f"${dim[0]}$")
    ax.set_ylabel(f"${dim[1]}$")

    ax.set_xlim((xedges[0], xedges[-1]))
    ax.set_ylim((yedges[0], yedges[-1]))
    ax.set_title(s_t_info(p), loc="left")

    for (x, y) in p["source_positions"]:
        ax.plot(x, y, "*", c="gold", ms=11, mec="0.35", mew=1.0)

    fig.tight_layout()

    if create_contourf:
        num = check_fig_num(f"final-pos-hist-{sdim}-contourf")
        fig2, ax = plt.subplots(num=num)

        levels = np.arange(1, H.max() + 1, 1)  # TODO: should adjust for log cnorm
        xc = xedges[:-1] + np.diff(xedges)
        yc = yedges[:-1] + np.diff(yedges)
        cs = ax.contourf(
            xc,
            yc,
            H.T,
            levels=levels,
            norm=norm,
            # extend='max'
        )

        cb = plt.colorbar(cs)
        cs.cmap.set_under("white")
        cs.changed()

        ax.set_xlabel(f"${dim[0]}$")
        ax.set_ylabel(f"${dim[1]}$")
        ax.set_title(s_t_info(p), loc="left")

        ax.set_xlim((xedges[0], xedges[-1]))
        ax.set_ylim((yedges[0], yedges[-1]))

        for (x, y) in p["source_positions"]:
            ax.plot(x, y, "*", c="gold", ms=11, mec="0.35", mew=1.0)

        fig2.tight_layout()
