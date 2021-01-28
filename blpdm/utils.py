"""
Miscellaneous utility functions for the model, plots, etc.
"""
from collections import namedtuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

# TODO: create fn to compare two sets of parameters and point out the changes


def get_open_fig_labels():
    return [plt.figure(num).get_label() for num in plt.get_fignums()]


def check_fig_num(label, n=0):
    """Create a fig num that is not already taken."""
    current_labels = get_open_fig_labels()
    if n == 0:
        labeln = label
    else:
        labeln = f"{label}_{n}"
    if labeln in current_labels:
        return check_fig_num(label, n=n + 1)
    else:
        return labeln


def to_sci_not(f):
    """Convert float f to scientific notation string.
    by using string formats 'e' and 'g'

    The output of this must be enclosed within `$`
    """
    s_e = f"{f:.4e}"
    s_dec, s_pow_ = s_e.split("e")
    s_dec = f"{float(s_dec):.4g}"
    pow_ = int(s_pow_)
    return f"{s_dec} \\times 10^{{ {pow_} }}"


def sec_to_str(total_seconds):
    """Choose best format to display the run time.

    note: can't use `strftime` with `timedelta`s
    """
    hours, remainder = divmod(total_seconds, 60 * 60)
    minutes, seconds = divmod(remainder, 60)

    hours = int(hours)
    minutes = int(minutes)
    seconds = int(seconds)  # most likely wouldn't run for seconds fractions, only multiples

    h = "\u2095"  # unicode subscript letters
    m = "\u2098"
    s = "\u209B"
    sep = "\u2009"  # thin space

    if total_seconds <= 60:
        s = f"{seconds}{s}"
    elif total_seconds <= 60 * 60:
        s = f"{minutes}{m}{sep}{seconds}{s}"
    else:
        s = f"{hours}{h}{sep}{minutes}{m}{sep}{seconds}{s}"
    return s
    # TODO: probably a better way to do this, checking for zeros and not displaying those
    #       or programmatically (iteratively or recursively) build the necessary fmt string


def moving_average(a, n=3, axis=0):
    """
    from: https://stackoverflow.com/a/14314054
    for now only axis=0 works
    """
    if axis != 0:
        raise NotImplementedError
    ret = np.cumsum(a, axis=axis, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1 :] / n


def s_t_info(p):
    """Create string to display info about run time and time step."""
    t_tot = p["t_tot"]
    dt = p["dt"]
    N_t = p["N_t"]
    if N_t >= 1000:
        s_N_t = to_sci_not(N_t)
    else:
        s_N_t = str(N_t)
    s_t_tot = sec_to_str(t_tot)
    return f"$t = ${s_t_tot}, $\\delta t = {dt}$ s, $N_t = {s_N_t}$"


def s_sample_size(p, *, N_p_only=False):
    """Create string with sample size info (number of particles Np and N=Np*Nt)."""
    Np, Nt = p["Np_tot"], p["N_t"]
    N = Np * Nt
    s_N_p = f"$N_p = {to_sci_not(Np)}$"
    s_N = f"$N = {to_sci_not(N)}$"
    if N_p_only:
        return s_N_p
    else:
        return "\n".join([s_N_p, s_N])

# TODO: fn to pre-process state for plots, removing data outside certain limits or with too high vel components ?


# TODO: use this fn (auto_grid) in the plotting routines

def auto_grid(positions, *,
    nx_max: int = 100,
    sd_mult: float = 2.0,
):
    """Determine a good grid that lets us focus on where most of the data is.

    positions
        can be:
        * container of 1-D arrays (in x,y[,z] order)
        * single np.array where columns are x,y[,z]

    sd_mult :
        standard deviation multiplier
        increase to capture more of the domain

    """
    # explicity pass in x,y,z positions instead of whole state?
    if isinstance(positions, (tuple, list)):
        X = positions[0]
        Y = positions[1]
    elif isinstance(positions, np.array):
        X = positions[:,0]
        Y = positions[:,1]
    else:
        raise TypeError("`positions`")
    # TODO: implement optional z binning

    # form linearly spaced bins
    # with grid edges based on mean and standard deviation
    Np = X.size  # number of particles in the snapshot
    xbar, xstd = X.mean(), X.std()
    ybar, ystd = Y.mean(), Y.std()
    mult = sd_mult
    nx = min(np.sqrt(Np).astype(int), nx_max)
    ny = nx
    x_edges = np.linspace(xbar - mult * xstd, xbar + mult * xstd, nx + 1)
    y_edges = np.linspace(ybar - mult * ystd, ybar + mult * ystd, ny + 1)
    bins = [x_edges, y_edges]

    return bins



_Bin_c_xy_ret = namedtuple("bin_c_xy", "x y xe ye c")

def bin_c_xy(X, Y, C, *, bins, stat="median"):
    r"""Using the `x`\, `y` positions of the particles, bin `c`, and calculate a representative value in each bin."""
    from scipy import stats

    if bins == "auto":
        bins = auto_grid((X, Y))

    # 1. Concentration of LPD particles
    H, xedges, yedges = np.histogram2d(X, Y, bins=bins)  # H is binned particle count
    conc_p_rel = (H / H.max()).T  # TODO: really should divide by level at source (closest bin?)

    # 2. chemistry
    ret = stats.binned_statistic_2d(X, Y, C, statistic="median", bins=bins)
    conc_c = ret.statistic.T  # it is returned with dim (nx, ny), we need y to be rows (dim 0)
    x = ret.x_edge
    y = ret.y_edge
    # ^ these are cell edges
    xc = x[:-1] + 0.5 * np.diff(x)
    yc = y[:-1] + 0.5 * np.diff(y)
    # ^ these are cell centers

    assert np.allclose(x, xedges) and np.allclose(y, yedges)
    # TODO: find a way to not hist by x,y more than once (here we have done it 2x)

    z = conc_p_rel * conc_c

    return _Bin_c_xy_ret(xc, yc, x, y, z)


def calc_t_out(p):
    """Calculate time-since-release for each particle at the end of simulation.

    This works for a simulation with constant particle release rate
    (number of particles released per time step per source).

    Args
    ----
    p : dict
        the model params+options dict

    Returns
    -------
    np.array
        time-since-release in seconds
    """
    # unpack needed model options/params
    Np_tot = p['Np_tot']
    dt = p['dt']
    N_t = p['N_t']  # number of time steps
    t_tot = p['t_tot']
    dNp_dt_ds = p['dNp_per_dt_per_source']
    N_s = p['N_sources']

    # Calculate time-since-release for every particle
    #! the method here is based on time as outer loop
    #! and will be incorrect if that changes
    # t_out = np.r_[[[(k+1)*numpart for p in range(numpart)] for k in range(N)]].flat
    t_out = np.ravel(np.tile(np.arange(dt, N_t*dt + dt, dt)[:,np.newaxis], dNp_dt_ds*N_s))
    # ^ need to find the best way to do this!
    # note: apparently `(N_t+1)*dt` does not give the same stop as `N_t*dt+dt` sometimes (precision thing?)

    # t_out = (t_out[::-1]+1) * dt
    t_out = t_out[::-1]

    # sanity checks
    assert np.isclose(t_tot, t_out[0])  # the first particle has been out for the full time
    assert t_out.size == Np_tot

    return t_out


def load_p(ds):
    """Load the model parameters/info `dict` from JSON stored in `ds` :class:`xarray.Dataset`."""
    import json
    import xarray as xr

    return json.loads(ds.attrs["p_json"])


def maybe_log_cnorm(log_cnorm=True, levels=30, vmin=None, vmax=100):
    if log_cnorm:
        norm = mpl.colors.LogNorm(vmin=vmin, vmax=vmax)
        if levels is not None:
            # https://matplotlib.org/3.1.3/gallery/images_contours_and_fields/contourf_log.html
            # https://matplotlib.org/3.1.3/api/ticker_api.html#matplotlib.ticker.LogLocator
            # locator = mpl.ticker.LogLocator(subs=(0.25, 0.5, 1.0))  # another way to get more levels in between powers of 10
            nlevels = levels if isinstance(levels, int) else np.asarray(levels).size
            locator = mpl.ticker.LogLocator(subs="all", numticks=nlevels)
            # TODO: although this^ works, the ticks are not all getting labeled. need to fix.
        else:
            locator = None
    else:
        norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
        locator = None

    return norm, locator  # TODO: named tuple would be nicer


def maybe_new_figure(try_num: str, ax=None):
    if ax is None:
        num = check_fig_num(try_num)
        fig, ax = plt.subplots(num=num)
    else:
        fig = ax.get_figure()

    return fig, ax
