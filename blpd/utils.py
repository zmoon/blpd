"""
Miscellaneous utility functions,
mostly for internal use.
"""

import inspect
import sys
from collections import namedtuple
from functools import partial

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

if sys.version_info >= (3, 9):
    from functools import cache
else:
    from functools import lru_cache

    cache = lru_cache(maxsize=None)

# Potentially publically useful
__all__ = (
    "auto_bins",
    "bin_ds",
    "bin_values_xy",
    "moving_average",
    "numbify",
    "unnumbify",
)

_SNIPPETS = {
    "bins_params": """
bins
    If `bins='auto'`, `blpd.utils.auto_bins` is used to construct the bins.
    Alternatively, you can pass bin specifications in any of the forms described
    [here](https://numpy.org/doc/stable/reference/generated/numpy.histogramdd.html).
    """.strip(),
    "sdim_params": """
sdim : str
    String representation of the set of dimensions.
    Valid options: `["xy", "xz", "yz", "xyz", "3d", "3-d"]`.
    """.strip(),
}


def _add_snippets(func):
    """Add docstring snippets from `_SNIPPETS` dict."""
    # ref: https://github.com/lukelbd/proplot/blob/master/proplot/internals/docstring.py
    func.__doc__ = inspect.getdoc(func)
    if func.__doc__:
        func.__doc__ %= {key: value.strip() for key, value in _SNIPPETS.items()}
    return func


def get_open_fig_labels():
    return [plt.figure(num).get_label() for num in plt.get_fignums()]


def check_fig_num(label, n=0):
    """Create a figure `num` (label) that is not already taken,
    appending `_{n}` if a figure with `label` already exists.
    """
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
    """Convert float `f` to a scientific notation string
    (not including the `$`)
    by using string formats `e` and `g`.
    """
    s_e = f"{f:.4e}"
    s_dec, s_pow_ = s_e.split("e")
    s_dec = f"{float(s_dec):.4g}"
    pow_ = int(s_pow_)
    return f"{s_dec} \\times 10^{{ {pow_} }}"


def sec_to_str(total_seconds):
    """Choose best format to display the run time.

    Note: can't use `strftime` with `timedelta`s
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
    """Moving average of `a`.

    Implementation taken from: <https://stackoverflow.com/a/14314054>.
    For now only `axis=0` works.
    """
    if axis != 0:
        raise NotImplementedError
    ret = np.cumsum(a, axis=axis, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1 :] / n


def s_t_info(p):
    """Create string to display info about run time and time step
    using model parameters `p`.
    """
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
    """Create string with sample size info (number of particles `Np` and `N=Np*Nt`)."""
    Np, Nt = p["Np_tot"], p["N_t"]
    N = Np * Nt
    s_N_p = f"$N_p = {to_sci_not(Np)}$"
    s_N = f"$N = {to_sci_not(N)}$"
    if N_p_only:
        return s_N_p
    else:
        return "\n".join([s_N_p, s_N])


# TODO: fn to pre-process state for plots, removing data outside certain limits or with too high vel components ?


def _auto_bins_1d(x, *, nx_max, std_mult, method, pos_only=False):
    x_bar, x_std = x.mean(), x.std()
    if std_mult is not None:
        a = 0 if pos_only else x_bar - std_mult * x_std
        b = x_bar + std_mult * x_std
        x_range = a, b
    else:
        x_range = None

    x_edges = np.histogram_bin_edges(x, bins=method, range=x_range)

    if x_edges.size > nx_max + 1:
        x_edges = np.linspace(*x_range, nx_max + 1)

    return x_edges


@_add_snippets
def auto_bins(
    positions,
    sdim="xy",
    *,
    nbins_max_1d: int = 100,
    std_mult: float = 2.0,
    method: str = "auto",
):
    """Determine bin edges for a 2- or 3-d horizontal grid
    that lets us focus on where most of the data is.

    Parameters
    ----------
    positions
        Container of 1-D arrays (in *x*,*y*[,*z*] order)
        OR single NumPy array where columns are *x*,*y*[,*z*].
    %(sdim_params)s
    nbins_max_1d
        Maximum number of bins in any direction/dim.
    std_mult
        Standard deviation multiplier.
        Increase to capture more of the domain.
        Or set to `None` to capture all of it.
    method
        Passed to :func:`numpy.histogram_bin_edges`.
        Usually use `'auto'` or `'sqrt'`.
    """
    pos = np.asarray(positions)
    assert pos.ndim == 2
    if pos.shape[0] in [2, 3]:
        # ^ need to flip (or we only have 2 or 3 data points, but that is unlikely!)
        pos = pos.T

    dims = dims_from_sdim(sdim)
    idims = ["xyz".index(dim1) for dim1 in dims]

    # TODO: optional `z_range=` arg for including certain range of heights, with defaults np.inf or None
    # TODO: optional centering cell over certain x, y value (like 0, 0)

    kwargs_1d = dict(nx_max=nbins_max_1d, std_mult=std_mult, method=method)
    bins = [_auto_bins_1d(pos[:, idim1], pos_only=idim1 == 2, **kwargs_1d) for idim1 in idims]

    return bins


auto_bins_xy = partial(auto_bins, sdim="xy")
auto_bins_xy.__doc__ = ":func:`utils.auto_bins` with ``sdim='xy'``"


_Binned_values_xy = namedtuple("Binned_values_xy", "x y xe ye v")


@_add_snippets
def bin_values_xy(x, y, values, *, bins="auto", stat="median"):
    """Using the `x` and `y` positions of the particles, bin `values`,
    and calculate a representative value in each bin.

    Parameters
    ----------
    %(bins_params)s
    stat : str
        Statistic to calculate on the binned values.
    """
    from scipy import stats

    if bins == "auto":
        bins = auto_bins_xy((x, y))

    # 1. Concentration of LPD particles
    H, xe0, ye0 = np.histogram2d(x, y, bins=bins)  # H is binned particle count
    conc_p_rel = (H / H.max()).T  # TODO: really should divide by level at source (closest bin?)

    # 2. In-particle values in each bin
    ret = stats.binned_statistic_2d(x, y, values, statistic=stat, bins=bins)
    v0 = ret.statistic.T  # it is returned with dim (nx, ny), we need y to be rows (dim 0)
    xe = ret.x_edge
    ye = ret.y_edge
    # ^ these are cell edges
    xc = xe[:-1] + 0.5 * np.diff(xe)
    yc = ye[:-1] + 0.5 * np.diff(ye)
    # ^ these are cell centers
    assert np.allclose(xe, xe0) and np.allclose(ye, ye0)  # confirm bins same

    # 3. Multiply in-particle stat by particle conc. to get final values
    v = conc_p_rel * v0

    return _Binned_values_xy(xc, yc, xe, ye, v)


@_add_snippets
def bin_ds(ds, sdim="xy", *, variables="all", bins="auto"):
    """Bin a particles dataset. For example,
    the `blpd.model.Model.to_xarray` dataset
    or `blpd.chem.calc_relative_levels_fixed_oxidants` dataset.

    Parameters
    ----------
    %(sdim_params)s
    variables : str, list(str)
        `'all'` to bin all variables, or pass a list of variable names.
    %(bins_params)s
    """
    import xarray as xr
    from scipy import stats

    if not tuple(ds.dims) == ("ip",):
        raise NotImplementedError("Must have only particle dim for now.")

    if variables == "all":
        vns = [vn for vn in ds.variables if vn not in list("xyz") + list(ds.dims) + list(ds.coords)]
    else:
        vns = variables

    # Deal with dims
    dims = dims_from_sdim(sdim)
    dims_e = tuple(f"{dim1}e" for dim1 in dims)  # bin edges
    dims_c = tuple(f"{dim1}" for dim1 in dims)  # bin centers
    pos0 = tuple(ds[dim1].values for dim1 in "xyz")  # must pass to auto_grid in xyz order
    pos = tuple(ds[dim1].values for dim1 in dims)  # for the stats
    if bins == "auto":
        bins = auto_bins(pos0, sdim)

    # Compute statistics, using `binned_statistic_dd` so we can pass the previous
    # result for efficiency.
    res = {}  # {vn: {stat: ...}, ...}
    ret = None  # for first run we don't have a result yet
    stats_to_calc = ["mean", "median", "std", "count"]
    # ^ note that sum can be recovered with mean and particle count
    for vn in vns:
        values = ds[vn].values
        rets = {}
        for stat in stats_to_calc:
            ret = stats.binned_statistic_dd(
                pos,
                values,
                statistic=stat,
                bins=bins,
                binned_statistic_result=ret,
            )
            rets[stat] = ret
        res[vn] = rets

        if vn == vns[0]:
            stats_to_calc.remove("count")  # only need it once

    # Construct coordinates dict
    coords = {}
    for dim1, bins, dim_e, dim_c in zip(dims, ret.bin_edges, dims_e, dims_c):
        xe = bins
        xc = xe[:-1] + 0.5 * np.diff(xe)
        coords[dim_c] = (dim_c, xc, {"units": "m", "long_name": f"{dim1} (bin center)"})
        coords[dim_e] = (dim_e, xe, {"units": "m", "long_name": f"{dim1} (bin edge)"})

    # Create dataset of the binned statistics on the variables
    ds = xr.Dataset(
        coords=coords,
        data_vars={
            f"{vn}_{stat}": (
                dims_c,
                ret.statistic,
                {
                    "units": ds[vn].attrs.get("units", ""),
                    "long_name": ds[vn].attrs.get("long_name", ""),
                },
            )
            for vn, d in res.items()
            for stat, ret in d.items()
        },
    )

    # Add particle count
    ds["Np"] = (dims_c, res[vns[0]]["count"].statistic, {"long_name": "Lagrangian particle count"})

    return ds.transpose()  # y first so 2-d xarray plots just work


bin_ds_xy = partial(bin_ds, sdim="xy")
bin_ds_xy.__doc__ = ":func:`utils.bin_ds` with ``sdim='xy'``"


def calc_t_out(p):
    """Calculate time-since-release for each particle at the end of simulation.

    This works for a simulation with constant particle release rate
    (number of particles released per time step per source).

    Parameters
    ----------
    p : dict
        The model parameters dict.

    Returns
    -------
    np.array
        time-since-release in seconds
    """
    # unpack needed model options/params
    Np_tot = p["Np_tot"]
    dt = p["dt"]
    N_t = p["N_t"]  # number of time steps
    t_tot = p["t_tot"]
    dNp_dt_ds = p["dNp_per_dt_per_source"]
    N_s = p["N_sources"]

    # Calculate time-since-release for every particle
    # ! the method here is based on time as outer loop
    # ! and will be incorrect if that changes
    # t_out = np.r_[[[(k+1)*numpart for p in range(numpart)] for k in range(N)]].flat
    t_out = np.ravel(np.tile(np.arange(dt, N_t * dt + dt, dt)[:, np.newaxis], dNp_dt_ds * N_s))
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


def maybe_new_figure(try_num: str, ax=None, subplot_kw=None):
    if subplot_kw is None:
        subplot_kw = {}

    if ax is None:
        num = check_fig_num(try_num)
        fig, ax = plt.subplots(num=num, subplot_kw=subplot_kw)
    else:
        fig = ax.get_figure()

    return fig, ax


def check_sdim(sdim: str):
    """Validate `sdim`."""
    valid_sdim = ["xy", "xz", "yz", "xyz", "3d", "3-d"]
    if sdim not in valid_sdim:
        raise ValueError(
            "for `sdim`, pick 2 or 3 from 'x', 'y', and 'z'. For example, 'xy'. "
            f"The full set of valid options is {valid_sdim}. "
            "The last two are aliases for 'xyz'."
        )


def dims_from_sdim(sdim: str):
    """Return tuple of dims corresponding to `sdim`."""
    check_sdim(sdim)  # first validate
    if sdim in ("xyz", "3d", "3-d"):
        dims = list("xyz")
    else:
        dims = list(sdim)

    return dims


# TODO: float precision option
def numbify(d, zerod_only=False):
    """Convert dict `d` to Numba-suitable format.

    Dict values should be NumPy arrays or individual floats.

    References
    ----------
    * <https://numba.pydata.org/numba-doc/dev/reference/pysupported.html#typed-dict>
    """
    import numba

    float64_array = numba.types.float64[:]  # maybe would be more efficient to pass size
    float64 = numba.types.float64
    d_nb = numba.typed.Dict.empty(
        key_type=numba.types.unicode_type,
        value_type=float64_array if not zerod_only else float64,
    )
    for k, v in d.items():
        # print(k, v, np.asarray(v).shape, np.asarray(v).shape == ())

        # d_nb[k] = np.asarray(v, dtype=np.float64)  # will convert bool to 0/1
        # d_nb[k] = np.asarray(v).astype(np.float64)

        # np.asarray leaves float/int with size=1 but shape=() so numba doesn't work
        # so hacking this for now
        if not zerod_only:
            if np.asarray(v).shape == ():
                x = np.float64((1,))
                x[:] = np.float64(v)
                d_nb[k] = x
            else:
                # d_nb[k] = np.asarray(v, dtype=np.float64)
                d_nb[k] = np.asarray(v).astype(np.float64)
        else:
            d_nb[k] = np.float64(v)

    # numba.typed.Dict.empty() creates a normal dict if it thinks jit is disabled
    # ref: https://github.com/numba/numba/blob/868b8e3e8d034dac0440b75ca31595e07f632d27/numba/typed/typeddict.py#L95
    assert isinstance(d_nb, numba.typed.Dict)

    return d_nb


def unnumbify(d_nb):
    """Convert numba dict `d_nb` to a normal dict of numpy arrays, etc."""
    import numba

    if not isinstance(d_nb, numba.typed.Dict):
        raise TypeError("this fn is for dicts of type `numba.typed.Dict`")

    d = {}
    for k, v in d_nb.items():
        # if isinstance(v, numba.types.float64[:]):
        if v.shape == (1,):
            d[k] = v[0]
        else:
            d[k] = v

    return d


def disable_numba():
    """Tell numba to disable JIT.

    References
    ----------
    * disabling JIT: https://numba.pydata.org/numba-doc/latest/user/troubleshoot.html#disabling-jit-compilation
    * reloading config: https://github.com/numba/numba/blob/868b8e3e8d034dac0440b75ca31595e07f632d27/numba/core/config.py#L369
    """
    import os

    import numba

    os.environ.update({"NUMBA_DISABLE_JIT": str(1)})
    numba.config.reload_config()
    assert numba.config.DISABLE_JIT == 1  # pylint: disable=no-member


def enable_numba():
    """Tell numba to enable JIT.
    See references in `disable_numba`.
    """
    import os

    import numba

    # By default (when numba is loaded normally), this env var is not set, so remove it
    os.environ.pop("NUMBA_DISABLE_JIT", default=None)  # avoid error
    numba.config.reload_config()
    assert numba.config.DISABLE_JIT == 0  # pylint: disable=no-member
