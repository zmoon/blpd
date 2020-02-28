"""

plots of lpdm results

"""

__all__ = ('pos_scatter', 'conc', 'trajectories')

from matplotlib.collections import LineCollection as _LineCollection
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
# could add `as _{}` to all of these
# to indicate that these are not intended to be public parts of the module name space
# since __all__ is not respected by linters or autocompleters

plt.close('all')



def get_open_fig_labels():
    return [plt.figure(num).get_label() for num in plt.get_fignums()]
# TODO: use this to check if a figure with certain num already exists, then append _X to it


def check_fig_num(label, n=0):
    current_labels = get_open_fig_labels()
    if n == 0:
        labeln = label
    else:
        labeln = f'{label}_{n}'
    if labeln in current_labels:
        return check_fig_num(label, n=n+1)
    else:
        return labeln


def moving_average(a, n=3, axis=0):
    """
    from: https://stackoverflow.com/a/14314054
    for now only axis=0 works
    """
    if axis != 0:
        raise NotImplementedError
    ret = np.cumsum(a, axis=axis, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


def pos_scatter(state, p):
    """
    """
    xpath = state['xp']
    ypath = state['yp']
    zpath = state['zp']
    T = state['t']

    t_tot = p['t_tot']
    dt = p['dt']
    N_t = p['N_t']
    Np_tot = p['Np_tot']

    numpart_tot = xpath.size
    assert( numpart_tot == Np_tot )

    fig, ax = plt.subplots(num='horizontal-end-positions')
    ax.plot(xpath, ypath, 'o', alpha=0.5, mew=0)

    ax.set_xlabel('x'); ax.set_ylabel('y')

    ax.set_title(f'$t = {T}$ s, $\\delta t = {dt}$, $N_t = {N_t}$', loc='left')
    ax.set_title(f'$N_p = {numpart_tot}$', loc='right')

    # should make fn for this
    for (x, y) in p['source_positions']:
        ax.plot(x, y, '*', c='gold', ms=10)

    fig.tight_layout()

    # fig, ax = plt.subplots(num='horizontal-end-positions-hist')
    # h, _, _, im = ax.hist2d(xpath, ypath)
    # ax.set_xlabel('x'); ax.set_ylabel('y')
    # fig.colorbar(im)
    # ax.set_title(f'N = {numpart_tot}, t = {T} s')


def to_sci_not(f):
    """Convert float f to scientific notation string
    by using string formats 'e' and 'g'
    
    The output of this must be enclosed within `$`
    """
    s_e = f'{f:.4e}'
    s_dec, s_pow_ = s_e.split('e')
    s_dec = f'{float(s_dec):.4g}'
    pow_ = int(s_pow_)
    return f'{s_dec} \\times 10^{{ {pow_} }}'


def trajectories(hist, p, 
    smooth=False, smooth_window_size=100, 
    ):

    fig, ax = plt.subplots(num='trajectories')

    # segs = hist['pos']
    pos = hist['pos']

    t_tot = p['t_tot']
    dt = p['dt']
    N_t = p['N_t']
    Np_tot = p['Np_tot']

    ltitle = f'$t = {t_tot}$ s, $\\delta t = {dt}$, $N_t = {N_t}$'

    s_Np_tot = to_sci_not(Np_tot)
    rtitle = f'$N_p = {s_Np_tot}$'

    if smooth:
        n = int(smooth_window_size)
        if n*dt > 0.5*t_tot:
            raise ValueError("can't do any smoothing with the requested window size (not enough points)")
        # print(pos.shape)
        pos0 = pos[:,0,:][:,np.newaxis,:]
        pos = np.swapaxes(moving_average(\
            np.swapaxes(pos, 0, 1), n=n, axis=0), 0, 1)
        # ^ np.swapaxes should return views, not create new arrays (ver >= 1.10.0)
        # print(pos0.shape, pos.shape)

        # preserve starting point
        pos = np.concatenate((pos0, pos), axis=1)  # axis=1 -> like hstack

        ltitle = f'$N_{{smooth}} = {n}$ ({n*dt:} s)\n{ltitle}'


    # segs = (pos[i,:,:2] for i in range(pos.shape[0]))

    segs = [pos[i,:,:2] for i in range(pos.shape[0])]


    lc = _LineCollection(segs,
        linewidths=0.5, colors='0.6', linestyles='solid', alpha=0.5, 
    )
    ax.add_collection(lc)

    for (x, y) in p['source_positions']:
        ax.plot(x, y, '*', c='gold', ms=10)

    ax.autoscale()

    ax.set_xlabel('x'); ax.set_ylabel('y')

    ax.set_title(ltitle, loc='left')
    ax.set_title(rtitle, loc='right')

    fig.tight_layout()


# marker size should be calculated dynamically
# but also allowed to pass!


def conc(state, conc, p, *, 
    plot_type='scatter', bins=(20, 10), levels=30, 
    cmap='gnuplot',
    ):
    xpath = state['xp']
    ypath = state['yp']
    zpath = state['zp']

    X = xpath
    Y = ypath
    Z = zpath

    compound_name = 'beta-ocimene'

    fig, ax = plt.subplots(num=f'horizontal-end-positions-with-conc_{compound_name}_{plot_type}')
    # plt.scatter(xpath, ypath, c=conc, marker='o', alpha=0.5, linewidths=0)
    # plt.colorbar()
    # default `s` is 6**2 (default lines.markersize squared)

    if plot_type == 'scatter':
        im = ax.scatter(X, Y, c=conc, s=7, marker='o', alpha=0.4, linewidths=0, cmap=cmap, vmax=100)
    elif plot_type in ('pcolor', 'contourf'):
        ret = stats.binned_statistic_2d(X, Y, conc, statistic='mean', bins=bins)
        z = ret.statistic.T  # it is returned with dim (nx, ny), we need y to be rows (dim 0)
        x = ret.x_edge
        y = ret.y_edge
        # ^ these are cell edges
        xc = x[:-1] + 0.5*np.diff(x)
        yc = y[:-1] + 0.5*np.diff(y)
        if plot_type == 'pcolor':
            im = ax.pcolormesh(x, y, z, cmap=cmap, vmax=100)
        elif plot_type == 'contourf':
            im = ax.contourf(xc, yc, z, levels, cmap=cmap, vmax=100)

    else:
        raise ValueError('invalid `plot_type`')
    
    cb = fig.colorbar(im, drawedges=False)


    cb.set_label(f'{compound_name} relative conc. (%)')
    ax.set_xlabel('x'); ax.set_ylabel('y')
    # ax.set_title(f'N = {numpart_tot}, t = {T} s')

    # ax.set_xlim((-10, 50))

    for (x, y) in p['source_positions']:
        ax.plot(x, y, '*', c='gold', ms=11, mec='0.35', mew=1.0)

    fig.tight_layout()



# TODO: statistics (hist etc.) of final positions and velocity time series (for traj)


def ws_hist_all(hist, p,
    bounds=False,
):

    ws = hist['ws']
    u_all = np.ravel(ws[:,:,0])
    v_all = np.ravel(ws[:,:,1])
    w_all = np.ravel(ws[:,:,2])

    num = check_fig_num('ws-hist-all')
    fig, axs = plt.subplots(3, 1, num=num, sharex=True)

    if not bounds:
        bins = 100
    else:
        bins = np.linspace(bounds[0], bounds[1], 100)

    labels = ['$u$', '$v$', '$w$']
    for i, (ui, ax) in enumerate(zip([u_all, v_all, w_all], axs.flat)):
        ax.hist(ui, bins)
        ax.text(0.01, 0.98, labels[i], va='top', ha='left', fontsize=13, transform=ax.transAxes)

    if bounds:
        axs[0].set_xlim(bounds)

    fig.tight_layout()

    return


