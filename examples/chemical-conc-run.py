# -*- coding: utf-8 -*-
"""
Demonstrate chemical concentration plots.

@author: zmoon
"""
import sys

sys.path.append("../")

import blpdm
from blpdm import plots


# %% Create case and run

m = blpdm.model()

# change some params
new_p = {
    "source_positions": [(-200, -200), (500, 0), (1000, 200)],
    # ^ similar to the ones Pratt used in their Fig 4.2
    "t_tot": 1 * 60.0,  # 1 min; change to 10 min for more excitement
    "continuous_release": True,  # this is a default
    "dNp_per_dt_per_source": 4,
    # 'chemistry_on': True,
}

m.update_p(new_p)

m.run()


# %% Plots
# demonstrate some of the settings

ds0 = m.to_xarray()
ds = blpdm.chem.calc_relative_levels_fixed_oxidants(ds0)

# apinene is the default species plotted if spc not specified

plots.conc_2d(ds)

plots.conc_2d(ds, plot_type="pcolor")
plots.conc_2d(ds, plot_type="pcolor", bins=(100, 50))
plots.conc_2d(ds, plot_type="pcolor", bins="auto")
plots.conc_2d(ds, plot_type="pcolor", bins="auto", log_cnorm=True)

plots.conc_2d(ds, plot_type="contourf")
plots.conc_2d(ds, plot_type="contourf", bins=(100, 50))  # number of bins in each dim
plots.conc_2d(ds, plot_type="contourf", bins="auto")
plots.conc_2d(ds, plot_type="contourf", bins="auto", log_cnorm=True)

# we can compare the above reactive species conc. plots to one of the non-reactice particles
plots.final_pos_hist2d(ds, bins="auto", log_cnorm=True)


# %% Plot the other species

for spc in ds.spc.values:

    print(ds.f_r.sel(spc=spc).mean())

    plots.conc_2d(ds, spc, plot_type="pcolor", bins="auto", log_cnorm=True, vmin=1.0)


# %% Centerline conc

plots.conc_xline(ds, spc="all")
