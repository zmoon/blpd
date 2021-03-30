# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: title,-all
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---
# %% [markdown]
# # Chemical conc. runs
#
# We use `blpd` to model canopy emissions (e.g., floral volatiles). Lagrangian particles have in-particle concentrations of chemical species, which can be spatially aggregated to form concentration maps (or at least relative to release).
# %%
import matplotlib.pyplot as plt

import blpd
from blpd import plots as plot


# %% [markdown]
# ## Run LPD
#
# For chemical concentrations, we use the (default) continuous-release run type.

# %%
# Start from default case
m = blpd.model()

# Change some params
new_p = {
    "source_positions": [(-200, -200), (500, 0), (1000, 200)],
    # ^ similar to the ones Pratt used in their Fig 4.2
    "t_tot": 1 * 60.0,  # 1 min; change to 10 min to make the plots more exciting
    "continuous_release": True,  # this is a default
    "dNp_per_dt_per_source": 4,
}
m.update_p(new_p)

# Run
m.run()


# %% [markdown]
# ## Calculate chemistry
#
# Chemistry is calculated offline, after the LPD model integration. Chemistry routines take the model output dataset (`xr.Dataset`) as input.

# %%
ds0 = m.to_xarray()
ds0

# %%
ds = blpd.chem.calc_relative_levels_fixed_oxidants(ds0)
ds

# %% [markdown]
# ## Plots
#
# `'apinene'` ([α-pinene](https://en.wikipedia.org/wiki/Alpha-Pinene)) is the default species plotted if `spc` is not specified.
#
# In the `conc_2d` plot, particles are binned in two dimensions only ($x$ and $y$ by default), so the result is akin to something like total column ozone.

# %%
plot.conc_2d(ds)

# %% [markdown]
# Below we demonstrate the impact of some `conc_2d` settings. The choice of `bins` is important.

# %%
kwargss = [
    dict(),  # default
    dict(bins=(100, 50)),
    dict(bins="auto"),
    dict(bins="auto", log_cnorm=True),
    dict(plot_type="contourf"),
    dict(plot_type="contourf", bins=(100, 50)),
    dict(plot_type="contourf", bins="auto"),
    dict(plot_type="contourf", bins="auto", log_cnorm=True),
]

fig, axs = plt.subplots(4, 2, sharex=True, sharey=True, figsize=(9, 12))

for ax, kwargs in zip(axs.flat, kwargss):
    plot.conc_2d(ds, **kwargs, ax=ax)


# %% [markdown]
# We can compare the above reactive species conc. plots to one of the non-reactice particles.

# %%
plot.final_pos_hist2d(ds, bins="auto", log_cnorm=True)

# %% [markdown]
# We can plot other chemical species if we want.

# %%
fig, axs = plt.subplots(3, 2, sharex=True, sharey=True, figsize=(9, 9))

for spc, ax in zip(ds.spc.values[:6], axs.flat):
    plot.conc_2d(ds, spc, plot_type="pcolor", bins="auto", log_cnorm=True, vmin=1.0, ax=ax)
    ax.text(
        0.02,
        0.98,
        f"mean: {ds.f_r.sel(spc=spc).mean().values:.4g}",
        ha="left",
        va="top",
        transform=ax.transAxes,
    )


# %% [markdown]
# And "line" concentrations.

# %%
plot.conc_xline(ds, spc="all")
