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
# # Demonstrate run types and default plots
# %%
import blpd


# %% [markdown]
# The default case has a single source at $(0, 0)$.
#
# The default run type is continuous-release. In this run, we only keep track of the current positions of the particles, not their trajectory history.

# %%
m0 = blpd.model()  # default settings
m0.run()
m0.plot()


# %% [markdown]
# The above series can also be done all in one line by chaining the calls. Note that even though we are using the same model inputs, the result is different since there is a stochastic element to the trajectory calculations.

# %%
blpd.model().run().plot()

# %% [markdown]
# The other type of run is single-release (`continuous_release=False`). With this run type, we store all particle trajectories and can plot them.

# %%
m1 = blpd.model(
    pu={
        "continuous_release": False,
        "dNp_per_dt_per_source": 1000,  # for single-release, the total number of particles to release
        "t_tot": 5 * 60,  # 5 minutes
        "dt": 0.1,
        "dt_out": 1.0,  # a multiple of model integration time step
    }
)
m1.run()
m1.plot()


# %% [markdown]
# The default plot can be modified by passing kwargs through to the relevant plotting function. Here we smooth the trajectories.

# %%
m1.plot(smooth_window_size=60)  # a 1 min window
