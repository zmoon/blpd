# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
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
# # Bees
# %%
import matplotlib.pyplot as plt
import numpy as np

import blpd

# %% [markdown]
# ## Flights
#
# The Fuentes et al. (2016) paper used [Lévy](https://en.wikipedia.org/wiki/L%C3%A9vy_flight)-like power law step size distribution with $\mu=2$, $l_0=1$ [m]. That is the default for `bees.flight`. The flights start at $(0, 0)$ with a northward heading by default. This occasionally produces steps that are quite large.

# %%
N = 10
n = 150
seed = 12345

# %%
fig, ax = plt.subplots()

np.random.seed(seed)

for _ in range(N):
    ax.plot(*blpd.bees.flight(n), ".-")

# %% [markdown]
# We can make long steps less likely by increasing $\mu$, or we can set a maximum step (clip), or both.

# %%
fig, ax = plt.subplots()

np.random.seed(seed)

for _ in range(N):
    ax.plot(*blpd.bees.flight(n, mu=2.5, l_max=50), ".-")

# %% [markdown]
# The default relative heading model is to sample angles from a uniform distribution. We can model a preference for continuing in the same direction by using the `heading_model="truncnorm"` option.

# %%
fig, ax = plt.subplots()

np.random.seed(seed)

for _ in range(N):
    ax.plot(*blpd.bees.flight(n, l_max=50, heading_model="truncnorm"), ".-")

# %% [markdown]
# We can adjust the preference. The default `std` is 1.5. Decreasing it, there is greater preference for continuing in the same direction.

# %%
fig, ax = plt.subplots()

np.random.seed(seed)

for _ in range(N):
    ax.plot(
        *blpd.bees.flight(
            n, mu=2, l_max=50, heading_model="truncnorm", heading_model_kwargs=dict(std=0.5)
        ),
        ".-"
    )

# %%
fig, ax = plt.subplots()

np.random.seed(seed)

for _ in range(N):
    ax.plot(
        *blpd.bees.flight(
            n, mu=2, l_max=50, heading_model="truncnorm", heading_model_kwargs=dict(std=3)
        ),
        ".-"
    )

# %% [markdown]
# ## Floral scents
#
# We can model the levels of floral volatiles that the bees encounter on their flights.

# %%
# TODO
