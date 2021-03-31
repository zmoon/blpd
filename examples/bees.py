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

import blpd

# %%
x, y = blpd.bees.flight(200, mu=2)

plt.plot(x, y, ".-")

# %%
x1, y1 = blpd.bees.flight(200, mu=2, l_max=40)

plt.plot(x1, y1, ".-")

# %%
# Modeling preference for continuing in same direction

import math
from scipy.stats import truncnorm

std = 1.5
mean = 0
clip_a, clip_b = -math.pi, math.pi

a, b = (clip_a - mean) / std, (clip_b - mean) / std

dist = truncnorm(a, b)

x = dist.rvs(10_000)

plt.hist(x, 40)
