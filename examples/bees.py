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
