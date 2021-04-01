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
# # Parameters
#
# How to set them, see them, update them...
# %%
import pprint

import blpd
from blpd.model import compare_params, INPUT_PARAM_DEFAULTS  # noreorder


# %% [markdown]
# First we print the *input* parameter defaults.

# %%
pp = pprint.PrettyPrinter()

msg = "All model input parameters"
print(msg)
print("-" * len(msg))
pp.pprint(INPUT_PARAM_DEFAULTS)

# %% [markdown]
# Creating a default model object, we can see that its parameters match the defaults.

# %%
m0 = blpd.model()

compare_params(m0.p)

# %% [markdown]
# It has other parameters besides the ones shown above, since additional parameters are derived from the *input* parameters.

# %%
set(m0.p) - set(INPUT_PARAM_DEFAULTS)

# %% [markdown]
# Creating a new model instance and changing some settings, we can see how the *input* and *derived* parameters change.

# %%
m = blpd.model()
m.update_p(t_tot=5 * 60, ustar=1.0)

# %% [markdown]
# Note that we can also make these changes at initialization.

# %%
m_alt = blpd.model(p={"t_tot": 5 * 60, "ustar": 1.0})
compare_params(m.p, m_alt.p)

# %% [markdown]
# If no 2nd argument passed, we compare to the default model instance.

# %%
compare_params(m.p)

# %% [markdown]
# We can also pass it and here obtain the same result, although the message has changed slightly.

# %%
compare_params(m.p, m0.p)

# %% [markdown]
# We can also do the comparisons without printing the messages.

# %%
compare_params(m.p, m0.p, print_message=False)

# %%
compare_params(m.p, m.p, print_message=False)

# %% [markdown]
# We can also elect to show only input parameters.

# %%
compare_params(m.p, input_params_only=True)
