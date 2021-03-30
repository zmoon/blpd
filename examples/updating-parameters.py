# -*- coding: utf-8 -*-
"""
Demonstrate updating parameters

@author: zmoon
"""
import pprint
import sys

sys.path.append("../")

import blpd
from blpd.main import input_param_defaults, compare_params


# %% View the defaults

pp = pprint.PrettyPrinter(indent=2)

msg = "All model input parameters"
print(msg)
print("-" * len(msg))
pp.pprint(input_param_defaults)

print("\n")

# %% Confirm default model instance has them

m0 = blpd.model()

compare_params(m0.p)

print("\n")

# %% Create a new instance and change some parameters

m = blpd.model()
m.update_p({"t_tot": 5 * 60, "ustar": 1.0})

compare_params(m.p)  # if no 2nd argument passed, we compare to default model instance
print()
compare_params(m.p, m0.p)  # confirm that that is the case
print()
compare_params(m.p, input_params_only=True)  #
