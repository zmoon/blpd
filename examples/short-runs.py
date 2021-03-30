# -*- coding: utf-8 -*-
"""
Demonstrate run types and default plots

@author: zmoon
"""
import sys

sys.path.append("../")

import blpd


# %% Default run - continuous release without history

m0 = blpd.model()
m0.run()
m0.plot()


# %% Trajectory run - single release with history

m1 = blpd.model()
m1.update_p(
    {
        "continuous_release": False,
        "dNp_per_dt_per_source": 1000,
        "t_tot": 10 * 60,  # 10 minutes
        "dt": 0.1,
        "dt_out": 1.0,  # a multiple of model integration time step
    }
)
m1.run()
m1.plot()
m1.plot(
    smooth=True
)  # pass keyword arguments through to the plotting function in module bldpm.plots
m1.plot(smooth_window_size=10)  # here 10 second window


# %% Single release run without history

# currently not allowed
