# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 11:01:38 2020

sample run for testing the model

@author: zmoon
"""

#from pprint import pprint
import pprint

import matplotlib.pyplot as plt

import main
from main import Model as model, get_default_params
import plots


#%% test parallelization capability?

#main.lpd.integrate_particles_one_timestep.parallel_diagnostics(level=4)


#%% examine defaults

pp = pprint.PrettyPrinter(indent=2)

# these are the ones that can be set; includes the MW derived params (only certain ones should be changed!)
print('Default params:')
pp.pprint(get_default_params())

m = model()

# this includes derived params
print('\n\nDefault params (including derived):')
pp.pprint(m.p)

# change some params
new_p = {
    'source_positions': [(-10, 10), (-5, 5), (0, 0), (-5, -5), (-10, -10)],
    't_tot': 5*60.,
    'continuous_release': False, 'dt_out': 1.0, 'dNp_per_dt_per_source': 200, 
    'release_height': 0.9, 'ustar': 0.25, 'dt': 0.001,
}
m.update_p(new_p)  # alternatively can pass on initialization 
print('\n\nDefault params (including derived, after update):')
pp.pprint(m.p)

print('\n\n\n')


#%% run

m.run()




#%% save results

#> save results
#  to test plotting codes 
#import pickle
#pickle.dump({'xp': xp, 'yp': yp, 'zp': zp, 'conc_BO': conc_BO}, open('a_24k_run.pkl', 'wb'))


#%% plots

plt.close('all')

state = m.state
p = m.p
hist = m.hist
dt = m.p['dt']

plots.pos_scatter(state, p)

# for a hist run only
if hist:
#    plots.trajectories(hist, p)
    plots.trajectories(hist, p, smooth=True, smooth_window_size=10)
    
    plots.ws_hist_all(hist, p)
    plots.ws_hist_all(hist, p, bounds=(-5, 5))
    plots.ws_hist_all(hist, p, bounds=(-10, 10))



#plots.conc(state, conc_BO, p)

#plots.conc(state, conc_BO, p, plot_type='pcolor')
#plots.conc(state, conc_BO, p, plot_type='pcolor', bins=(100, 50))
#
#
#plots.conc(state, conc_BO, p, plot_type='contourf')
#plots.conc(state, conc_BO, p, plot_type='contourf', bins=(100, 50))

