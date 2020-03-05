# -*- coding: utf-8 -*-
"""
Demonstrate chemical concentration plots.

@author: zmoon
"""

import sys
sys.path.append('../')

import blpdm
from blpdm import plots


#%% create and run

m = blpdm.model()

# change some params
new_p = {
    'source_positions': [(-200, -200), (500, 0), (1000, 200)],  # similar to the ones Pratt used in their Fig 4.2
    't_tot': 10*60.,  # 10 min
    'continuous_release': True,  # this is a default
    'dNp_per_dt_per_source': 4, 
    'chemistry_on': True,
}

m.update_p(new_p)

m.run()


#%% plots
#   demonstrate some of the settings

state = m.state
p = m.p

conc_BO = state['conc']['BO']  # beta-ocimene

plots.conc(state, conc_BO, p)

plots.conc(state, conc_BO, p, plot_type='pcolor')
plots.conc(state, conc_BO, p, plot_type='pcolor', bins=(100, 50))
plots.conc(state, conc_BO, p, plot_type='pcolor', bins='auto')
plots.conc(state, conc_BO, p, plot_type='pcolor', bins='auto', log_cnorm=True)


plots.conc(state, conc_BO, p, plot_type='contourf')
plots.conc(state, conc_BO, p, plot_type='contourf', bins=(100, 50))  # bin numbers
plots.conc(state, conc_BO, p, plot_type='contourf', bins='auto')
plots.conc(state, conc_BO, p, plot_type='contourf', bins='auto', log_cnorm=True)

