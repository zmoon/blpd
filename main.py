"""

"""

from copy import deepcopy as copy
import os
import warnings

# from numba import njit
# from numba.core import types  # in the docs: https://numba.pydata.org/numba-doc/dev/reference/pysupported.html#id6 but doesn't work
# from numba import types  # I guess this is the new method of importing 'types'
import numba  # 
# from numba.typed import Dict
import numpy as np

import lpd
import plots


p_MW_base_defaults = {
    'c1': 0.28,
    'c2': 0.37,
    'c3': 15.1,
    'gam1': 2.40,
    'gam2': 1.90, 
    'gam3': 1.25,
    'alpha': 0.05,
    'A2': 0.6
}


def get_p_MW(p):
    """
    from the base MW params and 

    ref: Massman and Weil (1999) [MW]
    """
    cd = p['foliage_drag_coeff']
    ustar = p['ustar']
    LAI = p['total_LAI']
    h = p['canopy_height']
    kconstant = p['von_Karman_constant']

    # c1 = 0.28  # the c's are only used here
    # c2 = 0.37
    # c3 = 15.1
    # gam1 = 2.40  # gam's used for tau calculations
    # gam2 = 1.90
    # gam3 = 1.25
    # alpha = 0.05  # this too
    # #A2 = 0.6  # this appears to be unused

    bd = p_MW_base_defaults
    c1 = bd['c1']
    c2 = bd['c2']
    c3 = bd['c3']
    gam1 = bd['gam1']
    gam2 = bd['gam2']
    gam3 = bd['gam3']
    alpha = bd['alpha']

    # derived params
    nu1 = (gam1**2+gam2**2+gam3**2)**(-0.5)
    nu3 = (gam1**2+gam2**2+gam3**2)**(1.5)  # ?? are these switched? (3 and 2)
    nu2 = nu3/6-gam3**2/(2*nu1)
    Lam = 3*nu1**2/alpha**2
    uh = ustar/(c1 - c2*np.exp(-c3*cd*LAI))  # u(h); Eq. 5
    n = cd*LAI/(2*ustar**2/uh**2)
    B1 = -(9*ustar/uh)/(2*alpha*nu1*(9/4-Lam**2*ustar**4/uh**4))
    d = h*(1-(1/(2*n))*(1 - np.exp(-2*n)))

    z0 = (h-d)*np.exp(-kconstant*uh/ustar)

    # Calculate dissipation at canopy top to choose matching approach (Massman and Weil)
    epsilon_ah = (ustar**3)/(kconstant*(h - d))
    sig_eh = ustar*(nu3)**(1/3)  # not used elsewhere
    epsilon_ch = sig_eh**3*(cd*LAI/h)/(nu3*alpha)
    if epsilon_ah >= epsilon_ch:
        epflag = True
    else:  # eps_a(h) < eps_c(h)  => usually indicates a relatively dense canopy
        epflag = False

    # exclude = ['p', 'cd', 'ustar', 'LAI', 'h', 'kconstant', 'exclude']
    # return {vn: val for vn, val in locals().items() if vn not in exclude}
    # ^ this is cool but I want to use more descriptive names

    r = {
        'MW_c1': c1,
        'MW_c2': c2,
        'MW_c3': c3,
        'MW_gam1': gam1,
        'MW_gam2': gam2,
        'MW_gam3': gam3,
        'MW_alpha': alpha,
        #'MW_A2': A2,
        'MW_nu1': nu1,
        'MW_nu2': nu2,
        'MW_nu3': nu3,
        'MW_Lam': Lam,
        #'MW_u_h': uh,  # wind speed at canopy height u(h)
        'U_h': uh,  # wind speed at canopy height u(h)
        'MW_n': n,
        'MW_B1': B1, 
        'displacement_height': d,
        'roughness_length': z0,
        'MW_epsilon_a_h': epsilon_ah,  # above-canopy diss rate at h
        'MW_epsilon_c_h': epsilon_ch,  # in-canopy " "
        'MW_epsilon_ah_gt_ch': epflag,  # name needs to be more descriptive
    }

    return r



rate_consts = {
    'BO_O3': 5.4e-16,
    'BO_OH': 2.52e-10,
    'BO_NO3': 2.2e-11,
}
# BO: beta-ocimene


def get_default_params():
    """
    all in one dict 
    might separate some in different sub dicts in future (like in pyAPES)
    """
    p = {
        'canopy_height': 1.1,  # m; h / h_c
        'release_height': 1.05,  # m; really should draw from a distribution of release heights for each particle!!
        'total_LAI': 2.0,  # (total) leaf area index
        'foliage_drag_coeff': 0.2,  # C_d
        'von_Karman_constant': 0.4,  # k
        'ustar': 0.25,  # m; u_*; friction velocity above canopy (assuming const shear layer)
        'Kolmogorov_C0': 5.5,
        'source_positions': [(0, 0)],  # point locs of the particle sources; must be iterable
        't_tot': 100.,  # s; total time of the run
        'dt': 0.25,  # s; time step for the 1-O Newton FT scheme; this is what Pratt used 
        'conc_fv_0': {  # fv: floral volatiles
            'BO': 100.  # 100 %; if using relative conc. this doesn't matter really
            },
        'n_air_cm3': 2.62e19,  # (dry) air number density (molec cm^-3)
        'oxidants_ppbv' : {
            'O3': 40.0,
            'OH': 1.0e-4,
            'NO3': 1.0e-5 
        },
        'dt_out': 0.,
        'continuous_release': True, 
        'dNp_per_dt_per_source': 2,  # should be int
        'use_numba': True,
        'chemistry_on': False
    }

    p.update(get_p_MW(p))
    # ^ some of these are derived and shouldn't be changed manually
    # TODO: should fix this potential issue

    return p


# def print_params(val, nesting=-4):
#     """

#     """




class Model():

    # class variables (as opposed to instance)
    _p_default = get_default_params()

    def __init__(self, pu={}):
        """
        pu : dict
            user-supplied parameters (to update the defaults)
        """
        self.p = copy(Model._p_default)  # start with defaults
        self.update_p(pu)  # update based on user input

        # checks (could move to separate `check_p` method or to `update_p`)
        assert( self.p['release_height'] <= self.p['canopy_height'] )  # particles must be released within canopy
        assert( self.p['dt_out'] % self.p['dt'] == 0 )  # output interval must be a multiple of dt
        if not self.p['use_numba']: raise NotImplementedError


        self.init_state()
        self.init_hist()


    def update_p(self, pu):
        """

        """
        allowed_keys = Model._p_default.keys()
        for k, v in pu.items():
            if k not in allowed_keys:
                msg = f'key {k} is not in the default parameter list. ignoring it.'
                warnings.warn(msg)
            else:
                self.p[k] = v

        # calculated parameters (probably should be in setter methods or something to prevent inconsistencies)
        # i.e., user changing them without using `update_p`
        self.p['N_sources'] = len(self.p['source_positions'])

        # calculate oxidant concentrations from ppbv values and air number density
        n_a = self.p['n_air_cm3']
        conc_ox = {}
        for ox_name, ox_ppbv in self.p['oxidants_ppbv'].items():
            conc_ox[ox_name] = n_a * ox_ppbv * 1e-9
        self.p.update({'conc_oxidants': conc_ox})

        # calculate number of time steps: N_t from t_tot
        t_tot = self.p['t_tot']
        dt = self.p['dt']
        N_t = np.floor(t_tot/dt).astype(int)  # number of time steps
        if abs(N_t-t_tot/dt) > 0.01:
            msg = f'N was rounded down from {t_tot/dt:.4f} to {N_t}'
            warnings.warn(msg)
        self.p['N_t'] = N_t  # TODO: consistentify style between N_p and N_t

        # calculate total run number of particles: Np_tot
        dNp_dt_ds = self.p['dNp_per_dt_per_source']
        N_s = self.p['N_sources']
        if self.p['continuous_release']:
            Np_tot = N_t * dNp_dt_ds * N_s
            Np_tot_per_source = (Np_tot / N_s).astype(int)

        else:
            Np_tot = dNp_dt_ds * N_s
            Np_tot_per_source = dNp_dt_ds
        self.p['Np_tot'] = Np_tot
        self.p['Np_tot_per_source'] = Np_tot_per_source

        # some variables change the lengths of the state and hist arrays
        if any( k in pu for k in ['t_tot', 'dNp_per_dt_per_source', 'N_sources', 'continuous_release'] ):
            self.init_state()
            self.init_hist()

        # some variables affect the derived MW variables
        if any( k in pu for k in ['foliage_drag_coeff', 'ustar', 'total_LAI', 'canopy_height', 'von_Karman_constant'] ):
            self.p.update(get_p_MW(self.p))


    def init_state(self):
        Np_tot = self.p['Np_tot']
        # also could do as 3-D? (x,y,z coords)

        # particle positions
        xp = np.empty((Np_tot, ))
        yp = np.empty((Np_tot, ))
        zp = np.empty((Np_tot, ))

        # local wind speed (perturbation?) at particle positions
        up = np.zeros((Np_tot, ))  # should initial u be = horiz wind speed? or + a perturbation?
        vp = np.zeros((Np_tot, ))
        wp = np.zeros((Np_tot, ))

        # seems ideal to generate sources and initial pos for each before going into time loop
        # can't be generator because we go over them multiple times
        # but could have a generator that generatoes that sources for each time step?
        N_sources = self.p['N_sources']
        Np_tot_per_source = self.p['Np_tot_per_source']
        release_height = self.p['release_height']
        source_positions = self.p['source_positions']
        for isource in range(N_sources):
            ib = isource * Np_tot_per_source
            ie = (isource+1) * Np_tot_per_source 
            xp[ib:ie] = source_positions[isource][0]
            yp[ib:ie] = source_positions[isource][1]
            zp[ib:ie] = release_height
        
        # rearrange by time (instead of by source)
        for p_ in (xp, yp, zp):
            groups = []
            for isource in range(N_sources):
                ib = isource * Np_tot_per_source
                ie = (isource+1) * Np_tot_per_source 
                groups.append(p_[ib:ie])
            p_[:] = np.column_stack(groups).flatten()
        # assuming sources same strength everywhere for now

        self.state = {
            # 'k': 0,
            # 't': 0,
            # 'Np_k': 0,
            'xp': xp,
            'yp': yp,
            'zp': zp,
            'up': up, 
            'vp': vp,
            'wp': wp,
            # 'test': np.r_[0.01]  # must be array, even a size 1 array (like in xr)
        }


    def init_hist(self):
        if self.p['continuous_release']:
            hist = False
        else:
            # set up hist if dt_out
            # should use xarray for this (and for other stuff too; like sympl)
            if self.p['dt_out'] <= 0:
                raise ValueError('dt_out must be pos. to use single-release mode')

            t_tot = self.p['t_tot']
            dt_out = self.p['dt_out']
            Np_tot = self.p['Np_tot']
            hist = dict()
            N_t_hist = int(t_tot/dt_out) + 1
            hist['pos'] = np.empty((Np_tot, N_t_hist, 3))  # particle, x, y, z
            hist['ws'] = np.empty((Np_tot, N_t_hist, 3))
            # could use list instead of particle dim
            # to allow for those with different record lengths

            xp = self.state['xp']
            yp = self.state['yp']
            zp = self.state['zp']
            up = self.state['up']
            vp = self.state['vp']
            wp = self.state['wp']
            hist['pos'][:,0,:] = np.column_stack((xp, yp, zp))  # initial positions
            hist['ws'][:,0,:] = np.column_stack((up, vp, wp))

        self.hist = hist


    # TODO: could change self.p to self._p, but have self.p return a view, but give error if user tries to set items


    def run(self):
        """run dat model"""

        Np_k = 0  # initially tracking 0 particles
        Np_tot = self.p['Np_tot']
        dt = self.p['dt']
        dt_out = self.p['dt_out']
        N_t = self.p['N_t']  # number of time steps
        t_tot = self.p['t_tot']
        dNp_dt_ds = self.p['dNp_per_dt_per_source']
        N_s = self.p['N_sources']
        # outer loop could be particles instead of time. might make some parallelization easier

        if self.p['use_numba']:
            #> alert lpd through env variable
            os.environ.update({'BLPD_USE_NUMBA': str(True)})

            #> prepare p for numba
            p_for_nb = {k: v for k, v in self.p.items() if not isinstance(v, (str, list, dict))}
            # p_for_nb = {k: v for k, v in self.p.items() if isinstance(v, (int, float, np.ndarray))}
            # p_nb = numbify(p_for_nb)
            p_nb = numbify(p_for_nb, zerod_only=True)  # only floats/ints

            #> prepare state for numba
            #  leaving out t and k for now, pass as arguments instead
            # state_for_nb = {k: v for k, v in self.state.items() if k in ('xp', 'yp', 'zp', 'up', 'vp', 'wp')}
            state_for_nb = self.state
            state_nb = numbify(state_for_nb)

            # for debug
            self.p_nb = p_nb
            self.state_nb = state_nb

            state_run = state_nb
            p_run = p_nb

        else:
            state_run = self.state
            p_run = self.p
        
        # print(state_run['xp'].shape)

        for k in range(1, N_t+1):

            if self.p['continuous_release']:
                Np_k += dNp_dt_ds * N_s
            else:  # only release at k=1 (at k=0 the particles are inside their release point)
                if k == 1:
                    Np_k += dNp_dt_ds * N_s
                else:
                    pass

            t = k*dt  # current (elapsed) time

            state_run.update(numbify({
                'k': k, 
                't': t, 
                'Np_k': Np_k
            }))
            # print(state_run['xp'].shape)

                    
            # pass numba-ified dicts here

            lpd.integrate_particles_one_timestep(state_run, p_run)

            # lpd.integrate_particles_one_timestep(xp, yp, zp, Np_k, p)

            if self.hist != False:
                if t % dt_out == 0:
                    o = int(t // dt_out)  # note that `int()` floors anyway
                    xp = state_run['xp']
                    yp = state_run['yp']
                    zp = state_run['zp']
                    up = state_run['up']
                    vp = state_run['vp']
                    wp = state_run['wp']
                    self.hist['pos'][:,o,:] = np.column_stack((xp, yp, zp))
                    self.hist['ws'][:,o,:] = np.column_stack((up, vp, wp))


        # self.state = state_run
        self.state = unnumbify(state_run)

        #> calculate chemistry
        #  this can be outside the time loop since the concentrations of oxidants are not changing with time
        #  so the amount of destruction only depends on the time that a given particle has been out
        #
        #  source strengths are not changing with time either

        if self.p['chemistry_on']:
            if not self.p['continuous_release']:
                warnings.warn('chemistry is calculated only for the continuous release option (`continuous_release=True`). not calculating chemistry')
                self.p['chemistry_on'] = False

        if self.p['chemistry_on']:

            # t_out = np.r_[[[(k+1)*numpart for p in range(numpart)] for k in range(N)]].flat
            #t_out = np.floor(np.arange(0, N_t, 1/Np_tot ))
            t_out = np.ravel(np.tile(np.arange(dt, (N_t+1)*dt, dt)[:,np.newaxis], dNp_dt_ds*N_s))
            # ^ need to find the best way to do this!

            # t_out = (t_out[::-1]+1) * dt
            t_out = t_out[::-1]

            assert np.isclose(t_tot, t_out[0])   # the first particle has been out for the full time

            conc_BO = np.full((Np_tot, ), self.p['conc_fv_0']['BO'])

            conc_O3 = self.p['conc_oxidants']['O3']
            conc_OH = self.p['conc_oxidants']['OH']
            conc_NO3 = self.p['conc_oxidants']['NO3']
            conc_BO *= np.exp(-rate_consts['BO_O3']*conc_O3*t_out) \
                     * np.exp(-rate_consts['BO_OH']*conc_OH*t_out) \
                     * np.exp(-rate_consts['BO_NO3']*conc_NO3*t_out)

        else:
            conc_BO = False

        self.state.update({
            'conc': {'BO': conc_BO}
        })




def numbify(d, zerod_only=False):
    """Convert dict to numba-suitable format.

    dict values must be numpy arrays (or individual floats)

    ref: https://numba.pydata.org/numba-doc/dev/reference/pysupported.html#id7
    """
    float64_array = numba.types.float64[:]  # maybe would be more efficient to pass size
    float64 = numba.types.float64
    d_nb = numba.typed.Dict.empty(
        key_type=numba.types.unicode_type,
        value_type=float64_array if not zerod_only else float64,
    )
    for k, v in d.items():
        # print(k, v, np.asarray(v).shape, np.asarray(v).shape == ())

        # d_nb[k] = np.asarray(v, dtype=np.float64)  # will convert bool to 0/1
        # d_nb[k] = np.asarray(v).astype(np.float64)

        # np.asarray leaves float/int with size=1 but shape=() so numba doesn't work
        # so hacking this for now
        if not zerod_only:
            if np.asarray(v).shape == ():
                x = np.float64((1,))
                x[:] = np.float64(v)
                d_nb[k] = x
            else:
                # d_nb[k] = np.asarray(v, dtype=np.float64)
                d_nb[k] = np.asarray(v).astype(np.float64)
        else:
            d_nb[k] = np.float64(v)

    return d_nb


def unnumbify(d_nb):
    """Convert numba dict to normal dict of numpy arrays."""
    if not isinstance(d_nb, numba.typed.Dict):
        raise TypeError('this fn is for numbified dicts')

    d = {}
    for k, v in d_nb.items():
        # if isinstance(v, numba.types.float64[:]):
        if v.shape == (1,):
            d[k] = v[0]
        else:
            d[k] = v

    return d



