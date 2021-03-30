"""
:class:`Model` class for running the LPD model, input parameters, ...
"""
import importlib
import math
import warnings
from copy import deepcopy as copy

import numpy as np

from . import lpd
from .utils import disable_numba
from .utils import enable_numba
from .utils import numbify
from .utils import unnumbify


input_param_defaults = {
    #
    # particle sources
    "release_height": 1.05,  # m; really should draw from a distribution of release heights for each particle!!
    "source_positions": [(0, 0)],  # point locs of the particle sources; must be iterable
    "dNp_per_dt_per_source": 2,  # should be int
    #
    # canopy
    "canopy_height": 1.1,  # m; h / h_c
    "total_LAI": 2.0,  # (total) leaf area index
    "foliage_drag_coeff": 0.2,  # C_d
    #
    # turbulence
    "ustar": 0.25,  # m; u_*; friction velocity above canopy (assuming const shear layer)
    "von_Karman_constant": 0.4,  # k
    "Kolmogorov_C0": 5.5,
    #
    # run options
    "dt": 0.25,  # s; time step for the 1-O Newton FT scheme; this is what Pratt used
    "t_tot": 100.0,  # s; total time of the run
    "dt_out": 1.0,
    "continuous_release": True,
    "use_numba": True,
    "chemistry_on": False,
    #
    # chemistry
    "fv_0": {},  # fv: floral volatiles. initial values (mol, ug, or somesuch) can be provided here.
    "n_air_cm3": 2.62e19,  # (dry) air number density (molec cm^-3)
    "oxidants_ppbv": {"O3": 40.0, "OH": 1.0e-4, "NO3": 1.0e-5},
    #
    # Massman and Weil (MW) canopy wind model parameters (could be dict)
    "MW_c1": 0.28,  # c's for above-canopy wind profile
    "MW_c2": 0.37,
    "MW_c3": 15.1,
    "MW_gam1": 2.40,  # gam_i = sig_i/u_star (velocity std's above canopy, in sfc layer)
    "MW_gam2": 1.90,
    "MW_gam3": 1.25,
    "MW_alpha": 0.05,  # parameter that controls in-canopy sigma_w and sigma_u
    "MW_A2": 0.6  # appears to be unused, but is part of their model
    # only alpha and A_1 are empirical constants (MW p. 89)
}
# could do more dict nesting like in pyAPES...


# TODO: opposite of this -- calculate above MW params from normal wind params
def calc_MW_derived_params(p):
    """
    from the base MW params and

    ref: Massman and Weil (1999) [MW]
    """
    cd = p["foliage_drag_coeff"]
    ustar = p["ustar"]
    LAI = p["total_LAI"]
    h = p["canopy_height"]
    kconstant = p["von_Karman_constant"]

    c1 = p["MW_c1"]
    c2 = p["MW_c2"]
    c3 = p["MW_c3"]
    gam1 = p["MW_gam1"]
    gam2 = p["MW_gam2"]
    gam3 = p["MW_gam3"]
    alpha = p["MW_alpha"]

    # derived params
    nu1 = (gam1 ** 2 + gam2 ** 2 + gam3 ** 2) ** (-0.5)  # MW p. 86
    nu3 = (gam1 ** 2 + gam2 ** 2 + gam3 ** 2) ** (1.5)
    nu2 = nu3 / 6 - gam3 ** 2 / (2 * nu1)
    # Lam2 = 7/(3*alpha**2*nu1*nu3) + [1/3 - gam3**2*nu1**2]/(3*alpha**2*nu1*nu2)  # the first Lambda^2
    Lam2 = 3 * nu1 ** 2 / alpha ** 2  # the simplified Lambda^2 expr; MW p. 87
    Lam = math.sqrt(Lam2)
    uh = ustar / (c1 - c2 * math.exp(-c3 * cd * LAI))  # u(h); MW Eq. 5
    n = cd * LAI / (2 * ustar ** 2 / uh ** 2)  # MW Eq. 4, definitely here "n" not "nu"
    B1 = -(9 * ustar / uh) / (2 * alpha * nu1 * (9 / 4 - Lam ** 2 * ustar ** 4 / uh ** 4))

    d = h * (1 - (1 / (2 * n)) * (1 - math.exp(-2 * n)))  # displacement height

    z0 = (h - d) * math.exp(-kconstant * uh / ustar)  # roughness length

    # Calculate dissipation at canopy top to choose matching approach (Massman and Weil)
    epsilon_ah = (ustar ** 3) / (kconstant * (h - d))
    sig_eh = ustar * (nu3) ** (1 / 3)  # not used elsewhere
    epsilon_ch = sig_eh ** 3 * (cd * LAI / h) / (nu3 * alpha)
    if epsilon_ah >= epsilon_ch:
        epflag = True
    else:  # eps_a(h) < eps_c(h)  => usually indicates a relatively dense canopy
        epflag = False

    r = {
        "MW_nu1": nu1,
        "MW_nu2": nu2,
        "MW_nu3": nu3,
        "MW_Lam": Lam,
        "MW_n": n,  # should be eta? (think not)
        "MW_B1": B1,
        "U_h": uh,  # mean wind speed at canopy height U(h)
        "displacement_height": d,
        "roughness_length": z0,
        "MW_epsilon_a_h": epsilon_ah,  # above-canopy TKE dissipation rate at h
        "MW_epsilon_c_h": epsilon_ch,  # in-canopy " "
        "MW_epsilon_ah_gt_ch": epflag,  # name needs to be more descriptive
    }

    return r


def compare_params(p, p0=None, input_params_only=False):
    """Compare `p` to reference `p0` (params dicts)."""

    if p0 is None:
        p0 = Model().p
        p0_name = "default"
    else:
        p0_name = "reference"

    if any(p[k] != p0[k] for k in p0):

        if input_params_only:
            p0 = {k: p0[k] for k in p0 if k in input_param_defaults}

        t = f"parameter: {p0_name} --> current"
        print(t)
        print("-" * len(t))
        for k, v0 in sorted(p0.items(), key=lambda x: x[0].lower()):  # don't separate uppercase
            v = p[k]
            if v0 != v:
                print(f"'{k}': {v0} --> {v}")
    else:
        print("all params same as defaults")


class Model:
    """The LPD model."""

    # class variables (as opposed to instance)
    _p_user_input_default = input_param_defaults
    _p_default_MW = calc_MW_derived_params(_p_user_input_default)
    _p_input_default = {**_p_user_input_default, **_p_default_MW}

    def __init__(self, pu={}):
        """
        pu : dict
            user-supplied parameters (to update the defaults)
        """
        self.p = copy(Model._p_input_default)  # start with defaults
        self.update_p(pu)  # update based on user input

        # checks (could move to separate `check_p` method or to `update_p`)
        assert (
            self.p["release_height"] <= self.p["canopy_height"]
        ), "particles must be released within canopy"
        assert (
            np.modf(self.p["dt_out"] / self.p["dt"])[0] == 0
        ), "output interval must be a multiple of dt"

        self._init_state()
        self._init_hist()

    def update_p(self, pu):
        """Use the dict `pu` of allowed user input parameters to check/update all model parameters."""
        if not isinstance(pu, dict):
            raise TypeError("must pass `dict`")

        allowed_keys = Model._p_user_input_default.keys()
        for k, v in pu.items():
            if k not in allowed_keys:
                msg = f"key '{k}' is not in the default parameter list. ignoring it."
                warnings.warn(msg)
            else:
                if isinstance(v, dict):
                    self.p[k].update(v)
                else:
                    self.p[k] = v

        # calculated parameters (probably should be in setter methods or something to prevent inconsistencies)
        # i.e., user changing them without using `update_p`
        self.p["N_sources"] = len(self.p["source_positions"])

        # calculate oxidant concentrations from ppbv values and air number density
        n_a = self.p["n_air_cm3"]
        conc_ox = {}
        for ox_name, ox_ppbv in self.p["oxidants_ppbv"].items():
            conc_ox[ox_name] = n_a * ox_ppbv * 1e-9
        self.p.update({"conc_oxidants": conc_ox})

        # calculate number of time steps: N_t from t_tot
        t_tot = self.p["t_tot"]
        dt = self.p["dt"]
        N_t = math.floor(t_tot / dt)  # number of time steps
        if abs(N_t - t_tot / dt) > 0.01:
            msg = f"N was rounded down from {t_tot/dt:.4f} to {N_t}"
            warnings.warn(msg)
        self.p["N_t"] = N_t  # TODO: consistentify style between N_p and N_t

        # calculate total run number of particles: Np_tot
        dNp_dt_ds = self.p["dNp_per_dt_per_source"]
        N_s = self.p["N_sources"]
        if self.p["continuous_release"]:
            Np_tot = N_t * dNp_dt_ds * N_s
            Np_tot_per_source = int(Np_tot / N_s)

        else:
            Np_tot = dNp_dt_ds * N_s
            Np_tot_per_source = dNp_dt_ds
        self.p["Np_tot"] = Np_tot
        self.p["Np_tot_per_source"] = Np_tot_per_source

        # some variables change the lengths of the state and hist arrays
        if any(
            k in pu for k in ["t_tot", "dNp_per_dt_per_source", "N_sources", "continuous_release"]
        ):
            self._init_state()
            self._init_hist()

        # some variables affect the derived MW variables
        #
        # these are the non-model-parameter inputs:
        MW_inputs = [
            "foliage_drag_coeff",
            "ustar",
            "total_LAI",
            "canopy_height",
            "von_Karman_constant",
        ]
        # check for these and also the MW model parameters
        if any(k in pu for k in MW_inputs) or any(k[:2] == "MW" for k in pu):
            self.p.update(calc_MW_derived_params(self.p))

        return self

    # TODO: could change self.p to self._p, but have self.p return a view,
    #       but give error if user tries to set items

    # TODO: init conc at this time too?
    def _init_state(self):
        Np_tot = self.p["Np_tot"]
        # also could do as 3-D? (x,y,z coords)

        # particle positions
        xp = np.empty((Np_tot,))
        yp = np.empty((Np_tot,))
        zp = np.empty((Np_tot,))

        # local wind speed (perturbation?) at particle positions
        up = np.zeros((Np_tot,))  # should initial u be = horiz wind speed? or + a perturbation?
        vp = np.zeros((Np_tot,))
        wp = np.zeros((Np_tot,))

        # seems ideal to generate sources and initial pos for each before going into time loop
        # can't be generator because we go over them multiple times
        # but could have a generator that generatoes that sources for each time step?
        N_sources = self.p["N_sources"]
        Np_tot_per_source = self.p["Np_tot_per_source"]
        release_height = self.p["release_height"]
        source_positions = self.p["source_positions"]
        for isource in range(N_sources):
            ib = isource * Np_tot_per_source
            ie = (isource + 1) * Np_tot_per_source
            xp[ib:ie] = source_positions[isource][0]
            yp[ib:ie] = source_positions[isource][1]
            zp[ib:ie] = release_height

        # rearrange by time (instead of by source)
        for p_ in (xp, yp, zp):
            groups = []
            for isource in range(N_sources):
                ib = isource * Np_tot_per_source
                ie = (isource + 1) * Np_tot_per_source
                groups.append(p_[ib:ie])
            p_[:] = np.column_stack(groups).flatten()
        # assuming sources same strength everywhere for now

        self.state = {
            # 'k': 0,
            # 't': 0,
            # 'Np_k': 0,
            "xp": xp,
            "yp": yp,
            "zp": zp,
            "up": up,
            "vp": vp,
            "wp": wp,
        }

    def _init_hist(self):
        if self.p["continuous_release"]:
            hist = False
        else:
            # set up hist if dt_out
            # should use xarray for this (and for other stuff too; like sympl)
            if self.p["dt_out"] <= 0:
                raise ValueError("dt_out must be pos. to use single-release mode")
                # TODO: should be ok to do continuous_release without hist if want

            t_tot = self.p["t_tot"]
            dt_out = self.p["dt_out"]
            Np_tot = self.p["Np_tot"]
            hist = dict()
            N_t_hist = int(t_tot / dt_out) + 1
            hist["pos"] = np.empty((Np_tot, N_t_hist, 3))  # particle, x, y, z
            hist["ws"] = np.empty((Np_tot, N_t_hist, 3))
            # could use list instead of particle dim
            # to allow for those with different record lengths

            xp = self.state["xp"]
            yp = self.state["yp"]
            zp = self.state["zp"]
            up = self.state["up"]
            vp = self.state["vp"]
            wp = self.state["wp"]
            hist["pos"][:, 0, :] = np.column_stack((xp, yp, zp))  # initial positions
            hist["ws"][:, 0, :] = np.column_stack((up, vp, wp))

        self.hist = hist

    def run(self):
        """run dat model"""
        import datetime

        # TODO: change to `time.perf_counter` for this?
        self._clock_time_run_start = datetime.datetime.now()

        Np_k = 0  # initially tracking 0 particles
        # Np_tot = self.p['Np_tot']
        dt = self.p["dt"]
        dt_out = self.p["dt_out"]
        N_t = self.p["N_t"]  # number of time steps
        # t_tot = self.p['t_tot']
        dNp_dt_ds = self.p["dNp_per_dt_per_source"]
        N_s = self.p["N_sources"]
        # outer loop could be particles instead of time. might make some parallelization easier

        # init of hist and state could go here

        if self.p["use_numba"]:
            enable_numba()  # ensure numba compilation is not disabled

            # > prepare p for numba
            p_for_nb = {k: v for k, v in self.p.items() if not isinstance(v, (str, list, dict))}
            # p_for_nb = {k: v for k, v in self.p.items() if isinstance(v, (int, float, np.ndarray))}
            # p_nb = numbify(p_for_nb)
            p_nb = numbify(p_for_nb, zerod_only=True)  # only floats/ints

            # > prepare state for numba
            #  leaving out t and k for now, pass as arguments instead
            # state_for_nb = {k: v for k, v in self.state.items() if k in ('xp', 'yp', 'zp', 'up', 'vp', 'wp')}
            state_for_nb = self.state
            state_nb = numbify(state_for_nb)

            # for debug
            self.p_nb = p_nb
            self.state_nb = state_nb

            state_run = state_nb
            p_run = p_nb

        else:  # model is set not to use numba (for checking the performance advantage of using numba)
            disable_numba()  # disable numba compilation

            state_run = self.state
            p_run = self.p

        # changing numba config in lpd redefines the njit decorated functions
        # but that isn't recognized right away unless we do this re-import
        # reloading numba in lpd doesn't seem to work
        # - at least not the first time trying to run after changing use_numba True->False
        importlib.reload(lpd)

        # print(state_run['xp'].shape)

        for k in range(1, N_t + 1):

            if self.p["continuous_release"]:
                Np_k += dNp_dt_ds * N_s
            else:  # only release at k=1 (at k=0 the particles are inside their release point)
                if k == 1:
                    Np_k += dNp_dt_ds * N_s
                else:
                    pass

            t = k * dt  # current (elapsed) time

            if self.p["use_numba"]:
                state_run.update(numbify({"k": k, "t": t, "Np_k": Np_k}))
            else:
                state_run.update({"k": [k], "t": [t], "Np_k": [Np_k]})
            # print(state_run['xp'].shape)

            # pass numba-ified dicts here

            lpd.integrate_particles_one_timestep(state_run, p_run)
            # integrate_particles_one_timestep(state_run, p_run)

            # TODO: option to save avg / other stats in addition to instantaneous? or specify avg vs instant?
            if self.hist is not False:
                if t % dt_out == 0:
                    o = int(t // dt_out)  # note that `int()` floors anyway
                    xp = state_run["xp"]
                    yp = state_run["yp"]
                    zp = state_run["zp"]
                    up = state_run["up"]
                    vp = state_run["vp"]
                    wp = state_run["wp"]
                    self.hist["pos"][:, o, :] = np.column_stack((xp, yp, zp))
                    self.hist["ws"][:, o, :] = np.column_stack((up, vp, wp))

        if self.p["use_numba"]:
            self.state = unnumbify(state_run)
        else:
            self.state = state_run

        self._clock_time_run_end = datetime.datetime.now()

        # self._maybe_run_chem()

        return self

    # def _maybe_run_chem(self):
    #     # note: adds conc dataset to self.state

    #     # this check/correction logic could be somewhere else
    #     if self.p['chemistry_on']:
    #         if not self.p['continuous_release']:
    #             warnings.warn(
    #                 'chemistry is calculated only for the continuous release option (`continuous_release=True`). not calculating chemistry',
    #                 stacklevel=2,
    #             )
    #             self.p['chemistry_on'] = False

    #     if self.p["chemistry_on"]:
    #         conc = chem_calc_options["fixed_oxidants"](self.to_xarray())

    #     else:
    #         conc = False

    #     self.state.update({
    #         'conc': conc
    #     })
    #     # maybe should instead remove 'conc' from state if not doing chem

    def to_xarray(self):
        """Returns :class:`xarray.Dataset` of the LPD run."""
        # TODO: smoothing/skipping options to reduce storage needed?
        import json
        import xarray as xr

        ip_coord_tup = (
            "ip",
            np.arange(self.p["Np_tot"]),
            {"long_name": "Lagrangian particle index"},
        )
        if self.hist:  # continuous release run
            t = np.arange(0, self.p["t_tot"] + self.p["dt_out"], self.p["dt_out"])
            # ^ note can use `pd.to_timedelta(t, unit="s")`
            dims = ("ip", "t")
            coords = {
                "ip": ip_coord_tup,
                "t": ("t", t, {"long_name": "Simulation elapsed time", "units": "s"}),
            }
            x = self.hist["pos"][..., 0]
            y = self.hist["pos"][..., 1]
            z = self.hist["pos"][..., 2]
            u = self.hist["ws"][..., 0]
            v = self.hist["ws"][..., 1]
            w = self.hist["ws"][..., 2]
        else:  # no hist, only current state
            dims = ("ip",)
            coords = {"ip": ip_coord_tup}
            x = self.state["xp"]
            y = self.state["yp"]
            z = self.state["zp"]
            u = self.state["up"]
            v = self.state["vp"]
            w = self.state["wp"]

        data_vars = {
            "x": (dims, x, {"long_name": "$x$", "units": "m"}),
            "y": (dims, y, {"long_name": "$y$", "units": "m"}),
            "z": (dims, z, {"long_name": "$z$", "units": "m"}),
            "u": (dims, u, {"long_name": "$u$", "units": "m s$^{-1}$"}),
            "v": (dims, v, {"long_name": "$v$", "units": "m s$^{-1}$"}),
            "w": (dims, w, {"long_name": "$w$", "units": "m s$^{-1}$"}),
        }

        # Serialize model parameters in JSON to allow saving in netCDF and loading later
        attrs = {
            "run_completed": self._clock_time_run_end,
            "run_runtime": self._clock_time_run_end - self._clock_time_run_start,
            # TODO: package version once the packaging is better...
            "p_json": json.dumps(self.p),
        }

        ds = xr.Dataset(
            coords=coords,
            data_vars=data_vars,
            attrs=attrs,
        )

        # TODO: Add extra useful coordinates: t_out for non-hist and t as Timedelta for hist

        return ds

    def plot(self, **kwargs):
        """Default plot of results based on run type."""
        # first check if model has been run
        p = self.p
        state = self.state
        hist = self.hist
        from . import plots

        if np.all(state["up"] == 0):  # model probably hasn't been run
            pass  # silently do nothing for now
        else:
            if (p["continuous_release"] is False) and hist:
                plots.trajectories(self.to_xarray(), **kwargs)
            else:
                plots.final_pos_scatter(self.to_xarray(), **kwargs)
