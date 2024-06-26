"""
Chemistry and emissions calculations
on LPD (Lagrangian particle dispersion) model output.

>>> ds0 = blpd.Model().run().to_xr()  # LPD output
>>> ds = blpd.chem.calc_relative_levels_fixed_oxidants(ds0)
"""

import numpy as np
import xarray as xr
from scipy import stats

from .utils import auto_bins_xy
from .utils import cache
from .utils import calc_t_out
from .utils import load_p

__all__ = (
    "calc_areal_emission_rates_canola",
    "calc_gridded_conc_canola",
    "calc_relative_levels_fixed_oxidants",
    "load_canola_species_data",
    "load_canola_flower_basal_emissions",
    "load_default_chemical_species_data",
)


@cache
def load_default_chemical_species_data():
    """Chemical species data used by default in `calc_relative_levels_fixed_oxidants`.
    Returns
    -------
    dict
        *keys*: ASCII chemical species names;
        *values*: another dict with *values*:
        `'display_name'`, `'kO3'`, `'kOH'`, `'kNO3'`
    """
    # key, display name string, kO3, kOH, kNO3
    # for now must have space after `,` !
    s = """
apinene, "α-pinene", 8.1e-17, 5.3e-11, 6.2e-12
bocimene, "β-ocimene", 5.4e-16, 2.52e-10, 2.2e-11
bpinene, "β-pinene", 2.4e-17, 7.8e-11, 2.5e-12
carene, "3-carene", 3.8e-17, 8.7e-11, 9.1e-12
cineole, "1,8-cineole", 6.0e-20, 1.0e-11, 1.7e-16
farnesene, "α-farnesene", 1.0e-15, 3.2e-10, 5.5e-11
limonene, "d-limonene", 2.5e-16, 1.6e-10, 1.3e-11
linalool, "linalool", 4.3e-16, 1.6e-10, 1.1e-11
myrcene, "β-myrcene", 4.7e-16, 2.1e-10, 1.3e-11
sabinene, "sabinene", 9.0e-17, 1.2e-10, 1.0e-11
thujene, "α-thujene", 4.4e-16, 8.7e-11, 1.1e-11
    """.strip()
    # TODO: store in separate file that makes it easier to add more (and other data). YAML?

    d = {}
    for line in s.split("\n"):
        parts = [p.strip() for p in line.split(", ")]
        dl = {
            "display_name": eval(parts[1]),
            "kO3": float(parts[2]),
            "kOH": float(parts[3]),
            "kNO3": float(parts[4]),
        }
        key = parts[0]
        d[key] = dl

    return d


# TODO: for relative levels, should always be 0--1 (or 0--100% optionally), so should remove `fv_0_default` and coordinate with gridded calc
def calc_relative_levels_fixed_oxidants(
    ds,
    species=None,
    *,
    t_out=None,
    p_overrides=None,
    fv_0_default: float = 100.0  # percentage mode by default
):
    """Calculate relative levels (rt. values at particle release)
    using fixed (over space and time) oxidant levels.

    The fixed oxidant levels greatly simplify the problem—fractional chemical destruction only depends on time-since-release.
    Since we release particles in a known non-random way (fixed release rate),
    we can derive time-since-release after the LPD run has finished,
    without even needing the trajectory data.
    (This assumes the source strengths remain constant as well.)

    Parameters
    ----------
    ds : xr.Dataset
        LPD model output generated by `blpd.model.Model.to_xarray`.
    species : dict
        *keys*: the species to calculate levels for;
        *values*: a `dict` for each species that must at least have the rate constants
        ``'kO3'``, ``'kOH'``, ``'kNO3'``
        but can also have ``'fv_0'``, the in-particle concentration at release.
        By default, the standard dataset `CHEMICAL_SPECIES_DATA` is used.
    fv_0_default
        Set to `1.0` instead for fractional mode (default is percentage mode)
        or pick a different value.
        Used for species in `species` that don't provide `'fv_0'` in their dict.

    Returns
    -------
    xr.Dataset
    """
    if species is None:
        species = load_default_chemical_species_data()

    p = load_p(ds)

    if p_overrides is not None:
        p.update(p_overrides)

    # unpack needed model options/params
    Np_tot = p["Np_tot"]

    if t_out is None:
        t_out = calc_t_out(p)

    # (molec cm^-3)
    conc_O3 = p["conc_oxidants"]["O3"]
    conc_OH = p["conc_oxidants"]["OH"]
    conc_NO3 = p["conc_oxidants"]["NO3"]

    # Save position dataset to merge with
    ds0 = ds.copy()

    ip = np.arange(Np_tot)  # for now
    spcs = sorted(species.keys())  # species keys
    n_spc = len(spcs)

    def d0():
        return np.empty((ip.size, n_spc))  # to be filled; or create d0 and use .copy()

    ds = xr.Dataset(
        coords={
            "ip": ("ip", ip, {"long_name": "Lagrangian particle index"}),
            "spc": ("spc", spcs, {"long_name": "chemical species"}),
        },
        data_vars={
            "f_r": (("ip", "spc"), d0(), {"long_name": "fraction remaining"}),
            "f_d_o3": (("ip", "spc"), d0(), {"long_name": "fraction destroyed by O3"}),
            "f_d_oh": (("ip", "spc"), d0(), {"long_name": "fraction destroyed by OH"}),
            "f_d_no3": (("ip", "spc"), d0(), {"long_name": "fraction destroyed by NO3"}),
        },
        attrs={
            "conc_O3": conc_O3,  # molec cm-3
            "conc_OH": conc_OH,
            "conc_NO3": conc_NO3,
            "p_json": ds0.attrs["p_json"],
        },
    )
    for spc, d_spc in species.items():
        conc0_val = p["fv_0"].get(spc, fv_0_default)

        kO3 = d_spc["kO3"]
        kOH = d_spc["kOH"]
        kNO3 = d_spc["kNO3"]

        # k_sum = kO3 + kOH + kNO3
        # k_inv_sum = 1/kO3 + 1/kOH + 1/kNO3
        kc_sum = conc_O3 * kO3 + conc_OH * kOH + conc_NO3 * kNO3
        kc_O3 = kO3 * conc_O3
        kc_OH = kOH * conc_OH
        kc_NO3 = kNO3 * conc_NO3

        # fraction remaining
        f_remaining = (
            1.0 * np.exp(-kc_O3 * t_out) * np.exp(-kc_OH * t_out) * np.exp(-kc_NO3 * t_out)
        )

        # fraction destroyed
        # breakdown by oxidant depends on the rate const and oxidant levels
        f_destroyed = 1 - f_remaining  # total
        f_d_O3 = kc_O3 / kc_sum * f_destroyed
        f_d_OH = kc_OH / kc_sum * f_destroyed
        f_d_NO3 = kc_NO3 / kc_sum * f_destroyed
        # note: transforming from f_r to these is linear
        # so could be done after binning, but we might want to plot these particle-wise

        f_r_spc = np.full((Np_tot,), conc0_val) * f_remaining

        # update dataset values
        ds["f_r"].loc[:, spc] = f_r_spc
        ds["f_d_o3"].loc[:, spc] = f_d_O3
        ds["f_d_oh"].loc[:, spc] = f_d_OH
        ds["f_d_no3"].loc[:, spc] = f_d_NO3

    ds_ret = xr.merge([ds, ds0[["x", "y", "z"]]])
    ds_ret.attrs.update(ds.attrs)

    # Add display names
    display_names = [d["display_name"] for d in species.values()]
    ds_ret["display_name"] = (
        "spc",
        display_names,
        {"long_name": "better chemical species names (UTF-8)"},
    )

    return ds_ret


@cache
def load_canola_species_data():
    """Canola species data as `dict` of `dict`s.

    Returns
    -------
    dict
        *keys*: ASCII chemical species names;
        *values*: another dict with *values*:
        `'MW'`, `'kO3'`, `'kOH'`, `'kNO3'`, `'OH_yield'`, `'HCHO_yield'`, `'display_name'`
    """

    def try_float(v):
        try:
            return float(v)
        except ValueError:
            return v

    fields = ["key", "MW", "kO3", "kOH", "kNO3", "OH_yield", "HCHO_yield", "display_name"]
    # ,mw,ko3,koh,kno3,oh_yield,hcho_yield,display_name
    s = """
apinene,136,8.09e-17,5.33e-11,6.16e-12,0.83,0.19,α-pinene
bpinene,136,2.35e-17,7.81e-11,2.51e-12,0.35,0.45,β-pinene
carene,136,3.8e-17,8.7e-11,9.1e-12,0.86,0.21,3-carene
cineole,154,6e-20,1e-11,1.7e-16,0.01,0,"1,8-cineole"
farnesene,204,1e-15,3.2e-10,5.5e-11,0.6,0.15,α-farnesene
limonene,136,2.5e-16,1.61e-10,1.3e-11,0.67,0.19,d-limonene
linalool,154,4.3e-16,1.6e-10,1.1e-11,0.72,0.36,linalool
myrcene,154,4.7e-16,2.1e-10,1.27e-11,0.63,0.74,β-myrcene
sabinene,136,9e-17,1.17e-10,1e-11,0.33,0.5,sabinene
thujene,136,4.4e-16,8.7e-11,1.1e-11,0.69,0.36,α-thujene
    """.strip()

    import csv

    lines = s.splitlines()
    d = {}
    for row in csv.DictReader(lines, fieldnames=fields):
        dl = {k: try_float(v) for k, v in row.items() if k != "key"}
        d[row["key"]] = dl

    return d


@cache
def load_canola_flower_basal_emissions():
    """Speciated basal emission rate data for canola plant."""

    # basal emissions in ng per floret-hour (ng floret^-1 hr^-1)
    # from Table 1 of Jakobsen et al. (1994) (the values for light conditions)
    # doi: https://doi.org/10.1016/S0031-9422(00)90341-8
    emiss_ng = {
        "myrcene": 12.3,
        "limonene": 7.2,
        "sabinene": 6.5,
        "farnesene": 5.0,
        "apinene": 4.2,
        "linalool": 3.0,
        "carene": 1.9,
        "bpinene": 2.0,
        "thujene": 1.9,
        "cineole": 1.7,
    }

    # convert to ug/s units
    emiss_ug_s = {}
    for spc, e_ng in emiss_ng.items():
        # calculate emission in ug (micrograms) per second
        e_ug_s = e_ng / 1000 / (60 * 60)
        emiss_ug_s[spc] = e_ug_s

    # reference temperature
    # near the end of the paper they say that the
    # Sampling was performed at 15 deg
    T_S_C = 15.0  # deg. C
    T_S_K = T_S_C + 273.15  # Kelvin

    return {
        "emiss_ug_per_floret-s": emiss_ug_s,
        "T_S_K": T_S_K,
    }


def calc_areal_emission_rates_canola(
    plant_density: float = 30.0,
    num_flowers_avg: float = 25.0,
    T: float = 298.0,
    beta: float = 0.06,
):
    """Calculate areal emission rates (m$^{-2}$ s$^{-1}$) for canola floral volatiles.

    Parameters
    ----------
    plant_density
        Average number of plants per m$^2$.
    num_flowers_avg
        Average number of flowers in the plants.
    T
        Temperature (K).

        Temperature sensitivity parameter for emissions (K$^{-1}$).

    Returns
    -------
    dict
        *values*: species ASCII; *keys*: dict of areal emissions in different units
    """

    # load basal data
    basal = load_canola_flower_basal_emissions()
    E_si_ug = basal["emiss_ug_per_floret-s"]
    T_S = basal["T_S_K"]

    # other needed
    N_A = 6.02214076e23  # also in scipy.constants

    # load canola species additional data
    data = load_canola_species_data()

    # loop through species
    E_i = {}
    for spc, e_si in E_si_ug.items():
        # ug per floret-s at temperature T
        e_i = e_si * np.exp(beta * (T - T_S))

        # compute areal emissions

        # ug per s per m^2
        e_i_areal = e_i * num_flowers_avg * plant_density

        # mol " "
        mw = data[spc]["MW"]  # molecular weight (g/mol)
        e_i_areal_mol = e_i_areal / 1e6 / mw  # note ug -> g

        # molec " "
        e_i_areal_molec = e_i_areal_mol * N_A

        E_i[spc] = {
            "emiss_areal_ug": e_i_areal,
            "emiss_areal_mol": e_i_areal_mol,
            "emiss_areal_molec": e_i_areal_molec,
        }

    return E_i


def calc_gridded_conc_canola(ds):
    """Some calculations for canola floral volatiles
    using the fixed oxidants chem.

    Parameters
    ----------
    ds : xr.Dataset
        LPD model output generated by `blpd.model.Model.to_xarray`.
    """
    # ! 2-D for now (not real 3-D/volume conc.)
    p = load_p(ds)

    # 1. Determine floral volatile levels per particle
    emiss_rates = calc_areal_emission_rates_canola()
    dt = p["dt"]
    dNp_per_dt_per_source = p["dNp_per_dt_per_source"]
    Np_per_sec_per_source = dNp_per_dt_per_source / dt
    fv_0 = {}  # floral volatile initial amounts in particle
    for spc, d in emiss_rates.items():
        # assume 1 m^2 source area
        # TODO: source's effective area for emissions calculation can be an input somewhere?
        # and that it does not overlap with other sources
        molec_per_p = d["emiss_areal_molec"] / Np_per_sec_per_source
        fv_0[spc] = molec_per_p

    # 2. Compute particle-relative levels
    #    decreased due to chemical destruction by oxidation
    canola = load_canola_species_data()
    ds_fo = calc_relative_levels_fixed_oxidants(ds, species=canola, fv_0_default=1.0)
    # ^ can also pass fv_0
    spcs_fo = ds_fo["spc"].values

    # 3. grid
    X, Y = ds.x.values, ds.y.values
    dx = X[1] - X[0]
    dy = Y[1] - Y[0]
    pos = (X, Y)
    bins = auto_bins_xy(pos)

    # 4. Calculate concentrations from particle concentration field
    #    and speciated fv_0
    res = {}  # store results here
    ret = None  # for first run we don't have a result yet
    stats_to_calc = ["sum", "count", "mean", "median", "std"]
    for spc in spcs_fo:
        f_r_spc = ds_fo["f_r"].sel(spc=spc).values  # fraction remaining, by particle
        f_d_o3_spc = ds_fo["f_d_o3"].sel(spc=spc).values  # fraction destroyed by O3
        # Calculate each of the stats
        rets = {}
        variables = {"f_r_spc": f_r_spc, "f_d_o3": f_d_o3_spc}
        for vn, values in variables.items():
            rets_vn = {}
            for stat in stats_to_calc:
                ret = stats.binned_statistic_dd(
                    np.column_stack(pos),  # binning by
                    values,  # calculating the statistic on
                    statistic=stat,
                    bins=bins,
                    binned_statistic_result=ret,
                )
                rets_vn[stat] = ret
            rets[vn] = rets_vn

        # Gridded particle count
        # should be same for all, don't really need to re-calc this way...
        n_p_g = rets["f_r_spc"]["count"].statistic

        # Calculate conc. by summing the relative concs. of particles
        # and scaling by the particle initial # of molec
        f_r_sum = rets["f_r_spc"]["sum"].statistic
        fv_spc_g = f_r_sum * fv_0[spc]

        # Link molec destroyed by O3 to OH and HCHO yields
        # via the ozonolysis oxidation pathway
        f_d_o3_sum = rets["f_d_o3"]["sum"].statistic
        oh_yield_oz_spc = canola[spc]["OH_yield"]
        hcho_yield_oz_spc = canola[spc]["HCHO_yield"]
        oh_y_tot = f_d_o3_sum * fv_0[spc] * fv_0[spc] * oh_yield_oz_spc
        hcho_y_tot = f_d_o3_sum * fv_0[spc] * fv_0[spc] * hcho_yield_oz_spc

        # xr.Datset per species then combine later??

        # collect
        res[spc] = {
            "particle_count": n_p_g,
            "molec_count": fv_spc_g,
            "conc_molec_areal": fv_spc_g / (dx * dy),
            "oh_yield_count": oh_y_tot,
            "oh_yield_areal": oh_y_tot / (dx * dy),
            "hcho_yield_count": hcho_y_tot,
            "hcho_yield_areal": hcho_y_tot / (dx * dy),
        }

    # 5. Specify grid centers
    x, y = ret.bin_edges[0], ret.bin_edges[1]
    xc = x[:-1] + 0.5 * np.diff(x)
    yc = y[:-1] + 0.5 * np.diff(y)

    # Create xr.Dataset

    # Specify units and long names for the above-calculated variables
    units_long_names = {
        "particle_count": ("", "Lagrangian particle count"),
        "molec_count": ("molec", "molecule count"),
        "conc_molec_areal": ("molec m^-2", "areal conc. (molec.)"),
        "oh_yield_count": ("radical", "OH yield from ozonolysis"),
        "oh_yield_areal": ("radical m^-2", "areal OH yield from ozonolysis"),
        "hcho_yield_count": ("molec", "HCHO yield from ozonolysis"),
        "hcho_yield_areal": ("molec m^-2", "areal HCHO yield from ozonolysis"),
    }
    units = {k: v[0] for k, v in units_long_names.items()}
    long_names = {k: v[1] for k, v in units_long_names.items()}

    # Aggregate: species as a third dim
    spc_sorted = sorted(spc for spc in res.keys())
    n_spc = len(spc_sorted)
    varnames = res[spc_sorted[0]].keys()
    res_agg = {}
    for vn in varnames:
        data = np.zeros((n_spc, xc.size, yc.size))
        for i, (spc, data_spc) in enumerate(res.items()):
            data[i, :, :] = data_spc[vn]
        res_agg[vn] = data

    ds = xr.Dataset(
        coords={
            "xe": ("xe", x, {"units": "m", "long_name": "x (bin edge)"}),
            "ye": ("ye", y, {"units": "m", "long_name": "y (bin edge)"}),
            "x": ("x", xc, {"units": "m", "long_name": "x (bin center)"}),
            "y": ("y", yc, {"units": "m", "long_name": "y (bin center)"}),
            "spc": ("spc", spc_sorted, {"long_name": "chemical species"}),
        },
        data_vars={
            vn: (
                ("spc", "x", "y"),
                values,
                {
                    "units": units[vn],
                    "long_name": long_names[vn],
                },
            )
            for vn, values in res_agg.items()
        },
        attrs={
            "t_tot_s": p["t_tot"],
        },
    )

    # Fix particle count (doesn't vary with species)
    ds["particle_count"] = ds["particle_count"].sel(spc="apinene")

    # Add display names
    display_names = [canola[spc]["display_name"] for spc in spc_sorted]
    ds["display_name"] = (
        "spc",
        display_names,
        {"long_name": "better chemical species names (UTF-8)"},
    )

    return ds


# Define chem calculation options for the main model class
chem_calc_options = {"fixed_oxidants": calc_relative_levels_fixed_oxidants}
