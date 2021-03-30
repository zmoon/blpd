"""
Code that runs Lagrangian stochastic particle dispersion
efficiently (hopefully) with the help of numba

"""
# fmt: off
# in order to pass in dicts as args for numba fn
# need to use their special dict type and specify the types of the varibles
#   https://numba.pydata.org/numba-doc/dev/reference/pysupported.html#typed-dict
import os
import sys
import warnings
from functools import wraps

import numba
import numpy as np
from numba import jit
from numba import njit
from numba import prange


@njit
def _calc_fd_params_above_canopy(pos, p):
    """Calculate the dispersion parameters above the canopy."""
    z = pos[2]

    ustar = p['ustar']
    kconstant = p['von_Karman_constant']
    d = p['displacement_height']

    z0 = p['roughness_length']
    gam1 = p['MW_gam1']
    gam2 = p['MW_gam2']
    gam3 = p['MW_gam3']

    umean = (ustar/kconstant)*np.log((z-d)/z0)
    dumeandz = (ustar/(kconstant))*(1/(z-d))

    tau11 = (ustar*gam1)**2
    tau22 = (ustar*gam2)**2
    tau33 = (ustar*gam3)**2
    tau13 = -(ustar**2)

    dtau11dz = 0
    dtau22dz = 0
    dtau33dz = 0
    dtau13dz = 0

    lam11, lam22, lam33, lam13 = \
        _calc_Rodean_lambdas(tau11, tau22, tau33, tau13)

    # Dissipation - above canopy (log wind profile)
    # ref: e.g., MW Eq. 12
    epsilon = (ustar**3)/(kconstant*(z - d))

    return (umean, dumeandz,
        # tau11, tau22, tau33, tau13,
        dtau11dz, dtau22dz, dtau33dz, dtau13dz,
        lam11, lam22, lam33, lam13,
        epsilon)


@njit
def _calc_fd_params_in_canopy(pos, p):
    """Calculate the dispersion parameters within the canopy."""
    z = pos[2]

    ustar = p['ustar']  # at h?
    kconstant = p['von_Karman_constant']
    d = p['displacement_height']
    h = p['canopy_height']
    uh = p['U_h']  # wind speed at canopy height
    cd = p['foliage_drag_coeff']
    LAI = p['total_LAI']

    # z0 = p['roughness_length']
    gam1 = p['MW_gam1']
    gam2 = p['MW_gam2']
    gam3 = p['MW_gam3']

    nu1 = p['MW_nu1']
    # nu2 = p['MW_nu2']  # unused?
    nu3 = p['MW_nu3']
    Lam = p['MW_Lam']
    n = p['MW_n']
    B1 = p['MW_B1']
    alpha = p['MW_alpha']

    epflag = p['MW_epsilon_ah_gt_ch']
    epsilon_ah = p['MW_epsilon_a_h']
    epsilon_ch = p['MW_epsilon_c_h']

    # zeta(z): "cumulative leaf drag area per unit planform area" (MW p. 82) (ζ)
    # The code we started from replace terms zeta(z)/zeta(h) by z/h.
    # This assumes a vertically homogeneous LAI dist (LAD const with z in canopy)
    # that accumulates from z=0, not z=h_c like LAI.
    # zeta(h) is replaced by cd*LAI

    umean = uh*np.exp(-n*(1-(z/h)))
    dumeandz = uh*(n/h)*np.exp(-n*(1-(z/h)))

    zet_h = cd*LAI  # zeta(h) total leaf drag area

    # ref: MW eqn. 10, p. 87
    sig_e = ustar * (
        nu3*np.exp(-Lam*zet_h*(1-z/h))
        + B1*(
            np.exp(-3*n*(1-z/h))
            - np.exp(-Lam*zet_h*(1-z/h))
        )
    )**(1./3)

    # I can't find a source for this eqn
    dsig_e2dz = (2/3)*ustar**2 * (
        ( nu3*np.exp(-Lam*zet_h*(1-z/h))
         + B1*(np.exp(-3*n*(1-z/h))
           - np.exp(-Lam*zet_h*(1-z/h))
           )
         )**(-1./3)
    ) * (
        (nu3*Lam*zet_h/h) * np.exp(-Lam*zet_h*(1-z/h))
        + B1*(
            (3*n/h)*np.exp(-3*n*(1-z/h))
            - (Lam*zet_h/h)*np.exp(-Lam*zet_h*(1-z/h))
        )
    )

    # From MW eq. 11, in the canopy:
    #   gam_i * nu_1 * sig_e = sig_i
    # and above the canopy:
    #   sig_i/u_star = gam_i    (p. 86)

    tau11 = (gam1*nu1*sig_e)**2
    dtau11dz = ((gam1*nu1)**2)*dsig_e2dz

    tau22 = (gam2*nu1*sig_e)**2
    dtau22dz = ((gam2*nu1)**2)*dsig_e2dz

    tau33 = (gam3*nu1*sig_e)**2
    dtau33dz = ((gam3*nu1)**2)*dsig_e2dz

    tau13 = -(ustar**2)*np.exp(-2*n*(1-(z/h)))
    dtau13dz = -(ustar**2)*(2*n/h)*np.exp(-2*n*(1-(z/h)))

    # Dissipation
    # ref: MW p. 88
    if epflag:
        epsilon = (sig_e**3) * zet_h / (h*nu3*alpha) * (epsilon_ah/epsilon_ch)
    else:
        if z <= d:  # log wind profile displacement height (could add small # to d)
            epsilon = (sig_e**3) * zet_h / (h*nu3*alpha)
        else:  # d < z <= h_c
            scale_choice_1 = zet_h * sig_e**3 / (nu3*alpha*ustar**3)
            scale_choice_2 = h/(kconstant*(z-d))
            # ^ this is a potential source of div by 0 (if z v close to d)
            scale_choices = np.array([scale_choice_1, scale_choice_2])
            epsilon = (ustar**3/h) * scale_choices.min()

    lam11, lam22, lam33, lam13 = \
        _calc_Rodean_lambdas(tau11, tau22, tau33, tau13)

    return (umean, dumeandz,
        # tau11, tau22, tau33, tau13,
        dtau11dz, dtau22dz, dtau33dz, dtau13dz,
        lam11, lam22, lam33, lam13,
        epsilon)


@njit
def _calc_Rodean_lambdas(tau11, tau22, tau33, tau13):
    # ref: Pratt thesis Eq. 2.14, p. 15
    lam11 = 1./(tau11 - ((tau13**2)/tau33))
    lam22 = 1./tau22
    lam33 = 1./(tau33 - ((tau13**2)/tau11))
    lam13 = 1./(tau13 - ((tau11*tau33)/tau13))

    return lam11, lam22, lam33, lam13


@njit
def calc_tends(pos, ws_local, p):
    """Calculate the position tendencies
    i.e., the velocity components ui + dui.
    """
    z = pos[2]  # x, y, z
    u1, u2, u3 = ws_local  # u, v, w

    h_c = p['canopy_height']
    C0 = p['Kolmogorov_C0']  # a Kolmogorov constant (3--10)
    dt = p['dt']

    # Calculate fd (fluid dynamics) params
    if z >= h_c:
        U1, dU1dx3, \
        dtau11dx3, dtau22dx3, dtau33dx3, dtau13dx3, \
        lam11, lam22, lam33, lam13, \
        eps \
            = _calc_fd_params_above_canopy(pos, p)
    else:
        U1, dU1dx3, \
        dtau11dx3, dtau22dx3, dtau33dx3, dtau13dx3, \
        lam11, lam22, lam33, lam13, \
        eps \
            = _calc_fd_params_in_canopy(pos, p)


    # Calculate new positions
    # ref: Pratt thesis Eqs. 2.12-13, p. 15
    # dW_j is an incremental Wiener process with a mean of 0, and a variance of dt
    # simulated as sqrt(dt)*N(0,1)  (p. 12)

    sqrt_C0eps = np.sqrt(C0*eps)
    sqrt_dt = np.sqrt(dt)

    # TODO: if wanted to be able to use other sorts of integrators,
    # would have to freeze the random seed or something?
    # (otherwise the tends will be different every time)
    # or separate the random part from the rest of the tend calculation.
    # The dist to draw from could be an input.

    # x-1 component
    randn = np.random.standard_normal()  # different for each component
    dW1 = sqrt_dt * randn
    du1 = (
        (-C0*eps/2 * (lam11*(u1-U1) + lam13*u3) + dU1dx3*u3 + dtau13dx3/2 ) * dt
        + (
            dtau11dx3*(lam11*(u1-U1) + lam13*u3)
            + dtau13dx3*(lam13*(u1-U1) + lam33*u3)
        ) * (u3/2*dt)
        + sqrt_C0eps * dW1
    )

    # x-2 component
    randn = np.random.standard_normal()
    dW2 = sqrt_dt * randn
    du2 = (
        (-C0*eps/2 * lam22*u2 + dtau22dx3*lam22*u2 * u3/2) * dt
        + sqrt_C0eps * dW2
    )

    # x-3 component
    randn = np.random.standard_normal()
    dW3 = sqrt_dt * randn
    du3 = (
        (-C0*eps/2 * (lam13*(u1-U1) + lam33*u3) + dtau33dx3/2) * dt
        + (
            dtau13dx3 * (lam11*(u1-U1) + lam13*u3)
            + dtau33dx3 * (lam13*(u1-U1) + lam33*u3)
        ) * (u3/2*dt)
        + sqrt_C0eps * dW3
    )

    # Check for too-high values
    if np.any(np.abs(np.array([du1, du2, du3])) > 50):
        print("one ws' is too high:", du1, du2, du3)
        print('umean, dumeandz, epsilon', U1, dU1dx3, eps)
        if z >= h_c:
            print('  z >= h_c')
        else:
            print('  z < h_c, z =', z)
        print('resetting new ws to 0')
        du1 = 0.
        du2 = 0.
        du3 = 0.
        u1 = 0.
        u2 = 0.
        u3 = 0.

    return [u1 + du1, u2 + du2, u3 + du3]



@njit
def _integrate_particle_one_timestep(pos, ws_local, p):
    """Take the state for one particle and integrate."""
    dt = p['dt']

    res = calc_tends(pos, ws_local, p)

    # Adjustments for undesired new z position
    # (below z = 0 or above PBL inversion)
    # TODO: original code noted that this needs work
    # This current method seems to correspond to completely elastic collision,
    # but maybe inelastic or some probability of deposition could be another option.
    z = pos[2]  # before integration
    znew = z + res[2]*dt  # res[2] is w velocity component
    if znew <= 0.1:
        res = [res[0], res[1], -res[2]]
        newpos = [pos[0] + res[0]*dt, pos[1] + res[1]*dt, 0.1]

    elif znew >= 800.:  # reflect at z_i too
        res = [res[0], res[1], -res[2]]
        newpos = [pos[0] + res[0]*dt, pos[1] + res[1]*dt, 800.]

    else:
        newpos = [p + dpdt*dt for (p, dpdt) in zip(pos, res)]

    # The pos tendencies are the (new) velocity components
    new_ws_local = res

    return newpos, new_ws_local


# @njit
@njit(parallel=True)
def integrate_particles_one_timestep(state, p):
    """Integrate all particles one time step, modifying `state` in place.

    `state` and p` must be Numba typed dicts (:class:`numba.typed.Dict`)
    in order for `@njit` to work if Numba is not disabled.
    """
    Np_k = int(state['Np_k'][0])
    xp = state['xp']
    yp = state['yp']
    zp = state['zp']
    up = state['up']
    vp = state['vp']
    wp = state['wp']

    # for i in range(Np_k):
    for i in prange(Np_k):  # pylint: disable=not-an-iterable

        # State for particle i: position and (wind) velocity
        pos = [xp[i], yp[i], zp[i]]  # position of particle i at current time step
        ws_local = [up[i], vp[i], wp[i]]  # local wind speed for particle

        # Calculate new state
        pos, ws_local = _integrate_particle_one_timestep(pos, ws_local, p)

        # Update state
        xp[i] = pos[0]
        yp[i] = pos[1]
        zp[i] = pos[2]
        up[i] = ws_local[0]
        vp[i] = ws_local[1]
        wp[i] = ws_local[2]

        # Note that we are not really time-integrating the wind speed.
        # We are making assumptions about the distribution of perturbations
        # about a mean x-direction wind!

        # del pos, ws_local
        # ^ seems like Numba might require this? (gave errors otherwise)
        # now, using numba v0.49.1 this raises error:
        #   `CompilerError: Illegal IR, del found at: del pos`
        # if trying to use sufficiently older versions of numba, may need to put it back
        # TODO: maybe change numba version spec in setup.cfg to >= 0.49
