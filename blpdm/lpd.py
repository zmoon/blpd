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
        #p_fd = _calc_fd_params_above_canopy(pos, p)

        U1, dU1dx3, \
        dtau11dx3, dtau22dx3, dtau33dx3, dtau13dx3, \
        lam11, lam22, lam33, lam13, \
        eps \
            = _calc_fd_params_above_canopy(pos, p)
        # return (umean, dumeandz,
        #     tau11, tau22, tau33, tau13,
        #     dtau11dz, dtau22dz, dtau33dz, dtau13dz,
        #     lam11, lam22, lam33, lam13,
        #     epsilon)
    else:
        #p_fd = _calc_fd_params_in_canopy(pos, p)
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
    du1 = \
        (-C0*eps/2 * (lam11*(u1-U1) + lam13*u3) + dU1dx3*u3 + dtau13dx3/2 ) * dt \
      + (dtau11dx3*(lam11*(u1-U1) + lam13*u3) \
        + dtau13dx3*(lam13*(u1-U1) + lam33*u3))*(u3/2*dt) \
      + sqrt_C0eps * dW1


    # x-2 component
    randn = np.random.standard_normal()
    dW2 = sqrt_dt * randn
    du2 = \
        (-C0*eps/2 * lam22*u2 + dtau22dx3*lam22*u2 * u3/2) * dt \
      + sqrt_C0eps * dW2


    # x-3 component
    randn = np.random.standard_normal()
    dW3 = sqrt_dt * randn
    du3 = \
        (-C0*eps/2 * (lam13*(u1-U1) + lam33*u3) + dtau33dx3/2) * dt \
      + (dtau13dx3 * (lam11*(u1-U1) + lam13*u3) \
        + dtau33dx3 * (lam13*(u1-U1) + lam33*u3)) * (u3/2*dt) \
      + sqrt_C0eps * dW3


    # if np.any(np.r_[du1, du2, du3] > 50):
    if np.any(np.abs(np.array([du1, du2, du3])) > 50):
    #     # msg = f'one wind speed is too high: {du1}, {du2}, {du3}'
        # msg = 'one wind speed is too high: ' + str(du1) + str(du2) + str(du3)  # numba nopython can't do str formatting
    #     warnings.warn(msg)
        # print(msg)
        print("one ws' is too high:", du1, du2, du3)

        # debugging info
        print('umean, dumeandz, epsilon', U1, dU1dx3, eps)
        if z >= h_c:
            print('  z >= h_c')
        else:
            print('  z < h_c, z =', z)

        # import pdb; pdb.set_trace()

        # for d in [du1, du2, du3]:  # this method doesn't seem to work in numba
        #     if np.abs(d) > 50:
        #         # d = 50.
        #         # d[:] = 50.
        # print('resetting to 50')

        # print('resetting all to 0')
        # du1 = 0.
        # du2 = 0.
        # du3 = 0.

        print('resetting new ws to 0')
        du1 = 0.
        du2 = 0.
        du3 = 0.
        u1 = 0.
        u2 = 0.
        u3 = 0.

        # print('  now:', du1, du2, du3)  # to check if the resetting is working or not

    #     # print(locals())
    #     for vn, val in locals().items():
    #         # print(f'{vn}:\t{val:.4g}')
    #         # print(f'{vn}:\t{val}')
    #         print(vn, ':\t', val)
    #     print('\n\n')
    #     sys.exit()

    # r = {
    #     'dxdt': u1 + du1,
    #     'dydt': u2 + du2,
    #     'dzdt': u3 + du3
    # }

    # return r

    # return u1 + du1, u2 + du2, u3 + du3
    return [u1 + du1, u2 + du2, u3 + du3]



@njit
def _integrate_particle_one_timestep(pos, ws_local, p):
    """
    Take the state for one particle and integrate
    """

    # pos = state['x'], state['y'], state['z']

    dt = p['dt']

    res = calc_tends(pos, ws_local, p)
    # ^ currently returns tuple like pos
    # res = [ws+dws for ws, dws in zip(ws_local, res)]

    # contact with ground?
    # * original code noted that this needs work
    # this seems to just give them an elastic collision instead of a deposition
    z = pos[2]  # before integration
    znew = z + res[2]*dt  # res[2] is w velocity component
    if znew <= 0.1:
        # z = 0.1
        # res[2] = -res[2]  # w -> -w
        # res = (res[0], res[1], -res[2])
        res = [res[0], res[1], -res[2]]
        # res = (res[0], res[1], 0)

        # newpos = (pos[0] + res[0]*dt, pos[1] + res[1]*dt, 0.1)
        newpos = [pos[0] + res[0]*dt, pos[1] + res[1]*dt, 0.1]

    elif znew >= 800.:  # reflect at z_i too
        res = [res[0], res[1], -res[2]]
        newpos = [pos[0] + res[0]*dt, pos[1] + res[1]*dt, 800.]

    else:
        # newpos = tuple([p + dpdt*dt for (p, dpdt) in zip(pos, res)])  # numba doesn't like tuple of list of float64
        newpos = [p + dpdt*dt for (p, dpdt) in zip(pos, res)]
    # this needs to be outside the calc_tends fn


    # the pos tendencies are the (new) velocity components
    new_ws_local = res

    # newpos = tuple(p + dpdt*dt for p, dpdt in zip(pos, res))  # generator expression version

    return newpos, new_ws_local



# @njit
@njit(parallel=True)
def integrate_particles_one_timestep(state, p):
    """
    Integrate all particles one time step

    p must be a numba typed dict (`numba.typed.Dict`)
    in order for @njit to work

    and state too, in order to use @njit on this fn
    """

    Np_k = int(state['Np_k'][0])
    # Np_k = np.int64(state['Np_k'])
    xp = state['xp']
    yp = state['yp']
    zp = state['zp']
    up = state['up']
    vp = state['vp']
    wp = state['wp']

    # for i in range(Np_k):
    for i in prange(Np_k):  # pylint: disable=not-an-iterable

        # state for one particle
        # pos and local wind speed

        # pos = xp[i], yp[i], zp[i]  # position of particle i at current time step
        pos = [xp[i], yp[i], zp[i]]  # position of particle i at current time step
        ws_local = [up[i], vp[i], wp[i]]  # local wind speed for particle

        # newpos = _integrate_particle_one_timestep(pos, p)
        # pos = _integrate_particle_one_timestep(pos, ws_local, p)
        pos, ws_local = _integrate_particle_one_timestep(pos, ws_local, p)

        # ip1 = i + 1

        xp[i] = pos[0]
        yp[i] = pos[1]
        zp[i] = pos[2]

        up[i] = ws_local[0]
        vp[i] = ws_local[1]
        wp[i] = ws_local[2]
        # ^ it seems like the wind speed components were not being stored in this way in the orig code
        #   like it was always reset to 0 in the next time step

        # we are not time-integrating the wind speed.
        # we are making assumptions about the distribution of perturbations
        # about a mean x-direction wind!

        # del pos, ws_local
        # ^ seems like Numba might require this? (gave errors otherwise)
        # now, using numba v0.49.1 this raises error:
        #   `CompilerError: Illegal IR, del found at: del pos`
        # if trying to use sufficiently older versions of numba, may need to put it back
        # TODO: maybe change numba version spec in setup.cfg to >= 0.49
