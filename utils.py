"""
Utilities for the model, plots, etc.
"""

import matplotlib.pyplot as plt
import numpy as np

# TODO: create fn to compare two sets of parameters and point out the changes


def get_open_fig_labels():
    return [plt.figure(num).get_label() for num in plt.get_fignums()]


def check_fig_num(label, n=0):
    """Create a fig num that is not already taken."""
    current_labels = get_open_fig_labels()
    if n == 0:
        labeln = label
    else:
        labeln = f"{label}_{n}"
    if labeln in current_labels:
        return check_fig_num(label, n=n + 1)
    else:
        return labeln


def to_sci_not(f):
    """Convert float f to scientific notation string.
    by using string formats 'e' and 'g'
    
    The output of this must be enclosed within `$`
    """
    s_e = f"{f:.4e}"
    s_dec, s_pow_ = s_e.split("e")
    s_dec = f"{float(s_dec):.4g}"
    pow_ = int(s_pow_)
    return f"{s_dec} \\times 10^{{ {pow_} }}"


def sec_to_str(total_seconds):
    """Choose best format to display the run time.
    
    note: can't use `strftime` with `timedelta`s 
    """
    hours, remainder = divmod(total_seconds, 60 * 60)
    minutes, seconds = divmod(remainder, 60)

    hours = int(hours)
    minutes = int(minutes)
    seconds = int(seconds)  # most likely wouldn't run for seconds fractions, only multiples

    h = "\u2095"  # unicode subscript letters
    m = "\u2098"
    s = "\u209B"
    sep = "\u2009"  # thin space

    if total_seconds <= 60:
        s = f"{seconds}{s}"
    elif total_seconds <= 60 * 60:
        s = f"{minutes}{m}{sep}{seconds}{s}"
    else:
        s = f"{hours}{h}{sep}{minutes}{m}{sep}{seconds}{s}"
    return s
    # TODO: probably a better way to do this, checking for zeros and not displaying those
    #       or programmatically (iteratively or recursively) build the necessary fmt string


def moving_average(a, n=3, axis=0):
    """
    from: https://stackoverflow.com/a/14314054
    for now only axis=0 works
    """
    if axis != 0:
        raise NotImplementedError
    ret = np.cumsum(a, axis=axis, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1 :] / n


def s_t_info(p):
    """Create string to display info about run time and time step."""
    t_tot = p["t_tot"]
    dt = p["dt"]
    N_t = p["N_t"]
    if N_t >= 1000:
        s_N_t = to_sci_not(N_t)
    else:
        s_N_t = str(N_t)
    s_t_tot = sec_to_str(t_tot)
    return f"$t = {s_t_tot}$ s, $\\delta t = {dt} s$, $N_t = {s_N_t}$"
