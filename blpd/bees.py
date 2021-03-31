"""
Bee flight model

Based on the Lévy flight random walk model as implemented in
- J.D. Fuentes et al. / Atmospheric Environment (2016)
"""
import math

import numpy as np
from scipy import stats


__all__ = ("flight",)


PI = math.pi


def get_step_length(*, l_0=1.0, mu=2.0, q=None):
    r"""
    Draw a step length `l` from the distribution

    .. math::
       p(l) = (l/l_0)^{-\mu}

    Parameters
    ----------
    l_0 : float
        The minimum step size
    mu : float
        Distribution shape parameter

        3 => Brownian motion

        2 => "super diffusive Lévy walk"
    q : float, array, optional
        Random number in [0, 1). By default, we draw from uniform.
    """
    if mu <= 1 or mu > 3:
        raise ValueError(f"`mu` should be in (1, 3] but is {mu!r}")

    if q is None:
        q = np.random.rand()  # draw from [0, 1) (uniform)

    l = l_0 * (1 - q) ** (1 / (1 - mu))  # noqa: E741
    # ^ note 1-mu not mu, which comes from the inverse of the CDF

    return l


def flight(
    n,
    *,
    x0=(0, 0),
    l_0=1.0,
    mu=2.0,
    l_max=None,
    heading0=PI / 2,
    heading_model="uniform",
    heading_model_kwargs=None,
):
    """
    Parameters
    ----------
    n : int
        Number of steps taken in the simulated flight.
        The final trajectory will have `n`+1 points.
    x0 : array_like
        xy-coordinates of the starting location.
    l_0, mu : float
        Parameters of the step length power law distribution.
    l_max : float, optional
        Used to clip the step size if provided.
    heading0 : float
        Initial heading of the flight.
        In standard polar coordinates, so pi/2 is north.
    heading_model : {'uniform', 'truncnorm'}
        Relative heading model.
    heading_model_kwargs : dict
        Used in the relative heading model.

        For truncnorm: `'std'`.
    """
    assert np.asarray(x0).size == 2, "xy-coordinates"

    if heading_model_kwargs is None:
        heading_model_kwargs = {}

    # Draw steps
    q = np.random.rand(n)
    steps = get_step_length(l_0=l_0, mu=mu, q=q)

    # Clip steps if desired
    if l_max is not None:
        np.clip(steps, None, l_max, out=steps)

    # Draw (relative) headings
    if heading_model == "uniform":
        headings = np.random.uniform(-PI, PI, size=n - 1)
    elif heading_model == "truncnorm":
        std = heading_model_kwargs.get("std", 1.5)
        mean = heading_model_kwargs.get("mean", 0)
        clip_a, clip_b = -PI, PI
        a, b = (clip_a - mean) / std, (clip_b - mean) / std
        dist = stats.truncnorm(a, b, scale=PI / b)
        # print(f"99.99th: {dist.ppf(0.9999)}")
        headings = dist.rvs(n - 1)
    else:
        raise NotImplementedError

    # Convert heading -> direction
    # (Heading is relative to current direction)
    angles = np.full((n,), heading0)
    for i, heading in enumerate(headings):
        angles[i + 1] = angles[i] + heading

    # Note: the above loop can be replaced by
    # `angles[1:] = np.cumsum(headings) + heading0`
    # (similarly for below loop)

    # Convert angles to [0, 2pi) range
    min_angle = angles.min()
    # Find the multiple of 2pi needed to make all values positive
    n_ = -math.floor(angles.min() / (2 * PI)) if min_angle < 0 else 0
    angles_mod = np.mod(angles + n_ * 2 * PI, 2 * PI)
    assert np.allclose(np.cos(angles), np.cos(angles_mod))
    angles = angles_mod

    # # Investigate the angles
    # import matplotlib.pyplot as plt
    # fig, [ax1, ax2, ax3] = plt.subplots(1, 3, figsize=(9, 3.5))
    # ax1.hist(headings); ax1.set_title("headings")
    # ax2.hist(angles); ax2.set_title("angles")
    # ax3.hist(angles_mod); ax3.set_title("angles % 2pi")
    # fig.tight_layout()

    # Walk
    x = np.full((n + 1,), x0[0], dtype=float)
    y = np.full((n + 1,), x0[1], dtype=float)
    for i, (step, angle) in enumerate(zip(steps, angles)):
        dx = step * math.cos(angle)
        dy = step * math.sin(angle)
        x[i + 1] = x[i] + dx
        y[i + 1] = y[i] + dy

    return x, y


# Note: `scipy.stats.powerlaw` doesn't let its param `a` be negative, so can't use it


class step_length_dist_gen(stats.rv_continuous):
    """Step length distribution class using the `scipy.stats` machinery."""

    def _pdf(self, x, l_0, mu):
        c = -l_0 / (1 - mu)
        p = (x / l_0) ** (-mu)
        return p / c

    def _cdf(self, x, l_0, mu):
        # https://www.wolframalpha.com/input/?i=anti-derivative+%28l%2Fc%29%5E-mu+dl
        # https://www.wolframalpha.com/input/?i=anti-derivative+%28l%2Fc%29%5E-mu+dl%2C+from+l%3Dc+to+x
        c = -l_0 / (1 - mu)
        # F = x * (x/l_0)**(-mu) / (1-mu) + c
        F = l_0 ** mu * x ** (1 - mu) / (1 - mu) + c
        # F[x < l_0] = 0
        return F / c

    def _ppf(self, q, l_0, mu):
        c = -l_0 / (1 - mu)
        x = ((q * c * (1 - mu) + l_0) * l_0 ** -mu) ** (1.0 / (1.0 - mu))

        return x

    def _argcheck(self, *args):
        l_0, mu = args  # unpack

        # Minimum step length `l_0` must be positive
        cond_l_0 = np.asarray(l_0) > 0

        # Shape parameter `mu` is supposed to be in range [1, 3]
        # not sure why it can't just be positive tho...
        arr_mu = np.asarray(mu)
        cond_mu = arr_mu > 1 and arr_mu <= 3

        arr_l_0 = np.asarray(l_0)
        cond_mu_2 = 1 * (1 - arr_mu) + arr_l_0 >= 0

        cond = 1
        for cond_ in [cond_l_0, cond_mu, cond_mu_2]:
            cond = np.logical_and(cond, cond_)

        return cond

    def _get_support(self, *args, **kwargs):
        return args[0], np.inf


step_length_dist = step_length_dist_gen(name="step_length_dist")


# scaling with `scale` not working for pdf/cdf
# need to specify them differently I guess
class step_length_dist2_gen(stats.rv_continuous):
    """Step length distribution class using the `scipy.stats` machinery."""

    # Instead of including l_0 as a param,
    # we just let scipy.stats do that via the `scale` param

    def _pdf(self, x, mu):
        c = -1.0 / (1 - mu)  # normalization const
        return (x ** -mu) / c

    def _cdf(self, x, mu):
        c = -1.0 / (1 - mu)
        F = (x ** (1 - mu) - 1) / (1 - mu)
        return F / c

    def _ppf(self, q, mu):
        c = -1.0 / (1 - mu)
        x = (q * c * (1 - mu) + 1) ** (1.0 / (1.0 - mu))
        return x

    def _argcheck(self, *args, **kwargs):
        mu = args  # unpack

        arr_mu = np.asarray(mu)
        cond_mu = arr_mu > 1 and arr_mu <= 3

        return cond_mu


step_length_dist2 = step_length_dist2_gen(a=1.0, name="step_length_dist2")


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    plt.close("all")

    # Set dist parameters
    # Note that all 3 agree for mu=2.0, l_0=1.0 (the values used in the paper)
    # But the analytical PDF/CDFs do not agree with the hist for mu < 2
    # Since the rvs results agree, this means the `_cdf` and `_pdf` methods are still off
    # though the `_ppf`s are fine.
    mu = 2.0
    l_0 = 1.0
    assert 1 - mu + l_0 >= 0
    n_steps = int(5e5)
    x_stop = 1000  # rightmost bin edge
    x = np.linspace(0, 100, 1000)  # for the line (analytical) plots
    bins = np.arange(0, x_stop, 0.1)  # for the histograms

    # Draw using original method
    steps = [get_step_length(l_0=l_0, mu=mu) for _ in range(n_steps)]

    fig, [ax, ax2] = plt.subplots(1, 2, figsize=(8, 4))

    # Histogram of steps using original method
    ax.hist(
        steps,
        bins=bins,
        density=True,
        histtype="stepfilled",
        alpha=0.7,
        label="orig drawing method",
    )

    # Create corresponding distribution objects
    dist = step_length_dist(mu=mu, l_0=l_0)
    dist2 = step_length_dist2(mu=mu, scale=l_0, loc=0)
    steps2 = dist.rvs(n_steps)
    steps3 = dist2.rvs(n_steps)

    # PDF
    ax.hist(
        steps2,
        bins=bins,
        density=True,
        histtype="stepfilled",
        alpha=0.25,
        color="orange",
        label="using dist.rvs",
    )
    ax.hist(
        steps3,
        bins=bins,
        density=True,
        histtype="stepfilled",
        alpha=0.25,
        color="magenta",
        label="using dist2.rvs",
    )
    ax.plot(x, dist.pdf(x), lw=3, label="pdf - analytical", color="orange")
    ax.plot(x, dist2.pdf(x), ":", lw=1, label="pdf - analytical 2", color="magenta")

    # CDF
    ax2.hist(
        steps,
        bins=bins,
        density=True,
        cumulative=True,
        histtype="stepfilled",
        alpha=0.25,
        label="orig drawing method",
    )
    ax2.hist(
        steps2,
        bins=bins,
        density=True,
        cumulative=True,
        histtype="stepfilled",
        alpha=0.25,
        color="orange",
        label="using dist.rvs",
    )
    ax2.hist(
        steps3,
        bins=bins,
        density=True,
        cumulative=True,
        histtype="stepfilled",
        alpha=0.25,
        color="magenta",
        label="using dist2.rvs",
    )
    ax2.plot(x, dist.cdf(x), lw=3, label="cdf - analytical", color="orange")
    ax2.plot(x, dist2.cdf(x), ":", lw=1, label="cdf - analytical 2", color="magenta")

    ax.set_xlim((0, l_0 * 10))
    ax.legend()
    ax2.set_xlim((0, l_0 * 10))
    ax2.legend()

    fig.tight_layout()

    plt.show()
