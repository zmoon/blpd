"""
Bee flight model

Based on the Lévy flight random walk model as implemented in
- J.D. Fuentes et al. / Atmospheric Environment (2016)
"""
import numpy as np
from scipy.stats import rv_continuous


def get_step_length(*, l_0=1.0, mu=2.0):
    """
    Draw a step length `l` from the distribution
      p(l) = (l/l_0)^-mu

    Parameters
    ----------
    l_0 : float
        the minimum step size
    mu : distribution shape parameter
        =3 => Brownian motion
        =2 => "super diffusive Lévy walk"

    """
    if mu <= 1 or mu > 3:
        raise ValueError(f"`mu` should be in (1, 3] but is {mu!r}")

    q = np.random.rand()  # draw from [0, 1] (uniform)
    l = l_0 * (1 - q) ** (1 / (1 - mu))  # noqa: E741
    # ^ note 1-mu not mu, this comes from the inverse of the CDF

    return l


# scipy.stats.powerlaw doesn't let its param `a` be negative, so can't use it


class step_length_dist_gen(rv_continuous):
    """Step length distribution class
    using the scipy.stats machinery
    """

    def _pdf(self, x, l_0, mu):
        c = -l_0 / (1 - mu)
        p = (x / l_0) ** (-mu)
        return p / c

    def _cdf(self, x, l_0, mu):
        # https://www.wolframalpha.com/input/?i=anti-derivative+%28l%2Fc%29%5E-mu+dl
        # https://www.wolframalpha.com/input/?i=anti-derivative+%28l%2Fc%29%5E-mu+dl%2C+from+l%3Dc+to+x
        # return l_0**mu * x**(1-mu) / (1-mu)
        # return x * (x/l_0)**(-mu) / (1-mu)

        c = -l_0 / (1 - mu)
        # if x >= l_0:
        #     return x * (x/l_0)**(-mu) / (1-mu) + c
        # else:
        #     return 0
        # F = x * (x/l_0)**(-mu) / (1-mu) + c
        F = l_0 ** mu * x ** (1 - mu) / (1 - mu) + c
        # F = l_0**mu * x**(1-mu) / (1-mu) + 0
        # F[x < l_0] = 0
        return F / c

        # return (l_0 - x*(x/l_0)**(-mu)) / (mu-1)

    # these forms seem to only work in a few special cases.
    # if we don't provide _ppf it will numerically estimate it from _cdf
    def _ppf(self, q, l_0, mu):
        # return (q*(mu-1) - l_0 + (1/l_0)**-mu)**(1/(1-mu))
        # return (q*(mu-1) - l_0 + l_0**mu)**(1/(1-mu))

        # x = (q*(mu-1) - l_0 + l_0**mu)**(1/(1-mu))
        # # x[x < l_0] = np.nan
        # return x

        c = -l_0 / (1 - mu)

        # this won't work when `q*(1-mu) + l_0` is negative
        x = ((q * c * (1 - mu) + l_0) * l_0 ** -mu) ** (1.0 / (1.0 - mu))

        return x

    #     # return ( (q-1)*(1-mu) / l_0**mu )**(1/(1-mu))

    def _argcheck(self, *args):
        # below is the default
        # --------------------
        # cond = 1
        # print(args)
        # for arg in args:
        #     cond = np.logical_and(cond, (np.asarray(arg) > 0))
        # return cond

        l_0, mu = args  # unpack

        # l_0 (minimum step length) must be positive
        cond_l_0 = np.asarray(l_0) > 0

        # shape parameter mu is supposed to be in range [1, 3]
        # not sure why it can't just be positive tho...
        arr_mu = np.asarray(mu)
        cond_mu = arr_mu > 1 and arr_mu <= 3

        # return np.logical_and(cond_l_0, cond_mu, cond_mu_2)

        arr_l_0 = np.asarray(l_0)
        cond_mu_2 = 1 * (1 - arr_mu) + arr_l_0 >= 0

        cond = 1
        for cond_ in [cond_l_0, cond_mu, cond_mu_2]:
            cond = np.logical_and(cond, cond_)

        return cond

    def _get_support(self, *args, **kwargs):
        # need to return support as 2-tuple
        return args[0], np.inf


# step_length_dist = step_length_dist_gen(a=1.0, name="step_length_dist")
step_length_dist = step_length_dist_gen(name="step_length_dist")


# scaling with `scale` not working for pdf/cdf
# need to specify them differently I guess
class step_length_dist2_gen(rv_continuous):
    """Step length distribution class
    using the scipy.stats machinery
    """

    # instead of including l_0 as a param,
    # just let scipy.stats do that via the `scale` param

    def _pdf(self, x, mu):
        c = -1.0 / (1 - mu)  # normalization const
        # return x**-mu
        return (x ** -mu) / c

    def _cdf(self, x, mu):
        c = -1.0 / (1 - mu)
        # F = x**(1-mu) / (1-mu) + c
        F = (x ** (1 - mu) - 1) / (1 - mu)
        return F / c

    def _ppf(self, q, mu):
        # this won't work when `q*(1-mu) + 1` is negative
        # x = (q*(1-mu) + 1) ** (1.0/(1.0-mu))
        c = -1.0 / (1 - mu)
        x = (q * c * (1 - mu) + 1) ** (1.0 / (1.0 - mu))
        return x

    def _argcheck(self, *args, **kwargs):

        mu = args  # unpack

        # shape parameter mu is supposed to be in range [1, 3]
        # not sure why it can't just be positive tho...
        arr_mu = np.asarray(mu)
        cond_mu = arr_mu > 1 and arr_mu <= 3

        # return np.logical_and(cond_l_0, cond_mu)
        return cond_mu

    # def _get_support(self, *args, **kwargs):
    #     # need to return support as 2-tuple
    #     return args[0], np.inf


step_length_dist2 = step_length_dist2_gen(a=1.0, name="step_length_dist2")


if __name__ == "__main__":

    import matplotlib.pyplot as plt

    # test the step length dist
    # all 3 agree for mu=2.0, l_0=1.0...
    mu = 2.2
    l_0 = 5
    # assert 1-mu + l_0 >= 0
    n_steps = int(5e5)
    x_stop = 1000  # in the plots
    steps = [get_step_length(l_0=l_0, mu=mu) for _ in range(n_steps)]
    fig, [ax, ax2] = plt.subplots(1, 2)
    bins = np.arange(0, x_stop, 0.1)
    ax.hist(
        steps,
        bins=bins,
        density=True,
        histtype="stepfilled",
        alpha=0.7,
        label="orig drawing method",
    )

    # compare to the scipy.stats version
    # x = np.linspace(l_0, x_stop, 200)
    x = np.linspace(0, 100, 1000)  # for the line (analytical) plots
    dist = step_length_dist(mu=mu, l_0=l_0)
    dist2 = step_length_dist2(mu=mu, scale=l_0, loc=0)
    steps2 = dist.rvs(n_steps)
    steps3 = dist2.rvs(n_steps)
    ax.hist(
        steps2,
        bins=bins,
        density=True,
        histtype="stepfilled",
        alpha=0.25,
        color="orange",
        label="using .rvs",
    )
    ax.hist(
        steps3,
        bins=bins,
        density=True,
        histtype="stepfilled",
        alpha=0.25,
        color="magenta",
        label="using .rvs 2",
    )
    ax.plot(x, dist.pdf(x), label="pdf - analytical", color="orange")
    ax.plot(x, dist2.pdf(x), label="pdf - analytical 2", color="magenta")

    # ax2.plot()
    # bins = np.linspace(0, x_stop, 200)  # like integrating from 0->inf
    # ax2.hist(steps2[steps2 > l_0],
    ax2.hist(
        steps2,
        bins=bins,
        density=True,
        cumulative=True,
        histtype="stepfilled",
        alpha=0.25,
        color="orange",
        label="using .rvs",
    )
    ax2.hist(
        steps3,
        bins=bins,
        density=True,
        cumulative=True,
        histtype="stepfilled",
        alpha=0.25,
        color="magenta",
        label="using .rvs 2",
    )
    ax2.plot(x, dist.cdf(x), lw=2, label="cdf - analytical", color="orange")
    ax2.plot(x, dist2.cdf(x), ":", label="cdf - analytical 2", color="magenta")

    ax.set_xlim((0, l_0 * 10))
    ax.legend()
    ax2.set_xlim((0, l_0 * 10))
    ax2.legend()

    fig.tight_layout()

    plt.show()
