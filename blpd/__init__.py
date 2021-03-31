"""
Lagrangian (stochastic) particle dispersion model to model bees encountering floral scents

The model class `blpd.main.Model` is included in this namespace for convenience.
>>> import blpd
>>> m = blpd.Model()
>>> m.run().plot()

Additional plots can be created using `blpd.plot`.
Some of these require a chem dataset.
>>> ds = m.to_xr()  # `xr.Dataset` of LPD results
>>> dsc = blpd.chem.calc_relative_levels_fixed_oxidants(ds)  # compute chem dataset
>>> blpd.plot.ws_hist_all(ds)  # plot wind speed histograms
>>> blpd.plot.conc_2d(dsc)  # plot 2-d chemical species levels

See [the examples](https://github.com/zmoon/blpd/tree/master/examples) for more info.
"""
# Define the namespace for the module `blpd`
# Importing `Model` also adds `lpd` and `main`
from . import bees  # noqa: F401 unused import
from . import plot  # noqa: F401 unused import
from .main import Model as model  # noqa: F401 unused import
