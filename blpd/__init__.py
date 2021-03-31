"""
Lagrangian (stochastic) particle dispersion model to model bees encountering floral scents
"""
# Define the namespace for the module `blpd`
# Importing `Model` also adds `lpd` and `main`
from . import bees  # noqa: F401 unused import
from . import plot  # noqa: F401 unused import
from .main import Model as model  # noqa: F401 unused import
