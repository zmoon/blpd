"""
Lagrangian (stochastic) particle dispersion model to model bees encountering floral scents
"""
# Define the namespace for the module `blpd`.
# This also adds `lpd` and `main`
from .main import Model as model  # noqa: F401 unused import

# Don't try to import `plots` if its dependencies are not installed
try:
    from . import plots  # noqa: F401 unused import
except (ImportError, ModuleNotFoundError):
    pass
