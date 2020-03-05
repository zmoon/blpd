"""
Lagrangian (stochastic) particle dispersion model to model bees encountering floral scents
"""
# Define the namespace for the module `bldpm`. 

from .main import Model as model  # also adds 'lpd' and 'main'

# don't try to import plots if its dependencies are not installed
try:
    from . import plots
except (ImportError, ModuleNotFoundError):
    pass  # 
