# blpd

*Lagrangian (stochastic) particle dispersion with Python+Numba to model bees finding floral scents*

[![CI workflow status](https://github.com/zmoon/blpd/actions/workflows/ci.yml/badge.svg)](https://github.com/zmoon/blpd/actions/workflows/ci.yml)
[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/zmoon/blpd/master.svg)](https://results.pre-commit.ci/latest/github/zmoon/blpd/master)
[![Run examples on Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/zmoon/blpd/HEAD?urlpath=lab/tree/examples)

[pdoc](https://pdoc.dev/) API documentation: <https://zmoon.github.io/blpd/blpd.html>
## Installing

Latest:
```
pip install https://github.com/zmoon/blpd/archive/master.zip
```

### For development

1. Clone the repo
   ```
   git clone https://github.com/zmoon/blpd
   ```
2. Register the package as an editable install, by navigating into the the repo and executing
   ```
   pip install -e .
   ```

#### Notes

:exclamation: An updated version of `pip` (>= 19.0) should be used for the install to ensure that the build backend specified in `pyproject.toml` will be read and used (installed if necessary). Otherwise, the `blpd` package version may not be correctly read into the package metadata.


## References

* K. Pratt's thesis: <https://etda.libraries.psu.edu/catalog/14063>. The version described there included a simpler treatment of the in-canopy statistics. [Marcelo Chamecki](http://people.atmos.ucla.edu/mchamecki/index.htm) added the Massman & Weil (1999) model to enhance the canopy treatment. The LPD code (module `blpd.lpd`) is based on a version of Pratt's model written in Matlab.
* Massman & Weil (1999): <https://doi.org/10.1023/A:1001810204560>
* Fuentes et al. (2016): <https://doi.org/10.1016/j.atmosenv.2016.07.002>
