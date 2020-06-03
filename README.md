# bee-lpdm

*Lagrangian (stochastic) particle dispersion with Python+Numba to model bees finding floral scents*

This project is based on a model written in Matlab by K. Pratt for his [MS thesis](https://etda.libraries.psu.edu/catalog/14063).

[Installing](#installing) | [References](#references)


## Installing

Until releases are made (on [PyPI](https://pypi.org/) and/or GitHub), the recommended method of installing is 
```
pip install git+https://github.com/zmoon92/bee-lpdm#egg=blpdm
```
or to ensure the plotting dependencies are met:
```
pip install git+https://github.com/zmoon92/bee-lpdm#egg=blpdm[plots]
```
These commands will install the current state of the code in the default branch (master).


### For development

If you plan to commit changes:
1. Clone the repo
   ```
   git clone https://github.com/zmoon92/bee-lpdm
   ```
2. Register the package as an editable install, by navigating into the the repo and executing
   ```
   pip install -e .
   ```

If you don't plan to commit changes, you can skip the manual `git clone` and just use:
```
pip install -e git+https://github.com/zmoon92/bee-lpdm#egg=blpdm
```

#### Notes

These are the dependencies to run the model:
```
numba>=0.44
numpy  # this is also a numba dependency
```
and for the plotting/analysis routines (extras group `plots`):
```
matplotlib
scipy
```

:exclamation: An updated version of `pip` (>= 19.0) should be used for the install to ensure that the build backend specified in `pyproject.toml` will be read and used (installed if necessary). Otherwise, the `blpdm` package version may not be correctly read into the package metadata.


## References

* K. Pratt's thesis: <https://etda.libraries.psu.edu/catalog/14063>. The version described there included a simpler treatment of the in-canopy statistics. [Marcelo Chamecki](http://people.atmos.ucla.edu/mchamecki/index.htm) added the Massman & Weil (1999) model to enhance the canopy treatment. 
* Massman & Weil (1999): <https://doi.org/10.1023/A:1001810204560>
* Fuentes et al. (2016): <https://doi.org/10.1016/j.atmosenv.2016.07.002>
