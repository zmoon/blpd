# bee-lpdm

This project is based on a model written in Matlab by K. Pratt for his [MS thesis](https://etda.libraries.psu.edu/catalog/14063).  


## Installing

### Using without installing

These are the dependencies to run the model:
```
numba>=0.44
numpy  # numba depends on this but putting anyway for clarity
```
and for the plotting/analysis routines:
```
matplotlib
scipy
```

If you have ensured that the dependencies are met (e.g., with `conda install {}`), the module `blpdm` can be imported without additional build/installation steps as long as Python can find it. 


### But if you want to really install (add to Python search path)

Download this repo and navigate to where `setup.py` is located. (The package is not on [pypi](https://pypi.org/) at this time.)

#### "develop" mode (don't add to `site-packages`)

ref: <https://setuptools.readthedocs.io/en/latest/setuptools.html#development-mode>

```
pip install -e .
```
or
```
python setup.py develop
```
These options create a `*.egg-link` in `site-packages` that points to the source code (so that you can make changes easily). 

To ensure that the (optional) dependencies for the plotting routines are met:
```
pip install -e .[plots]
```

#### non-"develop" (add to `site-packages`)

`pip install {}` (without `-e`)

or
```
python setup.py install
```

If permission errors, add `--user` flag. 

### Notes

If you have installed with `pip` (either option), or `python setup.py develop`, you can easily uninstall (`pip uninstall`, or `python setup.py develop --uninstall`). 

As an alternative to downloading or `git clone`-ing the repo, `pip` can download the repo for you (if you have `git` that is). For example:
```
pip install -e git+https://github.com/zmoon92/bee-lpdm#egg=blpdm
```

Only `pip` versions >= 10.0 will recognize build dependencies in `pyproject.toml` and update the requirements (if necessary) before invoking `setuptools`. The current consequence of this is that if you use an older build, the version will not be read properly into the package metadata if `setuptools` is too old to read and parse the version from the text file (instead giving it as `file-VERSION.txt`). 

Only `pip` versions >= 19.0 support specifying the build backend in `pyproject.toml`. 

`pip install -e` still requires `setup.py` at this time (v20.0.2). 

## References

* K. Pratt's thesis: <https://etda.libraries.psu.edu/catalog/14063>. The version described there included a simpler treatment of the in-canopy statistics. [Marcelo Chamecki](http://people.atmos.ucla.edu/mchamecki/index.htm) added the Massman & Weil (1999) model to enhance the canopy treatment. 
* Massman & Weil (1999): https://doi.org/10.1023/A:1001810204560

