
# Example scripts

For now, these add `blpd` to path manually (not requiring an install). This shouldn't mess anything up if `blpd` is actually installed (see [../README.md](../README.md)).

Hyphens are not allowed in Python module names, but these scripts are not (at this time) intended to be imported as modules by anything. (That is, `import {module-with-hyphens}` won't work, but note that `importlib.import_module('{module-with-hyphens}')` will...)
