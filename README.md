# Iterative Proportional Fitting on arbitrary partitions
A framework and example computations for iterative proportional fitting, generalized to partitions of the table.

The framework for iterative proportional fitting is contained in `src/ipf.py`.

This repository contains functionality to
estimate the [2025 internal migration table from Statistics Austria](https://statcube.at/statistik.at/ext/statcube/openinfopage?tableId=defaulttable_debevwan011).
This estimate can be exported as CSV by running `python export.py`
from the `exports` directory.
The result are CSV files in the working directory (one for each estimate),
of which `2025_full_tiled_with_vienna.csv` contains the highest-quality result.
Alternatively,
```python
import calculations.ipf_2025_full_tiled_with_vienna as ipf_estimate
import data.data_preprocessing as data_pre

estimate = ipf_estimate.ipf_estimate()
names_places = data_pre.names_places
names_age = data_pre.names_age
```
will store the table estimate in the variable `estimate`, as a 3D NumPy array
with dimensions origin, destination and age indexed in the order of `names_places`
and `names_age`.
