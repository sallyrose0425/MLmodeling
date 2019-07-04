
## Necessary Packages:
(Python 3)
  - Pandas
  - [DEAP](http://deap.readthedocs.org/)
  - [rdKit](https://www.rdkit.org/docs/GettingStartedInPython.html)
  - skLearn

## Usage:
```python split_optimization_analysis.py target_decoys_file.sdf.gz target_actives_file.sdf.gz target_output_file.csv```

In practice we use [taskSpooler](http://vicerveza.homeunix.net/~viric/soft/ts/man_ts.html) and run multiple instances in parallel:
```
# !/bin/bash
tsp -S 16
tsp python split_optimization_analysis.py target1_decoys_file.sdf.gz target1_actives_file.sdf.gz target1_output_file.csv
tsp python split_optimization_analysis.py target2_decoys_file.sdf.gz target2_actives_file.sdf.gz target2_output_file.csv
...
```
