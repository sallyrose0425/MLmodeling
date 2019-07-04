
## Necessary Packages:
(Python 3)
  - Pandas
  - [DEAP](http://deap.readthedocs.org/)
  - [rdKit](https://www.rdkit.org/docs/GettingStartedInPython.html)
  - skLearn

## Usage:


In practice we use [taskSpooler](http://vicerveza.homeunix.net/~viric/soft/ts/man_ts.html) and run multiple instances in parallel:
```
# !/bin/bash
tsp -S 16
tsp python 
tsp python   
...
```
