
# Necessary Packages:
(Python 3)
  - Pandas
  - DEAP
  - rdKit
  - skLearn

# Assumed File Structure:

These scripts assume they are in a folder (e.g., ```/DataSets```) containing folders named 'MUV', 'dekois', and 'DUDE'.
They further assume the following:

### MUV
For each target 'target_id', actives are stored in a file ```target_id_actives.sdf.gz```,
decoys in ```target_id_decoys.sdf.gz```

### dekois
Actives are stored in a folder named 'ligands', and decoys in a folder named 'decoys'.
For each target 'target_id', actives are stored in a file ```target_id.sdf.gz```,
decoys in ```target_id__Celling-v1.12_decoyset.sdf.gz```

### DUDE
Active and decoys are stored in folders named 'target_id'.
Within each target's folder, the data is stored in ```/actives_final.sdf.gz``` and ```/decoys_final.sdf.gz```.


# Usage:

### 'singleTargetProcess.py'
The preliminary computations are performed on a per-target basis by the script 'singleTargetProcess.py'. 
By default, running the script outputs pickle files (into the dataset folder) for each target in the data set:

  -```target_id_dataPackage.pkl```
  
  -```target_id_optRecord.pkl```
    
  -```target_id_perfStats.pkl```

Example:
```
python singleTargetProcess.py dekois CYP2A6
Creating data set CYP2A6
Gathering fingerprints
Computing distance matrix
Finishing initialization
CYP2A6 - Initializing optimizer ...
CYP2A6 - Beginning optimization...
CYP2A6 -- Generation 0 -- Time (sec): 48.86 -- Min score: 0.0049 -- Score parts: -0.14005602240896348, 0.1449491188880615
```
In practice we use taskSpooler and run multiple instances in parallel:
```
# !/bin/bash
tsp -S 16
tsp python singleTargetProcess.py dekois CYP2A6 
tsp python singleTargetProcess.py dekois PNP 
tsp python singleTargetProcess.py dekois ALR2 
tsp python singleTargetProcess.py dekois TP 
...
```
We create the run script via:
```
import os
from glob import glob


dataset = 'dekois'
prefix = os.getcwd() + '/' + dataset + '/'
files = glob(prefix + '*')
targets = []
sizes = []
for file in files:
    target_id = file.split('/')[-1].split('_')[0]
    targets.append(target_id)
targets = list(set(targets))
for target_id in targets:
    if dataset == 'dekois':
        activeFile = prefix + 'ligands/' + target_id + '.sdf.gz'
        decoyFile = prefix + 'decoys/' + target_id + '_Celling-v1.12_decoyset.sdf.gz'
    elif dataset == 'DUDE':
        activeFile = prefix + target_id + '/actives_final.sdf.gz'
        decoyFile = prefix + target_id + '/decoys_final.sdf.gz'
    elif dataset == 'MUV':
        activeFile = prefix + target_id + '_actives.sdf.gz'
        decoyFile = prefix + target_id + '_decoys.sdf.gz'
    try:
        size = os.path.getsize(activeFile)*os.path.getsize(decoyFile)
        sizes.append((size, target_id))
    except FileNotFoundError:
        pass
sizes.sort()

calls = ['# !/bin/bash\ntsp -S 16\n']

for pair in sizes:
    target_id = pair[1]
    call = f'tsp python singleTargetProcess.py {dataset} {target_id} \n'
    calls.append(call)

f = open('runScript', 'w+')
f.writelines(calls)
f.close()
```
