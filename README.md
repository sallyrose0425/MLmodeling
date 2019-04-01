# MLmodeling
## Necessary Packages:
(Python 3)
  - Pandas
  - DEAP
  - rdKit
  - skLearn

## Assumed File Structure:

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


## Usage:

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
