# MLmodeling

## Assumed File Structure:


These scripts assume they are in a folder containing folders named 'MUV', 'dekois', and 'DUDE'.
They further assume the following:

### MUV
For each target 'target_id', actives are stored in a file 'target_id_actives.sdf.gz',
decoys in 'target_id_decoys.sdf.gz'

### dekois
Actives are stored in a folder named 'ligands', and decoys in a folder named 'decoys'.
For each target 'target_id', actives are stored in a file 'target_id.sdf.gz',
decoys in 'target_id__Celling-v1.12_decoyset.sdf.gz'

### DUDE
Active and decoys are stored in files named 'target_id'.
Within each target's folder, the data is stored in '/actives_final.sdf.gz' and '/decoys_final.sdf.gz'.


## Usage:



### 'dataPrep.py'
The preliminary computations are performed on a per-dataset basis by the file 'dataPrep.py'. 
By default, running the script outputs pickle files (into the dataset folder) for each target data set:

  - the similarity matrix 'target_id_distances.pkl'
  
  - a dataframe with the 2048 bit fingerprint of each ligand (active and decoy)
    and a binary label 'target_id_unsplitDataFrame.pkl'
    
  - a dataframe containing the VE score, training ratio, and validation equity
    for a sampling of splits 'target_id_samples.pkl'
    
If the distance matrix has already been computed, the script loads it and proceedes to sampling.

Example:
```
  $ python dataPrep.py dekois
  Current target: 17betaHSD1 (1 of 81)
  Computing decoy fingerprints...
  Computing active fingerprints...
  Saved: /DataSets/dekois/17betaHSD1_unsplitDataFrame.pkl
  Computing distance matrix...
  Saved: /DataSets/dekois/17betaHSD1_distances.pkl
  Sampling...
  Saved: /DataSets/dekois/17betaHSD1_samples.pkl
  Current target: A2A (2 of 81)
  ...
```


'bias_analysis.py'
Some analysis tasks are automated in order to reproduce results presented in the paper.
Running the script for a data set:

  - prints statistics for the score means and standard deviations over all targets in the data set
  
  - plots the aggregated standardized scores over all targets
  
Example:
  $ python bias_analysis.py dekois
  
  
#######################

'modelCorrelation.py'
We train a random forest regressor for each data split in the sample (produced by dataPrep.py) and record the AUC score.
We standardize the AUC and VE scores for each target (across all of the splits for that target), and aggregate them.

Running the script for a data set:

  - prints statistics for the Pearson correlation coefficient between AUC and VE scores
    over all targets in the data set.
  
  - plots the (aggregated standardized) AUC scores against the VE scores. 
  
Example:
  $ python modelCorrelation.py dekois
  
  
#######################

'genSplitOpt.py'
We use the DEAP evolutionary algorithm package to produce a minimum VE score split for each target in the data set.
Running the script for a data set:

  - 
  
Example:
  $ python genSplitOpt.py dekois
  
  

#######################
The dataBias module:
#######################


The dataPrep.py script creates a data_set class instance from the data of each target.

The class variables include:
  - self.distanceMatrix 
    The similarity matrix as computed by sklearn metrics 'jaccard'
    
  - self.labels
    The binary labels for the data set
    
  - self.size
    The number of data points for the data set
    
  - self.bias_samples
    The list of VE scores computed from the sampled splits so far (initially empty)
    
  - self.times
    The list of computation times used to sample each split and evaluate its VE score
    
  - self.comp_time
  The total computation time used so far for sampling and evaluation
  
The methods include:
  - computeAVEbias(self, split)
  - randSplit(self, q)
  - sample(self, duration, q)


