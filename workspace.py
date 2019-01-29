

import pandas as pd


###############################################################################
# Ligands
from ligand import *

smilesStrings = pd.read_csv('Smiles.csv',
                            delimiter=',').values.flatten().tolist()
smilesStrings = [s.replace('\n','') for s in smilesStrings if type(s)==str]
smilesSeries = pd.Series(smilesStrings, index = smilesStrings)
ligands = smilesSeries.apply(ligand)
badSmiles = ligands.apply(lambda x: x.mol is None)
ligands = ligands.drop(index = L[badSmiles].index)

###############################################################################
# Fingerprints

F = ligands.apply(lambda x: x.fingerprint())
fingerprints = pd.DataFrame.from_records(F.tolist())

###############################################################################
# Similarity matrix

#Jaccard
from sklearn.metrics.pairwise import pairwise_distances
fingerprintSimilarity = pairwise_distances(fingerprints,
                                           metric='jaccard',
                                           n_jobs=-1)

#Tanimoto
from tanimoto import *
feat = similarityFeatures()
feat.fit(fingerprints.values)
fingerprintSimilarity = feat.transform(fingerprints.values)
