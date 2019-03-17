
"""
cd DataSets

from importlib import reload
import os
import ukyScore
import numpy as np
dataset = 'dekois'
target_id = 'ADRB2'
prefix = os.getcwd() + '/' + dataset + '/'
activeFile = prefix + 'ligands/' + target_id + '.sdf.gz'
decoyFile = prefix + 'decoys/' + target_id + '_Celling-v1.12_decoyset.sdf.gz'

reload(ukyScore)

data = ukyScore.data_set(activeFile, decoyFile, balanceTol=0.01)
"""
from copy import copy
valid = False
while not valid:
    split = data.randSplit()
    valid = data.validSplit(split)
labels = data.labels.values
minScore = data.computeScore(split)

validActiveIndices = np.where((split == 0) & (labels == 1))[0]
validDecoyIndices = np.where((split == 0) & (labels == 0))[0]
trainActiveIndices = np.where((split == 1) & (labels == 1))[0]
trainDecoyIndices = np.where((split == 1) & (labels == 0))[0]

flipBits = (np.random.choice(validActiveIndices),
            np.random.choice(validDecoyIndices),
            np.random.choice(trainActiveIndices),
            np.random.choice(trainDecoyIndices)
            )
splitMut = copy(split)
for bit in list(flipBits):
    splitMut[bit] = 1 - split[bit]


data.computeScore(splitMut)

minScore = 2.0
