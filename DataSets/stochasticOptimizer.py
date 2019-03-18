
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

from time import time
from multiprocessing import Pool
import os
import ukyScore

dataset = 'dekois'
target_id = 'ADRB2'
prefix = os.getcwd() + '/' + dataset + '/'
activeFile = prefix + 'ligands/' + target_id + '.sdf.gz'
decoyFile = prefix + 'decoys/' + target_id + '_Celling-v1.12_decoyset.sdf.gz'
data = ukyScore.data_set(activeFile, decoyFile, balanceTol=0.01)
bestSplit = data.randSplit()
labels = data.labels.values
bestScore = data.computeScore(bestSplit)


def sample(n, bestSplit, bestScore):
    for i in range(n):
        newSplit = data.randSplit()
        newScore = data.computeScore(newSplit)
        if newScore < bestScore:
            bestSplit = newSplit
            bestScore = newScore
    return bestSplit, bestScore


if __name__ == '__main__':
    p = Pool(15)
    p.map(sample, (10000, bestSplit,bestScore))
