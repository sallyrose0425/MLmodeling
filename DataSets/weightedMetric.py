# cd DataSets
import os
from ukyScore import data_set
import numpy as np
import pandas as pd

from importlib import reload
reload(data_set)

dataset = 'dekois'
target_id = '11betaHSD1'
prefix = os.getcwd() + '/' + dataset + '/'
activeFile = prefix + 'ligands/' + target_id + '.sdf.gz'
decoyFile = prefix + 'decoys/' + target_id + '_Celling-v1.12_decoyset.sdf.gz'
data = data_set(activeFile, decoyFile)
splits = data.geneticOptimizer(50, printFreq=50, scoreGoal=0.02)
scores = [data.computeScore(split) for split in splits]
split = np.array(splits[np.argmin(scores)])


################################################################################

def weight(x, a):
    return np.exp(a*(x-1)) / (np.exp(a*(x-1)) + 1)


actActDistances = data.distanceMatrix[(split == 0) & (data.labels == 1), :][:, (split == 1) & (data.labels == 1)]
actDecDistances = data.distanceMatrix[(split == 0) & (data.labels == 1), :][:, (split == 1) & (data.labels == 0)]
actWeights = weight((np.amin(actActDistances, axis=1) / np.amin(actDecDistances, axis=1)))

decActDistances = data.distanceMatrix[(split == 0) & (data.labels == 0), :][:, (split == 1) & (data.labels == 1)]
decDecDistances = data.distanceMatrix[(split == 0) & (data.labels == 0), :][:, (split == 1) & (data.labels == 0)]
decWeights = weight((np.amin(decActDistances, axis=1) / np.amin(decDecDistances, axis=1)))

holdWeights = np.zeros(data.size)
validActiveIndices = np.where((split == 0) & (data.labels == 1))[0]
for i in range(len(validActiveIndices)):
    holdWeights[validActiveIndices[i]] = actWeights[i]

validDecoyIndices = np.where((split == 0) & (data.labels == 0))[0]
for i in range(len(validDecoyIndices)):
    holdWeights[validDecoyIndices[i]] = decWeights[i]

labels = data.labels.values
predictions = data.nearestNeighborPredictions(split)
activeValidWeights = np.multiply(holdWeights, labels)
decoyValidWeights = np.multiply(holdWeights, 1 - labels)
activeTotalWeight = np.sum(activeValidWeights)
decoyTotalWeight = np.sum(decoyValidWeights)
truePositiveWeight = np.sum(np.multiply(activeValidWeights[split == 0], predictions))
trueNegativeWeight = np.sum(np.multiply(decoyValidWeights[split == 0], 1 - predictions))
truePositiveWeight / activeTotalWeight
trueNegativeWeight / decoyTotalWeight

perf = data.weightedPerformance(split, predictions, a=10)
perf[0]/perf[1], perf[2]/perf[3]