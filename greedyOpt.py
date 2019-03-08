from ukySplit import ukyDataSet
import numpy as np
import pandas as pd
# Making artificial data:
np.random.seed(42)
X = np.random.randint(2, size=(100, 10))
y = np.random.randint(2, size=100)
attr_df = pd.DataFrame(np.array([hex(t) for t in range(100)]), columns=['smiles_col'])
class Dset:
    def __init__(self, X, y):
        self.X = X
        self.y = y
dataset = Dset(X,y)
# Creating the ukyDataset, running optimizer, and splitting the data set:
data = ukyDataSet(dataset.X, dataset.y, ids=attr_df['smiles_col'].values, Metric='jaccard')

distanceMatrix = np.random.sample((100, 100))
distanceMatrix = np.matmul(distanceMatrix, np.transpose(distanceMatrix))
distanceMatrix = distanceMatrix / np.max(distanceMatrix)
for i in range(100):
    distanceMatrix[i,i]=1

# using absolute indices!
def candidateScore(j, active, decoy):
    distances = distanceMatrix[j,:]
    minAct = active[np.argmin(distances[active])]
    minActDist = distances[minAct]
    minDec = decoy[np.argmin(distances[decoy])]
    minDecDist = distances[minDec]
    return minDecDist, minActDist, minAct, minDec


def greedyOptimizer(self, targetActiveValid, targetDecoyValid):
    t0 = time()
    split = np.ones(self.size, dtype=int)
    i = 0
    while i < targetActiveValid:
        activeCandidates = np.where((self.labels == 1) & (split == 1))[0]
        activeTraining = np.where((self.labels == 1) & (split != 0))[0]
        decoyTraining = np.where((self.labels == 0) & (split != 0))[0]
        scores = [
            self.candidateScore(j, activeTraining, decoyTraining)[0]
            - self.candidateScore(j, activeTraining, decoyTraining)[1]
            for j in activeCandidates]
        toAdd = activeCandidates[np.argmin(scores)]
        minDecDist, minActDist, minAct, minDec = self.candidateScore(toAdd, activeTraining, decoyTraining)
        split[toAdd] = 0
        split[minAct] = -1
        split[minDec] = -1
        i += 1

    i = 0
    while i < targetDecoyValid:
        decoyCandidates = np.where((self.labels == 0) & (split == 1))[0]
        activeTraining = np.where((self.labels == 1) & (split != 0))[0]
        decoyTraining = np.where((self.labels == 0) & (split != 0))[0]
        scores = [
            self.candidateScore(j, activeTraining, decoyTraining)[1]
            - self.candidateScore(j, activeTraining, decoyTraining)[0]
            for j in activeCandidates]
        toAdd = decoyCandidates[np.argmin(scores)]
        minDecDist, minActDist, minAct, minDec = self.candidateScore(toAdd, activeTraining, decoyTraining)
        split[toAdd] = 0
        split[minAct] = -1
        split[minDec] = -1
        i += 1
    print(f'Elapsed time: {np.round(time() - t0, 2)}')
    return split, np.abs(split).astype(int)


numValidation = data.size - numTraining
trueRatio = float(numTraining) / data.size
ratioError = np.abs(data.targetRatio - trueRatio)
numActives = np.sum(data.labels)
balance = float(numActives) / data.size