import os
from glob import glob
import numpy as np
from time import time
import matplotlib.pyplot as plt

import ukyScore


def approx(scalar):
    return np.floor(50*scalar)


Approx = np.vectorize(approx)


prefix = os.getcwd() + '/DataSets/dekois/'
files = glob(prefix + '*')
targets = []
for file in files:
    target_id = file.split('/')[-1].split('_')[0]
    if target_id.find('.') == -1:
        targets.append(target_id)
targets = list(set(targets))
times = []
scores = []
for target_id in targets:
    print(target_id)
    time_tmp = []
    try:
        activeFile = prefix + 'ligands/' + target_id + '.sdf.gz'
        decoyFile = prefix + 'decoys/' + target_id + '_Celling-v1.12_decoyset.sdf.gz'
        data = ukyScore.data_set(target_id, activeFile, decoyFile, 0.8, 0.01, 0.02, atomwise=True, Metric='jaccard')
        split = data.randSplit()
        labels = data.labels
        aTest_aTrain_D = data.distanceMatrix[(split == 0) & (labels == 1), :][:, (split == 1) & (labels == 1)]
        aTest_iTrain_D = data.distanceMatrix[(split == 0) & (labels == 1), :][:, (split == 1) & (labels == 0)]
        iTest_aTrain_D = data.distanceMatrix[(split == 0) & (labels == 0), :][:, (split == 1) & (labels == 1)]
        iTest_iTrain_D = data.distanceMatrix[(split == 0) & (labels == 0), :][:, (split == 1) & (labels == 0)]

        t0 = time()
        for i in range(1000):
            aTest_aTrain_S = np.mean([np.mean(np.any(aTest_aTrain_D < t, axis=1)) for t in np.linspace(0, 1.0, 51)])
            aTest_iTrain_S = np.mean([np.mean(np.any(aTest_iTrain_D < t, axis=1)) for t in np.linspace(0, 1.0, 51)])
            iTest_iTrain_S = np.mean([np.mean(np.any(iTest_iTrain_D < t, axis=1)) for t in np.linspace(0, 1.0, 51)])
            iTest_aTrain_S = np.mean([np.mean(np.any(iTest_aTrain_D < t, axis=1)) for t in np.linspace(0, 1.0, 51)])
            Atomwise = aTest_aTrain_S - aTest_iTrain_S + iTest_iTrain_S - iTest_aTrain_S
        time_tmp.append(time() - t0)

        t0 = time()
        for i in range(1000):
            minPosPosDist = np.amin(aTest_aTrain_D, axis=1)
            minPosNegDist = np.amin(aTest_iTrain_D, axis=1)
            minNegPosDist = np.amin(iTest_aTrain_D, axis=1)
            minNegNegDist = np.amin(iTest_iTrain_D, axis=1)
            ukyApprox = (np.mean(Approx(minPosNegDist) - Approx(minPosPosDist))\
                        + np.mean(Approx(minNegPosDist) - Approx(minNegNegDist)))/51
        time_tmp.append(time() - t0)

        t0 = time()
        for i in range(1000):
            minPosPosDist = np.amin(aTest_aTrain_D, axis=1)
            minPosNegDist = np.amin(aTest_iTrain_D, axis=1)
            minNegPosDist = np.amin(iTest_aTrain_D, axis=1)
            minNegNegDist = np.amin(iTest_iTrain_D, axis=1)
            score = np.mean(minPosNegDist) - np.mean(minPosPosDist) +\
                    np.mean(minNegPosDist) - np.mean(minNegNegDist)
        time_tmp.append(time() - t0)

        t0 = time()
        for i in range(1000):
            minPosPosDist = np.amin(aTest_aTrain_D, axis=1)
            minPosNegDist = np.amin(aTest_iTrain_D, axis=1)
            minNegPosDist = np.amin(iTest_aTrain_D, axis=1)
            minNegNegDist = np.amin(iTest_iTrain_D, axis=1)
            uky_Score = np.sqrt((np.mean(minPosNegDist) - np.mean(minPosPosDist))**2 +\
                            (np.mean(minNegPosDist) - np.mean(minNegNegDist))**2)
        time_tmp.append(time() - t0)

        times.append(time_tmp)
        scores.append([Atomwise, ukyApprox, score, uky_Score])
    except AttributeError:
        print('Error')
        pass

times = np.array(times)
scores = np.array(scores)

aveTimes = np.mean(times, axis=0)
1.0/(aveTimes/aveTimes[0])[1:]

compare = np.abs(scores[:, 0] - scores[:, 2])/scores[:, 0]
np.mean(compare)
np.max(compare)


plt.figure()
plt.plot([0, 1], [0, 1], ls='--', c='k', label='AVE Bias', zorder=0)
# plt.plot([0,1],[0.02,1.02], c='g',zorder=1)
# plt.plot([0,1],[-0.02,0.98], c='g',zorder=2)
plt.scatter(scores[:, 0], scores[:, 1], alpha=0.8, label='Expression (3)', zorder=1)
plt.scatter(scores[:, 0], scores[:, 2], alpha=0.8, label='Expression (4)', zorder=2)
plt.scatter(scores[:, 0], scores[:, 3], alpha=0.8, label='VE Score', zorder=3)
plt.legend()
plt.xlabel('AVE Bias')

