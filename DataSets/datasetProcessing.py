import os
import numpy as np
import pandas as pd
from glob import glob

import matplotlib.pyplot as plt


dataset = 'dekois'
prefix = os.getcwd() + '/DataSets/' + dataset + '/'
files = glob(os.getcwd() + '/DataSets/' + dataset + '/*_perfStats.pkl')
perfs = []
for file in files:
    perfs.append(pd.read_pickle(file))
perfs = pd.DataFrame(perfs, columns=['score', 'rfAUC', 'rfAUC_weighted', 'nnSimilarity',
                                     'rfAUC_PR', 'rfAUC_PR_weighted',
                                     'optScore', 'rfAUCopt', 'nnSimilarityOpt',
                                     'rfAUC_PR_Opt', 'rfAUC_PR_Opt_weighted'])


# nn distance and AUC before optimization
plt.figure()
plt.subplot(211)
plt.ylim(.975, 1.001)
plt.scatter(perfs['score'], perfs['nnSimilarity'])
pearson = np.round(perfs['score'].corr(perfs['nnSimilarity']), 2)
plt.title(f'Pearson {pearson}')
plt.xlabel('AVE bias')
plt.ylabel('NN similarity')

plt.subplot(212)
plt.scatter(perfs['score'], perfs['rfAUC_PR'])
plt.xlabel('AVE bias')
plt.ylabel('AUC')
pearson = np.round(perfs['score'].corr(perfs['rfAUC_PR']), 2)
plt.title(f'Pearson {pearson}')
plt.tight_layout()


# nn distance and AUC after optimization
plt.figure()
plt.subplot(211)
plt.ylim(0.9, 1.005)
plt.scatter(perfs['optScore'], perfs['nnSimilarityOpt'])
pearson = np.round(perfs['optScore'].corr(perfs['nnSimilarityOpt']), 2)
plt.title(f'Pearson {pearson}')
plt.xlabel('AVE bias')
plt.ylabel('NN similarity')

plt.subplot(212)
plt.scatter(perfs['optScore'], perfs['rfAUC_PR_Opt'])
plt.xlabel('AVE bias')
plt.ylabel('AUC')
pearson = np.round(perfs['optScore'].corr(perfs['rfAUC_PR_Opt']), 2)
plt.title(f'Pearson {pearson}')
plt.tight_layout()

###############################################################
# with new score (ATOMWISE==False)

files = glob(os.getcwd() + '/DataSets/' + dataset + '/*_perfStatsNew.pkl')
perfs = []
for file in files:
    perfs.append(pd.read_pickle(file))
perfs = pd.DataFrame(perfs, columns=['score', 'rfAUC', 'rfAUC_weighted', 'nnSimilarity',
                                     'rfAUC_PR', 'rfAUC_PR_weighted',
                                     'optScore', 'rfAUCopt', 'nnSimilarityOpt',
                                     'rfAUC_PR_Opt', 'rfAUC_PR_Opt_weighted'])


# nn distance and AUC before optimization
plt.figure()
plt.subplot(211)
# plt.ylim(.975, 1.001)
plt.scatter(perfs['score'], perfs['rfAUC_PR'])
pearson = np.round(perfs['score'].corr(perfs['rfAUC_PR']), 2)
plt.title(f'Pearson {pearson}')
plt.xlabel('AVE bias')
plt.ylabel('PR-AUC')

plt.subplot(212)
plt.scatter(perfs['score'], perfs['rfAUC_PR_Opt'])
plt.xlabel('AVE bias')
plt.ylabel('PR-AUC')
pearson = np.round(perfs['score'].corr(perfs['rfAUC_PR_Opt']), 2)
plt.title(f'Pearson {pearson}')
plt.tight_layout()


# nn distance and AUC after optimization
plt.figure()
plt.subplot(211)
plt.ylim(0.9, 1.005)
plt.scatter(perfs['optScore'], perfs['nnSimilarityOpt'])
pearson = np.round(perfs['optScore'].corr(perfs['nnSimilarityOpt']), 2)
plt.title(f'Pearson {pearson}')
plt.xlabel('AVE bias')
plt.ylabel('NN similarity')

plt.subplot(212)
plt.scatter(perfs['optScore'], perfs['rfAUC_PR_Opt'])
plt.xlabel('AVE bias')
plt.ylabel('AUC')
pearson = np.round(perfs['optScore'].corr(perfs['rfAUC_PR_Opt']), 2)
plt.title(f'Pearson {pearson}')
plt.tight_layout()


###############################################################

# AUC weighted
plt.figure()
plt.subplot(211)
plt.scatter(perfs['score'], perfs['rfAUC_PR_weighted'])
pearson = np.round(perfs['score'].corr(perfs['rfAUC_PR_weighted']), 2)
plt.title(f'Pearson {pearson}')
plt.xlabel('AVE bias')
plt.ylabel('weighted AUC')

plt.subplot(212)
plt.scatter(perfs['nnSimilarity'], perfs['rfAUC_PR_weighted'])
pearson = np.round(perfs['nnSimilarity'].corr(perfs['rfAUC_PR_weighted']), 2)
plt.title(f'Pearson {pearson}')
plt.xlabel('NN similarity')
plt.ylabel('weighted AUC')
plt.tight_layout()


plt.scatter(perfs['nnSimilarityOpt'], perfs['rfAUC_PR_Opt'])
plt.xlabel('NN similarity')
plt.ylabel('AUC')
pearson = np.round(perfs['nnSimilarityOpt'].corr(perfs['rfAUC_PR_Opt']), 2)
plt.title(f'Pearson {pearson}')


files = glob(os.getcwd() + '/DataSets/' + dataset + '/*_dataPackage.pkl')
weights = pd.Series([])
for file in files:
    target_weights = pd.read_pickle(file)['weights']
    # target_weights = target_weights[target_weights > 0]
    weights = weights.append(target_weights)
weights = weights[weights > 0].apply(lambda x: x-1)
weights.plot.box(grid=True, showfliers=False)
weights.hist(histtype='stepfilled', density=True, bins=100)





