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
perfs = pd.DataFrame(perfs, columns=['score', 'rfAUC', 'rfAUC_weighted', 'nnSimilarity', 'optScore', 'rfAUCopt', 'nnSimilarityOpt'])
# perfs['nnSimilarity'] = 1 - perfs['nnSimilarity']
# perfs['nnSimilarityOpt'] = 1 - perfs['nnSimilarityOpt']

# nn distance before and after optmization
plt.figure()
plt.subplot(311)
plt.scatter(perfs['score'], perfs['nnSimilarity'])
pearson = np.round(perfs['score'].corr(perfs['nnSimilarity']), 2)
plt.title(f'Pearson {pearson}')
plt.xlabel('mean AVE bias')
plt.ylabel('mean nn similarity')
plt.subplot(312)
plt.scatter(perfs['nnSimilarity'], perfs['rfAUC'])
plt.xlabel('mean nn similarity')
plt.ylabel('mean rf AUC')
pearson = np.round(perfs['nnSimilarity'].corr(perfs['rfAUC']), 2)
plt.title(f'Pearson {pearson}')
plt.subplot(313)
plt.scatter(perfs['score'], perfs['rfAUC'])
plt.xlabel('mean AVE bias')
plt.ylabel('mean rf AUC')
pearson = np.round(perfs['score'].corr(perfs['rfAUC']), 2)
plt.title(f'Pearson {pearson}')
plt.tight_layout()


plt.figure()
plt.subplot(311)
plt.scatter(perfs['optScore'], perfs['nnSimilarityOpt'])
pearson = np.round(perfs['optScore'].corr(perfs['nnSimilarityOpt']), 2)
plt.title(f'Pearson {pearson}')
plt.xlabel('AVE bias')
plt.ylabel('nn similarity')
plt.subplot(313)
plt.scatter(perfs['optScore'], perfs['rfAUCopt'])
plt.xlabel('AVE bias')
plt.ylabel('rf AUC')
pearson = np.round(perfs['optScore'].corr(perfs['rfAUCopt']), 2)
plt.title(f'Pearson {pearson}')
plt.subplot(312)
plt.scatter(perfs['nnSimilarityOpt'], perfs['rfAUCopt'])
plt.xlabel('nn similarity')
plt.ylabel('rf AUC')
pearson = np.round(perfs['nnSimilarityOpt'].corr(perfs['rfAUCopt']), 2)
plt.title(f'Pearson {pearson}')
plt.tight_layout()

plt.figure()
plt.scatter(perfs['score'], perfs['nnSimilarity'])
plt.xlabel('mean AVE bias')
plt.ylabel('mean nn similarity')
plt.subplot(211)
plt.scatter(perfs['score'], perfs['rfAUC'])
pearson = np.round(perfs['score'].corr(perfs['rfAUC']), 2)
plt.title(f'Pearson {pearson}')
plt.xlabel('mean AVE bias')
plt.ylabel('mean AUC')
plt.subplot(212)
plt.scatter(perfs['score'], perfs['rfAUC_weighted'])
pearson = np.round(perfs['score'].corr(perfs['rfAUC_weighted']), 2)
plt.title(f'Pearson {pearson}')
plt.xlabel('mean AVE bias core')
plt.ylabel('weighted AUC score')
plt.tight_layout()




