import os
import sys

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold

import ukyScore

ATOMWISE = True  # (False) Use the atomwise approximation
metric = 'jaccard'  # ('jaccard') Metric for use in determining fingerprint distances


def main(dataset, target_id):
    prefix = os.getcwd() + '/' + dataset + '/'
    if dataset == 'dekois':
        activeFile = prefix + 'ligands/' + target_id + '.sdf.gz'
        decoyFile = prefix + 'decoys/' + target_id + '_Celling-v1.12_decoyset.sdf.gz'
    elif dataset == 'DUDE':
        activeFile = prefix + target_id + '/actives_final.sdf.gz'
        decoyFile = prefix + target_id + '/decoys_final.sdf.gz'
    elif dataset == 'MUV':
        activeFile = prefix + target_id + '_actives.sdf.gz'
        decoyFile = prefix + target_id + '_decoys.sdf.gz'
    else:
        print('Invalid dataset specified. Did you mean MUV, dekois, or DUDE?')
        return
    # Create data_set class instance called "data"
    print(f'Creating data set {target_id}')
    data = ukyScore.data_set(target_id, activeFile, decoyFile, atomwise=ATOMWISE, Metric=metric)
    skf = StratifiedKFold(n_splits=3, shuffle=True)
    splits = [(train, test) for train, test in skf.split(data.fingerprints, data.labels)]
    for splitIndices in splits:
        trainIndices, validIndices = splits[0]
        X_train = data.fingerprints.T[trainIndices].T
        X_valid = data.fingerprints.T[validIndices].T
        y_train = data.labels[trainIndices]
        y_valid = data.labels[validIndices]
        split = np.array([int(x in trainIndices) for x in range(data.size)])
        data.computeScores(split)
    # Record features, labels, split, and some metrics
    data.fingerprints['labels'] = data.labels
    data.fingerprints['split'] = split
    data.fingerprints['weights'] = data.weights(split)
    pd.to_pickle(data.fingerprints, prefix + target_id + '_dataPackage.pkl')



if __name__ == '__main__':
    if len(sys.argv) > 2:
        main(sys.argv[1], sys.argv[2])
    else:
        print("Specify dataset and target...")

dataset = 'dekois'
target_id = '11betaHSD1'