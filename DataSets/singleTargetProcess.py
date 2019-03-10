import os
import sys

import pandas as pd
import numpy as np

import ukyScore

ATOMWISE = False  # (False) Use the atomwise approximation
metric = 'jaccard'  # ('jaccard') Metric for use in determining fingerprint distances
score_goal = 0.02  # (0.02) Early termination of genetic optimizer if goal is reached
numGens = 1000  # (1000) Number of generations to run in genetic optimizer
print_frequency = 100  # (100) How many generations of optimizer before printing update
targetRatio = 0.8
ratioTol = 0.01
balanceTol = 0.05


def main(dataset, target_id):
    prefix = os.getcwd() + '/' + dataset + '/'
    if dataset == 'dekois':
        activeFile = prefix + 'ligands/' + target_id + '.sdf.gz'
        decoyFile = prefix + 'decoys/' + target_id + '_Celling-v1.12_decoyset.sdf.gz'
    elif dataset == 'DUDE':
        activeFile = prefix + target_id + '/actives_final.sdf.gz'
        decoyFile = prefix+ target_id + '/decoys_final.sdf.gz'
    elif dataset == 'MUV':
        activeFile = prefix + target_id + '_actives.sdf.gz'
        decoyFile = prefix+ target_id + '_decoys.sdf.gz'
    else:
        print('Invalid dataset specified. Did you mean MUV, dekois, or DUDE?')
        return
    data = ukyScore.data_set(activeFile, decoyFile, targetRatio, ratioTol, balanceTol, atomwise=ATOMWISE, Metric=metric)
    splits = data.geneticOptimizer(numGens, printFreq=print_frequency, scoreGoal=score_goal)
    scores = [data.computeScore(split) for split in splits]
    split = splits[np.argmin(scores)]
    data.fingerprints['labels'] = data.labels
    data.fingerprints['split'] = split
    pd.to_pickle(data.fingerprints, prefix + target_id + '_dataPackage.pkl')


if __name__ == '__main__':
    if len(sys.argv) > 2:
        main(sys.argv[1], sys.argv[2])
    else:
        print("Specify dataset and target...")
