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
    perf = []
    for splitIndices in splits:
        trainIndices, validIndices = splitIndices
        trainingFeatures = data.fingerprints.T[trainIndices].T
        validFeatures = data.fingerprints.T[validIndices].T
        trainingLabels = data.labels[trainIndices]
        validationLabels = data.labels[validIndices]
        weights = (data.weights(split))[validIndices]  # temporary weighting
        split = np.array([int(x in trainIndices) for x in range(data.size)])
        score = data.computeScores(split, check=False)
        score = np.sqrt(score[0]**2 + score[1]**2)
        rf = RandomForestClassifier(n_estimators=100)
        rf.fit(trainingFeatures, trainingLabels)
        rfProbs = rf.predict_proba(validFeatures)[:, 1]
        rfAUC = roc_auc_score(validationLabels, rfProbs)
        rfAUC_weighted = roc_auc_score(validationLabels, rfProbs, sample_weight=weights)
        perf.append((score, rfAUC, rfAUC_weighted))
    pd.to_pickle(perf, f'{prefix}{target_id}_performance.pkl')


if __name__ == '__main__':
    if len(sys.argv) > 2:
        main(sys.argv[1], sys.argv[2])
    else:
        print("Specify dataset and target...")

dataset = 'dekois'
target_id = '11betaHSD1'