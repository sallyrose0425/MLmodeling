import os
import sys

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, auc, roc_curve

import ukyScore

ATOMWISE = True  # (False) Use the atomwise approximation
metric = 'jaccard'  # ('jaccard') Metric for use in determining fingerprint distances


def sigmoid(scalar):
    sig = (1 + np.exp(-scalar))**(-1)
    return sig


def weightedROC(t, optPackage):
    optPackage['rfPreds'] = optPackage['rfProbs'] > t
    optPackage['rfPreds'] = optPackage['rfPreds'].apply(lambda x: int(x))
    Pos = optPackage[optPackage['rfPreds'] == 1]
    Neg = optPackage[optPackage['rfPreds'] == 0]
    Pos = Pos[Pos['split'] == 0]
    Neg = Neg[Neg['split'] == 0]
    truePos = Pos[Pos['labels'] == 1]
    falsePos = Pos[Pos['labels'] == 0]
    trueNeg = Neg[Neg['labels'] == 0]
    falseNeg = Neg[Neg['labels'] == 1]
    # falseNeg['weights'] = falseNeg['weights'].apply(lambda x: 0 if x == 0 else 1/x)
    # falsePos['weights'] = falsePos['weights'].apply(lambda x: 0 if x == 0 else 1/x)
    TPR = truePos['weights'].sum() / (truePos['weights'].sum() + falseNeg['weights'].sum())
    FPR = falsePos['weights'].sum() / (falsePos['weights'].sum() + trueNeg['weights'].sum())
    return [FPR, TPR]


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
        split = np.array([int(x in trainIndices) for x in range(data.size)])
        weights = data.weights(split)  # temporary weighting
        score = data.computeScores(split, check=False)
        if ATOMWISE:
            score = score[0] + score[1]
        else:
            score = np.sqrt(score[0]**2 + score[1]**2)
        rf = RandomForestClassifier(n_estimators=100)
        rf.fit(trainingFeatures, trainingLabels)
        rfProbs = rf.predict_proba(data.fingerprints)[:, 1]
        rfAUC = roc_auc_score(validationLabels, rfProbs[validIndices])
        metricFrame = pd.DataFrame([data.labels, split, weights, rfProbs],
                                   index=['labels', 'split', 'weights', 'rfProbs']).T
        curve = np.array([weightedROC(t, metricFrame) for t in np.linspace(0, 1, num=100)])
        curve = np.vstack([np.array([1, 1]), curve])
        rfAUC_weighted = auc(curve[:, 0], curve[:, 1])
        # rfAUC_weighted = roc_auc_score(validationLabels, rfProbs, sample_weight=weights])
        perf.append((score, rfAUC, rfAUC_weighted))
    pd.to_pickle(perf, f'{prefix}{target_id}_performance.pkl')


if __name__ == '__main__':
    if len(sys.argv) > 2:
        main(sys.argv[1], sys.argv[2])
    else:
        print("Specify dataset and target...")
