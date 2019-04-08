import os
import sys
import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, jaccard_similarity_score, auc, precision_recall_curve

import ukyScore

ATOMWISE = True  # (False) Use the atomwise approximation
metric = 'jaccard'  # ('jaccard') Metric for use in determining fingerprint distances
score_goal = 0.01  # (0.02) Early termination of genetic optimizer if goal is reached
numGens = 500  # (1000) Number of generations to run in genetic optimizer
popSize = 500
print_frequency = 50  # (100) How many generations of optimizer before printing update
targetRatio = 0.8  # (0.8) Target training/validation ratio of the split
ratioTol = 0.01  # (0.01) tolerance for targetRatio
balanceTol = 0.02  # (0.02) Tolerance for active/decoy ratio in validations set relative to total data set


def nnPredictor(weight, label):
    if weight < 1:
        return label
    else:
        return 1 - label


def distToNN(probs, nn):
    dist = []
    for t in np.linspace(0, 1, num=100):
        preds = probs > t
        preds = preds.apply(lambda a: int(a)).values
        dist.append(jaccard_similarity_score(preds, nn.values))
    return max(dist)


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
    data = ukyScore.data_set(target_id, activeFile, decoyFile, targetRatio, ratioTol, balanceTol, atomwise=ATOMWISE, Metric=metric)
    # Three-fold cross validation stats
    skf = StratifiedKFold(n_splits=3, shuffle=True)
    splits = [(train, test) for train, test in skf.split(data.fingerprints, data.labels)]
    perf = []
    rf = RandomForestClassifier(n_estimators=100)
    for splitIndices in splits:
        trainIndices, validIndices = splitIndices
        trainingFeatures = data.fingerprints.T[trainIndices].T
        trainingLabels = data.labels[trainIndices]
        validationLabels = data.labels[validIndices]
        split = np.array([int(x in trainIndices) for x in range(data.size)])
        score = data.computeScores(split, check=False)
        if ATOMWISE:
            score = score[0] + score[1]
        else:
            score = np.sqrt(score[0] ** 2 + score[1] ** 2)
        rf.fit(trainingFeatures, trainingLabels)
        rfProbs = rf.predict_proba(data.fingerprints)[:, 1]
        precision, recall, thresholds = precision_recall_curve(validationLabels, rfProbs[validIndices])
        rfAUC_PR = auc(recall, precision)
        rfAUC = roc_auc_score(validationLabels, rfProbs[validIndices])
        weights = pd.Series(data.weights(split))  # temporary weighting
        metricFrame = pd.DataFrame([data.labels, split, weights, rfProbs],
                                   index=['labels', 'split', 'weights', 'rfProbs']).T
        metricFrame['nn'] = metricFrame.apply(lambda t: nnPredictor(t.loc['weights'], t.loc['labels']), axis=1)
        nnDist = distToNN(metricFrame['rfProbs'], metricFrame['nn'])
        compWeights = weights[validIndices].values
        hist, bin_edges = np.histogram(compWeights, density=True, bins=100)
        hist = np.cumsum(hist * np.diff(bin_edges))

        def cfd(x):
            try:
                findBin = [x >= y for y in bin_edges].index(False)
            except ValueError:
                return 1
            if findBin == 100:
                return 1
            else:
                return hist[findBin]

        weights = np.array([cfd(x) for x in weights])
        precision, recall, thresholds = precision_recall_curve(validationLabels, rfProbs[split == 0],
                                                               sample_weight=weights[validIndices])
        rfAUC_PR_weighted = auc(recall, precision)
        rfAUC_weighted = roc_auc_score(validationLabels, rfProbs[validIndices], sample_weight=weights[validIndices])
        perf.append((score, rfAUC, rfAUC_weighted, nnDist, rfAUC_PR, rfAUC_PR_weighted))
    meanScore, meanRfAUC, meanRfAUCWeighted, meanNnDist, rfAUC_PR, rfAUC_PR_weighted = np.mean(np.array(perf), axis=0)
    # Run the geneticOptimizer method on data
    splits = data.geneticOptimizer(numGens, POPSIZE=popSize, printFreq=print_frequency, scoreGoal=score_goal)
    # Grab optimal split from polulation
    scores = [data.objectiveFunction(split)[0] for split in splits]
    split = splits[np.argmin(scores)]
    # Record performance stats
    trainingFeatures = data.fingerprints[split == 1]
    trainingLabels = data.labels[split == 1]
    rf.fit(trainingFeatures, trainingLabels)
    rfProbs = rf.predict_proba(data.fingerprints)[:, 1]
    validationLabels = data.labels[split == 0]
    rfAUC = roc_auc_score(validationLabels, rfProbs[split == 0])
    precision, recall, thresholds = precision_recall_curve(validationLabels, rfProbs[split == 0])
    rfAUC_PR_Opt = auc(recall, precision)
    weights = pd.Series(data.weights(split))  # temporary weighting
    precision, recall, thresholds = precision_recall_curve(validationLabels, rfProbs[split == 0],
                                                           sample_weight=weights[split == 0])
    rfAUC_PR_Opt_weighted = auc(recall, precision)
    metricFrame = pd.DataFrame([data.labels, split, weights, rfProbs],
                               index=['labels', 'split', 'weights', 'rfProbs']).T
    metricFrame['nn'] = metricFrame.apply(lambda t: nnPredictor(t.loc['weights'], t.loc['labels']), axis=1)
    metricFrame = metricFrame[split == 0]
    nnDistOpt = distToNN(metricFrame['rfProbs'], metricFrame['nn'])
    # Record features, labels, split, and some metrics
    data.fingerprints['labels'] = data.labels
    data.fingerprints['split'] = split
    data.fingerprints['weights'] = data.weights(split)
    pd.to_pickle(data.fingerprints, prefix + target_id + '_dataPackage.pkl')
    pd.to_pickle(pd.DataFrame(data.optRecord, columns=['time', 'AA-AI', 'II-IA', 'score']),
                 prefix + target_id + '_optRecordNew.pkl')
    statsArray = np.array([meanScore, meanRfAUC, meanRfAUCWeighted, meanNnDist,
                           rfAUC_PR, rfAUC_PR_weighted,
                           min(scores), rfAUC, nnDistOpt,
                           rfAUC_PR_Opt, rfAUC_PR_Opt_weighted])
    pd.to_pickle(statsArray, prefix + target_id + '_perfStats.pkl')


if __name__ == '__main__':
    if len(sys.argv) > 2:
        main(sys.argv[1], sys.argv[2])
    else:
        print("Specify dataset and target...")
