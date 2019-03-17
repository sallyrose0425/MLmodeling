import numpy as np
import pandas as pd
from glob import glob
import warnings
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, roc_auc_score
import matplotlib.pyplot as plt


def sigmoid(scalar):
    sig = (1 + np.exp(-scalar))**(-1)
    return sig


def nnPrediction(x, p=1):
    label = x['labels']  # true label
    w = x['weights'] - 1  # w < 0 predict same class as label
    if w < 0:
        w = -(-w)**(1/p)
        w = sigmoid(w)
        return label, w
    else:
        w = w**(1/p)
        w = sigmoid(w)
        return 1 - label, w


columnNames = ['target_id', 'rfF1', 'rfF1_weighted', 'rfAUC', 'rfAUC_weighted',
               'nnF1', 'nnF1_weighted', 'nnAUC', 'nnAUC_weighted', 'optScore', 'atomwise time', 'atomwise bias']


dataset = 'dekois'
files = glob(dataset + '/*_dataPackage.pkl')
targets = []
for file in files:
    target_id = file.split('/')[1].split('_')[0]
    package = pd.read_pickle(file)
    features = package.drop(['split', 'labels', 'weights'], axis=1)
    training = package[package['split'] == 1]
    trainingFeatures = training.drop(['split', 'labels', 'weights'], axis=1)
    trainingLabels = training['labels']
    rf = RandomForestClassifier(n_estimators=100)
    rf.fit(trainingFeatures, trainingLabels)
    package['rfProbs'] = rf.predict_proba(features)[:, 1]
    package['rfPreds'] = package['rfProbs'].apply(lambda x: int(x > 0.5))
    package['nnPreds'] = package.apply(lambda x: nnPrediction(x, 1)[0], axis=1)
    package['nnProbs'] = package.apply(lambda x: nnPrediction(x, 2)[1], axis=1)
    nnPredictions = package[package['split'] == 0]['nnPreds']
    nnProbs = package[package['split'] == 0]['nnProbs']
    rfPredictions = package[package['split'] == 0]['rfPreds']
    rfProbabilities = package[package['split'] == 0]['rfProbs']
    validationLabels = package[package['split'] == 0]['labels']
    with warnings.catch_warnings():
        # Suppress warning from predicting no actives
        warnings.simplefilter("ignore")
        rfF1 = f1_score(validationLabels, rfPredictions)
        rfAUC = roc_auc_score(validationLabels, rfProbabilities)
        nnF1 = f1_score(validationLabels, nnPredictions)
        nnAUC = roc_auc_score(validationLabels, nnProbs)
        weights = package[package['split'] == 0]['weights']  # temporary weighting
        nnF1_weighted = f1_score(validationLabels, nnPredictions, sample_weight=weights)
        nnAUC_weighted = roc_auc_score(validationLabels, nnProbs, sample_weight=weights)
        rfF1_weighted = f1_score(validationLabels, rfPredictions, sample_weight=weights)
        rfAUC_weighted = roc_auc_score(validationLabels, rfProbabilities, sample_weight=weights)
    log = pd.read_pickle(dataset + '/' + target_id + '_optRecord.pkl')
    Alog = pd.read_pickle(dataset + '/' + target_id + '_atomwiseLog.pkl')
    optScore = log.tail(1).values[0, 1]
    atomwiseLog = Alog.tail(1).values[0]
    log = log.values
    Alog = Alog.values
    targets.append(pd.DataFrame([target_id, rfF1, rfF1_weighted, rfAUC, rfAUC_weighted,
                                 nnF1, nnF1_weighted, nnAUC, nnAUC_weighted, optScore, atomwiseLog[0], atomwiseLog[1]]).T)
    plt.figure()
    plt.plot(log[:, 0], log[:, 1], 'r', label='ukyOpt')
    plt.plot(Alog[:, 0], Alog[:, 1], 'k', label='Atomwise')
    plt.xlabel('Time (sec)')
    plt.ylabel('Score')
    plt.title(target_id)
    plt.legend()
    plt.savefig(dataset + '/' + target_id + '_opts')


contribFrame = pd.concat(targets)
contribFrame.columns = columnNames
contribFrame = contribFrame.set_index('target_id')

# save scatterplots
plt.figure()
plt.subplot(221)
plt.scatter(contribFrame['optScore'], contribFrame['rfF1'], marker='.')
plt.xlabel('Score')
plt.ylabel('RF F1')
plt.subplot(222)
plt.scatter(contribFrame['optScore'], contribFrame['rfAUC'], marker='.')
plt.xlabel('Score')
plt.ylabel('RF AUC')
plt.subplot(223)
plt.scatter(contribFrame['optScore'], contribFrame['nnF1'], marker='.')
plt.xlabel('Score')
plt.ylabel('NN F1')
plt.subplot(224)
plt.scatter(contribFrame['optScore'], contribFrame['nnAUC'], marker='.')
plt.xlabel('Score')
plt.ylabel('NN AUC')
plt.subplots_adjust(top=0.92, bottom=0.12, left=0.10, right=0.95, hspace=0.25,
                    wspace=0.35)
plt.savefig(dataset + '/' + 'scoreScatter')







