import os
import numpy as np
import pandas as pd
from glob import glob
import warnings
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, roc_auc_score, auc
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import stats


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


def expModel(x, a, b, c, d):
    return a + b*np.exp(-c*x) + d*x


def fitModel(x, y):
    popt, pcov = curve_fit(expModel, x, y, p0=(0.0, 0.5, 0.005, 0.0))

    def f(z):
        return expModel(z, *popt)
    return f, popt


columnNames = ['target_id', 'rfF1', 'rfF1_weighted', 'rfAUC', 'rfAUC_weighted',
               'nnF1', 'nnF1_weighted', 'nnAUC', 'nnAUC_weighted', 'optScore']


dataset = 'dekois'
files = glob(os.getcwd() + '/DataSets/' + dataset + '/*_dataPackage.pkl')
targets = []
params = []
paramsAtom = []
aggStats = []
for file in files:
    target_id = file.split('/')[-1].split('_')[0]
    package = pd.read_pickle(file)
    features = package.drop(['split', 'labels', 'weights'], axis=1)
    training = package[package['split'] == 1]
    trainingFeatures = training.drop(['split', 'labels', 'weights'], axis=1)
    trainingLabels = training['labels']
    rf = RandomForestClassifier(n_estimators=100)
    rf.fit(trainingFeatures, trainingLabels)
    package['rfProbs'] = rf.predict_proba(features)[:, 1]
    package['rfPreds'] = package['rfProbs'].apply(lambda x: int(x > 0.1))
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
        try:
            rfF1 = f1_score(validationLabels, rfPredictions)
            rfAUC = roc_auc_score(validationLabels, rfProbabilities)
            nnF1 = f1_score(validationLabels, nnPredictions)
            nnAUC = roc_auc_score(validationLabels, nnProbs)
            weights = package[package['split'] == 0]['weights']**2  # temporary weighting
            nnF1_weighted = f1_score(validationLabels, nnPredictions, sample_weight=weights)
            nnAUC_weighted = roc_auc_score(validationLabels, nnProbs, sample_weight=weights)
            rfF1_weighted = f1_score(validationLabels, rfPredictions, sample_weight=weights)
            rfAUC_weighted = roc_auc_score(validationLabels, rfProbabilities, sample_weight=weights)
            log = pd.read_pickle(os.getcwd() + '/DataSets/' + dataset + '/' + target_id + '_optRecord.pkl')
            optScore = log.tail(1).values[0, 1]
            targets.append(pd.DataFrame([target_id, rfF1, rfF1_weighted, rfAUC, rfAUC_weighted, nnF1, nnF1_weighted, \
                                         nnAUC, nnAUC_weighted, optScore]).T)
        except ValueError:
            pass

    samples = pd.read_pickle(os.getcwd() + '/DataSets/' + dataset + '/' + target_id + '_samples.pkl')
    s, t, mean, var, skew, kurt = stats.describe(samples)

    log = log.rename({0:'time', 1:'AA-AI', 2:'II-IA', 3:'score'}, axis=1)

    #logNew = pd.read_pickle(os.getcwd() + '/DataSets/' + dataset + '/' + target_id + '_optRecordNewScore.pkl')
    #logNew = logNew.rename({0:'time', 1:'AA-AI', 2:'II-IA', 3:'score'}, axis=1)
    try:
        Alog = pd.read_pickle(os.getcwd() + '/DataSets/' + dataset + '/' + target_id + '_atomwiseLog.pkl')
        if len(Alog)>0:

            atomwiseLog = Alog.tail(1).values[0]

            log = log.values
            #logNew = logNew.values
            Alog = Alog.values
            with warnings.catch_warnings():
                # Suppress warning from predicting no actives
                warnings.simplefilter("ignore")
                if (len(log) > 4) and (len(Alog) > 4):
                    try:
                        model, par = fitModel(log[1:, 0], log[1:, 3])
                        modelAtom, parAtom = fitModel(Alog[1:, 0], Alog[1:, 1])
                        params.append(par)
                        paramsAtom.append(parAtom)
                        aggStats.append((mean, var, skew, kurt))
                    except RuntimeError:
                        pass
            # generate optimizer comparison plot
            fig = plt.figure()
            plt.plot(log[:, 0], log[:, 3], 'r', marker='^', linewidth=0, label='ukyOpt')
            # plt.plot(logNew[:, 0], logNew[:, 3], 'b', marker='s', linewidth=0, label='ukyOptNew')
            plt.plot(Alog[:, 0], Alog[:, 1], 'k', marker='.', linewidth=0, label='Atomwise')
            plt.xlabel('Time (sec)')
            plt.ylabel('Score')
            plt.title(target_id)
            plt.legend()
            plt.savefig(os.getcwd() + '/DataSets/' + dataset + '/' + target_id + '_optsScore')
            plt.close(fig)
    except FileNotFoundError:
        log = log.values
        fig = plt.figure()
        plt.plot(log[:, 0], log[:, 3], 'r', marker='^', linewidth=0, label='ukyOpt')
        plt.title(target_id)
        plt.savefig(os.getcwd() + '/DataSets/' + dataset + '/' + target_id + '_optsScore')
        plt.close(fig)

# generate aggregate model plots
params = pd.DataFrame(params)
paramsAtom = pd.DataFrame(paramsAtom)
meanParams = params.mean().values
meanAtomParams = paramsAtom.mean().values
X = np.linspace(9,4000, 200)
plt.figure()
plt.plot(X, expModel(X, *meanParams), 'b', linewidth=5, label='ukyOpt')
plt.plot(X, expModel(X, *meanAtomParams), 'r', linewidth=5, label='Atomwise')
plt.title('Model Means')
plt.legend()
plt.show()





statsDF = pd.DataFrame(aggStats)
combined = pd.concat([params, statsDF], axis=1)
combined.corr().values[4:,0:4]


contribFrame = pd.concat(targets)
contribFrame.columns = columnNames
contribFrame = contribFrame.set_index('target_id')
contribFrame = contribFrame.astype(float)

# save scatterplots
plt.figure()
plt.subplot(221)
plt.scatter(contribFrame['optScore'], contribFrame['rfF1'], marker='.')
plt.xlabel('Score')
plt.ylabel('RF F1')
pearson = np.round(contribFrame['optScore'].corr(contribFrame['rfF1']), 2)
plt.title(f'Pearson {pearson}')
plt.subplot(222)
plt.scatter(contribFrame['optScore'], contribFrame['rfAUC'], marker='.')
plt.xlabel('Score')
plt.ylabel('RF AUC')
pearson = np.round(contribFrame['optScore'].corr(contribFrame['rfAUC']), 2)
plt.title(f'Pearson {pearson}')
plt.subplot(223)
plt.scatter(contribFrame['optScore'], contribFrame['nnF1'], marker='.')
plt.xlabel('Score')
plt.ylabel('NN F1')
pearson = np.round(contribFrame['optScore'].corr(contribFrame['nnF1']), 2)
plt.title(f'Pearson {pearson}')
plt.subplot(224)
plt.scatter(contribFrame['optScore'], contribFrame['nnAUC'], marker='.')
plt.xlabel('Score')
plt.ylabel('NN AUC')
pearson = np.round(contribFrame['optScore'].corr(contribFrame['nnAUC']), 2)
plt.title(f'Pearson {pearson}')
plt.subplots_adjust(top=0.92, bottom=0.12, left=0.10, right=0.95, hspace=0.25,
                    wspace=0.35)
plt.savefig(dataset + '/' + 'scoreScatter')


# save weighted scatterplots
plt.figure()
plt.subplot(221)
plt.scatter(contribFrame['optScore'].values, contribFrame['rfF1_weighted'], marker='.')
plt.xlabel('Score')
plt.ylabel('RF F1')
plt.subplot(222)
plt.scatter(contribFrame['optScore'], contribFrame['rfAUC_weighted'], marker='.')
plt.xlabel('Score')
plt.ylabel('RF AUC')
plt.subplot(223)
plt.scatter(contribFrame['optScore'],  contribFrame['nnF1_weighted'], marker='.')
plt.xlabel('Score')
plt.ylabel('NN F1')
plt.subplot(224)
plt.scatter(contribFrame['optScore'],  contribFrame['nnAUC_weighted'], marker='.')
plt.xlabel('Score')
plt.ylabel('NN AUC')
plt.subplots_adjust(top=0.92, bottom=0.12, left=0.10, right=0.95, hspace=0.25,
                    wspace=0.35)
plt.savefig(dataset + '/' + 'weightedScoreScatter')




########################################################################################################################
dataset = 'dekois'
prefix = os.getcwd() + '/DataSets/' + dataset + '/'
files = glob(os.getcwd() + '/DataSets/' + dataset + '/*_performanceNew.pkl')
perfs = []
for file in files:
    target_id = file.split('/')[-1].split('_')[0]
    optPackage = pd.read_pickle(prefix + target_id + '_dataPackageNew.pkl')
    features = optPackage.drop(['split', 'labels', 'weights'], axis=1)
    training = optPackage[optPackage['split'] == 1]
    trainingFeatures = training.drop(['split', 'labels', 'weights'], axis=1)
    trainingLabels = training['labels']
    rf = RandomForestClassifier(n_estimators=100)
    rf.fit(trainingFeatures, trainingLabels)
    optPackage['rfPreds'] = rf.predict(features)
    rfPredictions = optPackage[optPackage['split'] == 0]['rfPreds']
    validationLabels = optPackage[optPackage['split'] == 0]['labels']
    rfAUC = roc_auc_score(validationLabels, rfPredictions)
    log = pd.read_pickle(os.getcwd() + '/DataSets/' + dataset + '/' + target_id + '_optRecordNew.pkl')
    optScore = log.tail(1).values[0, 3]
    assert optScore > -0.2, 'low score!'
    performance = pd.read_pickle(file)
    perf = list(np.mean(np.array(performance), axis=0))
    perf.extend([optScore, rfAUC])
    perfs.append(perf)
perfs = pd.DataFrame(perfs)
# save scatterplots
plt.figure()
plt.subplot(311)
plt.scatter(perfs[0], perfs[1])
plt.xlabel('Score')
plt.ylabel('RF AUC')
pearson = np.round(perfs[0].corr(perfs[1]), 2)
plt.title(f'Pearson {pearson}')
plt.subplot(312)
plt.scatter(perfs[0], perfs[2])
plt.xlabel('Score')
plt.ylabel('RF AUC weighted')
pearson = np.round(perfs[0].corr(perfs[2]), 2)
plt.title(f'Pearson {pearson}')
plt.subplot(313)
plt.scatter(perfs[3], perfs[4])
plt.xlabel('Score')
plt.ylabel('RF AUC')
pearson = np.round(perfs[3].corr(perfs[4]), 2)
plt.title(f'Pearson {pearson}')
plt.tight_layout()
plt.savefig(dataset + '/' + 'scoreScatter')


