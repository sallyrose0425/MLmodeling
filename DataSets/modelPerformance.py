import os
import sys
from glob import glob

import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score

from DataSets.ukyScore import data_set

###############################################################################

def main(dataset):
        prefix = os.getcwd() + '/' + dataset + '/'
        files = glob(prefix + '*_optSplit.pkl')
        targets = sorted(list(set([f.split('_')[0].split('/')[-1] for f in files])))
        acumPerf = []
        for target_id in targets:
            picklePrintName = prefix + target_id + '_unsplitDataFrame.pkl'
            fPrints = pd.read_pickle(picklePrintName)
            pickleDistName = prefix + target_id + '_distances.pkl'
            distances = pd.read_pickle(pickleDistName)
            data = data_set(distances, fPrints)
            pickleSplitName = prefix + target_id + '_optSplit.pkl'
            split = pd.read_pickle(pickleSplitName).values.flatten()
            splitScore = split[-1]
            split = split[:-1].astype(int)
            data.splitData(split)
            clf = RandomForestClassifier(n_estimators=100)
            clf.fit(data.trainingFeatures, data.trainingLabels)
            predictions = clf.predict(data.validationFeatures)
            nNeighborPred = data.nearestNeighborPredictions(split)
            scoreAUC = roc_auc_score(data.validationLabels, predictions)
            nnAUC = roc_auc_score(data.validationLabels, nNeighborPred)
            if sum(predictions) > 0:
                scoreF1 = f1_score(data.validationLabels, predictions)
            else:
                scoreF1 = 0.0
            if sum(nNeighborPred) > 0:
                nnF1 = f1_score(data.validationLabels, nNeighborPred)
            else:
                nnF1 = 0.0
            acumPerf.append((target_id, splitScore, scoreAUC, nnAUC, scoreF1, nnF1))
            print('Target {} finished'.format(target_id))
        Perf = pd.DataFrame(acumPerf, columns=['target','split score', 'AUC',
                                        'nnAUC', 'F1', 'nnF1'])
        Perf.to_pickle(prefix + 'performance.pkl')


if __name__ == '__main__':
    if len(sys.argv)>1:
        main(sys.argv[1])
    else:
        print('No data set specified')


