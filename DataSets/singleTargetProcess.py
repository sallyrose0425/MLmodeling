import os
import sys
import warnings
import gzip
import psutil

import pandas as pd
import numpy as np

from rdkit import Chem
from rdkit.Chem import AllChem

from sklearn.metrics import pairwise_distances

import ukyScore

ATOMWISE = False  # (False) Use the atomwise approximation
metric = 'jaccard'  # ('jaccard') Metric for use in determining fingerprint distances
score_goal = 0.02  # (0.02) Early termination of genetic optimizer if goal is reached
numGens = 1000  # (1000) Number of generations to run in genetic optimizer

print_frequency = 100  # (100) How many generations of optimizer before printing update
safetyFactor = 3  # (3) Fraction of avaliable RAM to use for distance matrix computation

targetRatio = 0.8
ratioTol = 0.01
balanceTol = 0.05


mem = psutil.virtual_memory()
sizeBound = int(np.sqrt(mem.available / 8)/safetyFactor)
# sizeBound = 15100
"""sizeBound: max size of dataset that reliably
 fits distance matrix in user's computer's memory."""


def finger(mol):
    fprint = AllChem.GetMorganFingerprintAsBitVect(mol, 2)
    return list(fprint)


def makePrints(s):
    try:
        inf = gzip.open(s)
        gzsuppl = Chem.ForwardSDMolSupplier(inf)
        mols = [x for x in gzsuppl if x is not None]
        prints = [finger(mol) for mol in mols]
        prints = pd.DataFrame(prints).dropna()
        return prints
    except:
        print('Unable to open...')
        return


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
    decoyPrints = makePrints(decoyFile)
    activePrints = makePrints(activeFile)
    activePrints['Labels'] = int(1)
    decoyPrints['Labels'] = int(0)
    fingerprints = activePrints.append(decoyPrints, ignore_index=True)

    size = fingerprints.shape[0]
    if size > sizeBound:
        distanceMatrix = np.array([])
    else:
        with warnings.catch_warnings():
            # Suppress warning from distance matrix computation (int->bool)
            warnings.simplefilter("ignore")
            distanceMatrix = pairwise_distances(fingerprints.drop('Labels', axis=1), metric=Metric)
    data = ukyScore.data_set(distanceMatrix, fingerprints, targetRatio, ratioTol,
                             balanceTol, atomwise=ATOMWISE, Metric=metric)
    splits = data.geneticOptimizer(numGens, printFreq=print_frequency, scoreGoal=score_goal)
    scores = [data.computeScore(split) for split in splits]
    split = splits[np.argmin(scores)]
    fingerprints['split'] = split
    pd.to_pickle(fingerprints, prefix + target_id + '_dataPackage.pkl')


if __name__ == '__main__':
    if len(sys.argv) > 2:
        main(sys.argv[1], sys.argv[2])
    else:
        print("Specify dataset and target...")
