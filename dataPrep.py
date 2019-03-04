import os
import sys
import warnings
import glob
import gzip
import pandas as pd
import numpy as np

from scoop import futures

from rdkit import Chem
from rdkit.Chem import AllChem

from sklearn.metrics.pairwise import pairwise_distances

import dataBias
###############################################################################
#Set parameters
parallel = True
sample = True
timeLimit = 60 #(60) seconds
sizeBound = 15100
ratio = 0.8

def finger(mol):
    #fPrint = FingerprintMols.FingerprintMol(mol)
    fprint = AllChem.GetMorganFingerprintAsBitVect( mol, 2 )
    return list(fprint)

###############################################################################

def makePrints(s):
    try:
        inf = gzip.open(s)
        gzsuppl = Chem.ForwardSDMolSupplier(inf)
        mols = [x for x in gzsuppl if x is not None]
        prints = futures.map(finger, mols)
        #prints = [finger(mol) for mol in mols]
        prints = pd.DataFrame(prints).dropna()
        return prints
    except:
        print('Unable to open...')
        return

###############################################################################

def main(s):
    dataset = s
    if dataset == 'dekois':
        active_suffix = '.sdf.gz'
        decoys_suffix = '_Celling-v1.12_decoyset.sdf.gz'
        files = glob.glob(dataset + '/decoys/*.sdf.gz')
        targets = sorted(list(set([f.split('_')[0].split('/')[-1] for f in files])))
    elif dataset == 'DUDE':
        active_suffix = '/actives_final.sdf.gz'
        decoys_suffix = '/decoys_final.sdf.gz'
        files = glob.glob(dataset + '/*')
        targets = sorted(list(set([f.split('/')[1].split('_')[0] for f in files])))
    elif dataset == 'MUV':
        active_suffix = '_actives.sdf.gz'
        decoys_suffix = '_decoys.sdf.gz'
        files = glob.glob(dataset + '/*.sdf.gz')
        targets = sorted(list(set([f.split('_')[0].split('/')[-1] for f in files])))
    else:
        print('Invalid dataset specified. Did you mean MUV, dekois, or DUDE?')
        return

    toCompute = len(targets)
    t=0
    skipFiles = glob.glob(dataset + '/*distances*')

    for target_id in targets[1:]:
        t+=1
        print('Current target: {} ({} of {})'.format(target_id, t, toCompute))
        prefix = os.getcwd() + '/' + dataset + '/'
        picklePrintName = prefix + target_id + '_unsplitDataFrame.pkl'
        pickleDistName = prefix + target_id + '_distances.pkl'
        distanceFileName = dataset + '/' + target_id + '_distances.pkl'

        if distanceFileName not in skipFiles:
            try:
                if dataset == 'MUV':
                    print('Computing decoy fingerprints...')
                    decoyPrints = makePrints(prefix + target_id + decoys_suffix)
                    print('Computing active fingerprints...')
                    activePrints = makePrints(prefix + target_id + active_suffix)

                elif dataset == 'dekois':
                    print('Computing decoy fingerprints...')
                    decoyPrints = makePrints(prefix + 'decoys/' + target_id + decoys_suffix)
                    print('Computing active fingerprints...')
                    activePrints = makePrints(prefix + 'ligands/' + target_id + active_suffix)

                elif dataset == 'DUDE':
                    print('Computing decoy fingerprints...')
                    decoyPrints = makePrints(prefix + target_id + decoys_suffix)
                    print('Computing active fingerprints...')
                    activePrints = makePrints(prefix + target_id + active_suffix)

                activePrints['Labels'] = int(1)
                decoyPrints['Labels'] = int(0)

                size = decoyPrints.shape[0] + activePrints.shape[0]
                if size > sizeBound:
                    print('{} dataset too big: {}'.format(target_id, size))
                    continue
                fingerprints = activePrints.append(decoyPrints, ignore_index=True)
                fingerprints.to_pickle(picklePrintName)
                print('Saved: ' + picklePrintName)
                print('Computing distance matrix...')
                #Compute distance matrix (Jaccard)
                with warnings.catch_warnings():
                    #Suppress warning from distance matrix computation (int->bool)
                    warnings.simplefilter("ignore")
                    if parallel == True:
                        distanceMatrix = pairwise_distances(
                                            fingerprints.drop('Labels', axis=1),
                                            metric='jaccard',
                                            n_jobs=-1)
                    else:
                        distanceMatrix = pairwise_distances(
                                            fingerprints.drop('Labels', axis=1),
                                            metric='jaccard')
                    pd.DataFrame(distanceMatrix).to_pickle(pickleDistName)
                    print('Saved' + pickleDistName)
            except:
                pass
        else:
            print('Reading data...')
            fingerprints = pd.read_pickle(picklePrintName)
            distanceMatrix = pd.read_pickle(pickleDistName).values

        if sample == True:
            pickleSamplesName = prefix + target_id + '_samples.pkl'
            data = dataBias.data_set(distanceMatrix,
                                     fingerprints['Labels'])
            print('Sampling...')
            with warnings.catch_warnings():
                    #Suppress warning from undefined bias
                    warnings.simplefilter("ignore")
                    data.sample(timeLimit, 0.8)
            trainingSizes = [np.sum(x) for x in data.splits]
            validationSizes = [(data.size - x) for x in trainingSizes]
            trainingRatios = [float(x)/data.size for x in trainingSizes]
            trainingPosSizes = [np.sum(data.labels & x) for x in data.splits]
            numPos = np.sum(data.labels)
            validationPosSizes = [numPos - x for x in trainingPosSizes]
            validPosEquity= [(data.size*x)/(numPos*y) for x, y in zip(validationPosSizes, validationSizes)]
            exp = pd.DataFrame([data.bias_samples, trainingRatios, validPosEquity]).T
            exp.columns = ['bias', 'training ratio', 'validation pos equity']
            pd.DataFrame(exp).to_pickle(pickleSamplesName)
            print('Saved: ' + pickleSamplesName)

if __name__ == '__main__':
    if len(sys.argv)>1:
        main(sys.argv[1])
    else:
        print('No data set specified...')

