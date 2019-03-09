import os
import sys
import warnings
import glob
import gzip
import psutil

import pandas as pd
import numpy as np

from rdkit import Chem
from rdkit.Chem import AllChem

from sklearn.metrics.pairwise import pairwise_distances
#######################################################

mem = psutil.virtual_memory()
safetyFactor = 3  # (3)
sizeBound = int(np.sqrt(mem.available / 8)/safetyFactor)
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

    # TODO create fingerprints
    # TODO attempt distance matrix
    # TODO create data_set
    # TODO run optimizer



if __name__ == '__main__':
    if len(sys.argv) > 2:
        main(sys.argv[1], sys.argv[2])
    else:
        print("Specify dataset and target...")
