import pandas as pd
import numpy as np
from rdkit import Chem 
#from rdkit.Chem import AllChem


from sklearn.metrics.pairwise import pairwise_distances

###############################################################################
#Read SMILES strings and produce fingerprints
#cd Desktop/bias_tmp

def fPrintFromMols(mol):
    '''Function for computing Morgan fingerprint.
    With default radius of 2 it is comperable to ECFP4 (rdkit).'''
    try:
        fingerPrint = Chem.rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, 2)
        return np.array(fingerPrint)
    except:
        return 'Bad Mol from SMILES'
def molFromSmiles(s):
    try:
        mol = Chem.MolFromSmiles(s)
        if mol==None:
            return 'Bad SMILES (None-type)'
        else:
            return mol
    except:
        return 'Bad SMILES'
        
smilesDF = pd.read_csv('Smiles.csv', delimiter=',')
smilesDF['1']=smilesDF['0']
smilesDF = smilesDF.set_index('1')['0']
Mols = smilesDF.apply(molFromSmiles)
badMols = Mols[Mols.apply(lambda x: type(x)==str)]
Mols = Mols.drop(badMols.index)
fingerPrints = Mols.apply(fPrintFromMols)
fingerPrints = pd.DataFrame.from_items(
        zip(fingerPrints.index, fingerPrints.values)).T

###############################################################################
#Write/read fingerprints to .csv

fingerPrints.to_csv('fingerPrints.csv')
badMols.to_csv('badSmiles.csv', header=True)

fingerPrints = pd.read_csv('fingerPrints.csv', index_col=0)

###############################################################################
#Compute distance matrix (Jaccard)

parallel = True

#Jaccard
if parallel == True:
    distanceMatrix = pairwise_distances(fingerPrints,
                                               metric='jaccard',
                                               n_jobs=-1)
else:
    distanceMatrix = pairwise_distances(fingerPrints,
                                               metric='jaccard')
#Write/read distance matrix
pd.DataFrame(distanceMatrix).to_csv('distanceMatrix.csv')
distanceMatrix = pd.DataFrame.from_csv('distanceMatrix.csv').values

###############################################################################
#Read labels (1/active, 0/inactive)

#Generate example labels
labels = np.random.randint(0, 2, size=2377)
pd.DataFrame(labels).to_csv('labels.csv')

#Load labels
Labels = pd.read_csv('labels.csv', index_col=0).values.flatten()

###############################################################################




