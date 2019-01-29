
from rdkit import Chem 
from rdkit.Chem import AllChem
import numpy as np

######################################################


class ligand:
    '''class for producing ligands from their SMILES string'''
    def __init__(self, smilesString):
        # Create rdkit molecule from valid SMILES
        self.SMILES = smilesString
        if type(smilesString) != str:
            raise Exception('Input should be string')
        try:
            self.mol = Chem.MolFromSmiles(smilesString)
        except:
            raise Exception('Invalid SMILES')
        # embed molecule in 3D (rdkit recommends adding/removing hydrogens)
        if self.mol is not None:
            Chem.AddHs(self.mol)

    def fingerprint(self, radius = 2):
        '''Method for computing Morgan fingerprint.
        With default radius of 2 it is comperable to ECFP4 (rdkit).'''
        #Create Bit Vector of fingerprint
        if type(radius) != int:
            raise Exception('radius argument should be positive int')
        if radius < 1:
            raise Exception('radius argument should be positive int')
        F = Chem.rdMolDescriptors.GetMorganFingerprintAsBitVect(self.mol,
                                                                radius)
        # Return list of ints 0/1
        return tuple([int(f) for f in F.ToBitString()])
        