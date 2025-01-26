from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from rdkit import DataStructs
import numpy as np


# https://github.com/deepchem/deepchem/blob/master/deepchem/feat/molecule_featurizers/circular_fingerprint.py

def smiles_to_fingerprint(smiles_list, radius=2, n_bits=2048):
    fingerprints = []
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=2048)
        fp = mfpgen.GetFingerprint(mol)
        arr = np.zeros((n_bits,))
        DataStructs.ConvertToNumpyArray(fp, arr)
        fingerprints.append(arr)

    return np.array(fingerprints)
