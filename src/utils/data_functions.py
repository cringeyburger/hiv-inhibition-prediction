"""Contains functions for manipulating data"""

from rdkit import Chem
import pandas as pd
from typing import List, Optional
import numpy as np
from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles


def refactor_columns(data_path, save_path):
    """Modify the 'HIV_active' column based on the values in the 'activity' column,
    and drop the 'activity' column afterward"""
    # This is just a verification step, the dataset already treats CMs and CAs as HIV active
    data = pd.read_csv(data_path)
    data["HIV_active"] = data["activity"].apply(lambda x: 1 if x in ["CM", "CA"] else 0)
    data = data.drop(columns=["activity"])
    data.to_csv(save_path, index=False)

    print(
        "***Updated the HIV_active column based on activity and removed the activity column***"
    )

# NOTE: Deprecating this for the other implementation

# def scaffold_split_and_save(dataset_path, train_path, test_path, train_frac=0.8):
#     """
#     Perform Bemis-Murcko scaffold-based splitting and save the splits to disk
#     """
#     dataset = pd.read_csv(dataset_path)

#     scaffold_dict = defaultdict(list)
#     for idx, row in dataset.iterrows():
#         smiles = row["smiles"]
#         mol = Chem.MolFromSmiles(smiles)
#         if mol:
#             scaffold = MurckoScaffold.GetScaffoldForMol(mol)
#             scaffold_smiles = Chem.MolToSmiles(scaffold)
#             scaffold_dict[scaffold_smiles].append(idx)

#     scaffolds = list(scaffold_dict.keys())
#     random.shuffle(scaffolds)

#     train_cutoff = int(train_frac * len(scaffolds))
#     train_scaffolds = scaffolds[:train_cutoff]
#     test_scaffolds = scaffolds[train_cutoff:]

#     train_indices = [idx for scaf in train_scaffolds for idx in scaffold_dict[scaf]]
#     test_indices = [idx for scaf in test_scaffolds for idx in scaffold_dict[scaf]]

#     train_data = dataset.iloc[train_indices]
#     test_data = dataset.iloc[test_indices]

#     train_data.to_csv(train_path, index=False)
#     test_data.to_csv(test_path, index=False)

#     print(
#         "***Split the dataset into train and test based on Bemis-Murcko Scaffolding***"
#     )


# https://github.com/deepchem/deepchem/blob/master/deepchem/splits/splitters.py
# https://github.com/rdkit/rdkit/blob/master/rdkit/Chem/Scaffolds/MurckoScaffold.py

def _generate_scaffold(smiles: str) -> Optional[str]:
    """Generates a Bemis-Murcko scaffold from a SMILES string."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return MurckoScaffoldSmiles(mol=mol)


def generate_scaffolds_from_df(df, smiles_column, log_every_n=1000) -> List[List[int]]:
    """
    Groups molecules in the DataFrame by their Bemis-Murcko scaffolds.
    """
    scaffolds = {}

    for ind, smiles in enumerate(df[smiles_column]):
        if ind % log_every_n == 0:
            print(f"Processing molecule {ind}/{len(df)}...")
        scaffold = _generate_scaffold(smiles)
        if scaffold:
            scaffolds.setdefault(scaffold, []).append(ind)

    # Sort scaffolds by size (descending) and index (ascending)
    sorted_scaffolds = sorted(
        scaffolds.items(), key=lambda x: (len(x[1]), x[1][0]), reverse=True
    )
    return [sorted(value) for _, value in sorted_scaffolds]


def train_test_split_df(
    df,
    smiles_column,
    frac_train=0.8,
    frac_test=0.2,
    log_every_n=1000,
):
    """
    Splits the DataFrame into training and testing DataFrames
    """
    np.testing.assert_almost_equal(frac_train + frac_test, 1.0, decimal=5)

    scaffold_sets = generate_scaffolds_from_df(df, smiles_column, log_every_n)

    train_cutoff = int(frac_train * len(df))

    train_inds, test_inds = [], []

    for scaffold_set in scaffold_sets:
        if len(train_inds) + len(scaffold_set) > train_cutoff:
            test_inds.extend(scaffold_set)
        else:
            train_inds.extend(scaffold_set)

    train_df = df.iloc[train_inds].reset_index(drop=True)
    test_df = df.iloc[test_inds].reset_index(drop=True)

    return train_df, test_df


# Can use this based on the performance of the models

# def apply_smote_and_adasyn(train_data_path, save_path):
#     """
#     Apply SMOTE and ADASYN to the training data and save the results.
#     """

#     import sys
#     current_dir = os.path.dirname(os.path.abspath(__file__))
#     src_path = os.path.join(current_dir, "../../src")
#     sys.path.append(src_path)
#     from modules.config.ml_dataset import smiles_to_fingerprint
#     data = pd.read_csv(train_data_path)

#     X_train = smiles_to_fingerprint(data["smiles"].tolist())
#     y = data["HIV_active"]
#     del data

#     smote = SMOTE(random_state=42)
#     X_smote, y_smote = smote.fit_resample(X_train, y)

#     smote_save_path = os.path.join(save_path, "smote_data.csv")
#     smote_data = pd.concat(
#         [pd.DataFrame(X_smote), pd.DataFrame(y_smote, columns=["HIV_active"])], axis=1
#     )
#     smote_data.to_csv(smote_save_path, index=False)
#     del smote_data

#     adasyn = ADASYN(random_state=42)
#     X_adasyn, y_adasyn = adasyn.fit_resample(X_train, y)

#     adasyn_save_path = os.path.join(save_path, "adasyn_data.csv")
#     adasyn_data = pd.concat(
#         [pd.DataFrame(X_adasyn), pd.DataFrame(y_adasyn, columns=["HIV_active"])], axis=1
#     )
#     adasyn_data.to_csv(adasyn_save_path, index=False)
#     del adasyn_data

#     return smote_save_path, adasyn_save_path
