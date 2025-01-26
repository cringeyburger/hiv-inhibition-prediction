"""Contains preprocessing functions for the dataset"""

import os
import sys
import pandas as pd

current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(current_dir, "../../src")
sys.path.append(src_path)

from utils import data_functions

required_dirs = [
    "data/raw",
    "data/preprocessed",
    "data/processed",
]

for directory in required_dirs:
    if not os.path.exists(directory):
        os.makedirs(directory)

created_dirs = [directory for directory in required_dirs if os.path.exists(directory)]
print(f"***Created/Verified directories: {created_dirs}***")


current_dir = os.getcwd()
RAW_DATASET = os.path.join(current_dir, "data/raw/HIV.csv")
DATASET = os.path.join(current_dir, "data/preprocessed/dataset.csv")
PROCESSED = os.path.join(current_dir, "data/processed")
TRAIN = os.path.join(current_dir, "data/processed/train.csv")
TEST = os.path.join(current_dir, "data/processed/test.csv")

# Merge and rename the activity values
data_functions.refactor_columns(RAW_DATASET, DATASET)

# Split dataset based on Murcko scaffolding
# data_functions.scaffold_split_and_save(DATASET, TRAIN, TEST)

# Split dataset using Deepchem Scaffolding
# data_functions.deepchem_scaffold_split(DATASET, TRAIN, TEST)

data = pd.read_csv(DATASET)
train_data, test_data = data_functions.train_test_split_df(
    data, smiles_column="smiles", frac_train=0.8, frac_test=0.2
)
train_data.to_csv(TRAIN, index=False)
test_data.to_csv(TEST, index=False)


# Get SMOTE and ADASYN datasets
# data_functions.apply_smote_and_adasyn(TRAIN, PROCESSED)
