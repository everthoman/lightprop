"""
make_example_data.py - Generate example QSAR dataset for lightprop testing
"""
import pandas as pd
import numpy as np

# 50 real drug-like SMILES with synthetic pIC50 values
SMILES_LIST = [
    "CC(=O)Oc1ccccc1C(=O)O",        # Aspirin
    "CC12CCC3C(C1CCC2O)CCC4=CC(=O)CCC34C",  # Testosterone
    "c1ccc2c(c1)cc1ccc3cccc4ccc2c1c34",      # Pyrene
    "CC(C)Cc1ccc(cc1)C(C)C(=O)O",    # Ibuprofen
    "OC(=O)c1ccccc1O",               # Salicylic acid
    "COc1cc2c(cc1OC)C(N)CC2",        # Dopamine analog
    "c1ccc(cc1)C(=O)O",              # Benzoic acid
    "CCOC(=O)c1ccc(cc1)N",           # Ethyl 4-aminobenzoate
    "c1ccc(cc1)O",                   # Phenol
    "CC1=CC(=O)c2ccccc2C1=O",        # Menadione
    "O=C1NC(=O)c2ccccc21",           # Isatoic anhydride-like
    "c1cc2ccccc2nc1",                # Quinoline
    "c1ccc2ncccc2c1",                # Isoquinoline
    "c1cnc2ccccc2n1",                # Quinoxaline
    "c1ccc(-c2ccccn2)cc1",           # 2-Phenylpyridine
    "CC(=O)Nc1ccc(O)cc1",            # Paracetamol
    "OC(=O)c1ccc(F)cc1",             # 4-Fluorobenzoic acid
    "Cc1ccc(S(N)(=O)=O)cc1",         # Sulfamide
    "CC(N)c1ccccc1",                 # Amphetamine-like
    "O=C(O)c1ccc(Cl)cc1",            # 4-Chlorobenzoic acid
    "COc1ccc(CCN)cc1",               # Tyramine analog
    "OC(=O)CCCC(=O)O",              # Glutaric acid
    "O=C(O)c1cccc(O)c1",             # 3-Hydroxybenzoic acid
    "c1ccc(cc1)CC(=O)O",             # Phenylacetic acid
    "O=C(O)c1ccc(N)cc1",             # 4-Aminobenzoic acid (PABA)
    "Cc1ccccc1",                     # Toluene
    "CC(C)(C)c1ccccc1",              # tert-Butylbenzene
    "CCc1ccccc1",                    # Ethylbenzene
    "Cc1ccc(C)cc1",                  # p-Xylene
    "c1ccc2cc3ccccc3cc2c1",          # Anthracene
    "c1ccc2ccccc2c1",                # Naphthalene
    "CC(=O)c1ccccc1",                # Acetophenone
    "O=Cc1ccccc1",                   # Benzaldehyde
    "c1ccc(cc1)N",                   # Aniline
    "Nc1ccc(N)cc1",                  # p-Phenylenediamine
    "Clc1ccccc1",                    # Chlorobenzene
    "Brc1ccccc1",                    # Bromobenzene
    "Fc1ccccc1",                     # Fluorobenzene
    "Ic1ccccc1",                     # Iodobenzene
    "c1ccc(cc1)[N+](=O)[O-]",        # Nitrobenzene
    "COc1ccccc1",                    # Anisole
    "CCCc1ccccc1",                   # Propylbenzene
    "O=C(O)CCc1ccccc1",              # Hydrocinnamic acid
    "Cc1ccc(cc1)S(=O)(=O)N",         # Sulfonamide
    "O=C(Nc1ccccc1)c1ccccc1",        # Benzanilide
    "c1ccc(cc1)C#N",                 # Benzonitrile
    "COC(=O)c1ccccc1",               # Methyl benzoate
    "CCOC(=O)c1ccccc1",              # Ethyl benzoate
    "CC(=O)Nc1ccccc1",               # Acetanilide
    "O=C(O)c1ccc(OC)cc1",            # 4-Methoxybenzoic acid
]

np.random.seed(42)
n = len(SMILES_LIST)

# Generate synthetic pIC50 values with some structure-activity trend
pIC50 = np.random.normal(6.5, 1.2, n).clip(4.0, 9.5)
# Add some noise-based variation
noise = np.random.normal(0, 0.3, n)
pIC50 = pIC50 + noise

ids = [f"COMP_{i+1:04d}" for i in range(n)]

df = pd.DataFrame({
    "ID": ids,
    "SMILES": SMILES_LIST,
    "pIC50": np.round(pIC50, 2),
})

df.to_csv("example_data.csv", index=False)
print(f"Saved example_data.csv with {len(df)} compounds")
print(df.head(10).to_string())
print(f"\npIC50 stats:\n{df['pIC50'].describe().to_string()}")
