import torch
import os
from rdkit.Chem import rdchem
from rdkit import Chem

def show_top_molecules(num_molecules=10):
    """Display the top N molecules as SMILES strings."""
    val_path = "data/standard/val/val_molecules.pt"
    
    if not os.path.exists(val_path):
        print(f"File not found: {val_path}")
        return
    
    # Allow RDKit objects for loading
    torch.serialization.add_safe_globals([rdchem.Mol])
    
    # Load the data
    molecules = torch.load(val_path, weights_only=False)
    
    print(f"=== Top {num_molecules} Validation Molecules ===")
    print(f"Total molecules in validation set: {len(molecules)}")
    print()
    
    for i, mol in enumerate(molecules[:num_molecules]):
        smiles = Chem.MolToSmiles(mol)
        print(f"{i+1:2d}. {smiles}")
    
    # Also show from the SMILES file for comparison
    smiles_file = "data/standard/val/val_smiles.txt"
    if os.path.exists(smiles_file):
        print(f"\n=== From {smiles_file} ===")
        with open(smiles_file, 'r') as f:
            smiles_lines = f.readlines()
            for i, line in enumerate(smiles_lines[:num_molecules]):
                print(f"{i+1:2d}. {line.strip()}")

if __name__ == "__main__":
    show_top_molecules(10)