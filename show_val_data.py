import torch
import os
from rdkit.Chem import rdchem

def show_validation_data():
    """Simple script to display validation data information."""
    val_path = "data/standard/val/val_molecules.pt"
    
    if not os.path.exists(val_path):
        print(f"File not found: {val_path}")
        return
    
    # Allow RDKit objects for loading
    torch.serialization.add_safe_globals([rdchem.Mol])
    
    # Load the data
    data = torch.load(val_path, weights_only=False)
    
    print("=== Validation Data Summary ===")
    print(f"File: {val_path}")
    print(f"Type: {type(data)}")
    
    if isinstance(data, dict):
        print("Keys:", list(data.keys()))
        for key, value in data.items():
            if hasattr(value, 'shape'):
                print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
            elif isinstance(value, list):
                print(f"  {key}: list with {len(value)} items")
            else:
                print(f"  {key}: {type(value)} - {value}")
    
    elif hasattr(data, '__len__'):
        print(f"Length: {len(data)}")
        if len(data) > 0:
            print("First item sample:")
            print(f"  Type: {type(data[0])}")
            if hasattr(data[0], 'keys'):
                print(f"  Keys: {list(data[0].keys())}")
            elif hasattr(data[0], '__dict__'):
                print(f"  Attributes: {vars(data[0])}")
    
    print("\n=== Sample Data ===")
    if isinstance(data, dict):
        for key, value in data.items():
            if hasattr(value, 'shape') and value.numel() > 0:
                print(f"{key} sample:")
                print(f"  {value[0]}")
                break
    elif len(data) > 0:
        print(f"Sample item: {data[0]}")

if __name__ == "__main__":
    show_validation_data()