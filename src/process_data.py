import os
import torch
from rdkit import Chem
import pandas as pd
from tqdm import tqdm
import glob

def process_smi_file(input_path, output_dir):
    """
    Processes a .smi file to convert SMILES to molecular graphs and saves them as a PyTorch file.

    Args:
        input_path (str): The path to the input .smi file.
        output_dir (str): The directory to save the processed data.
    """
    print(f"Processing {input_path}...")
    try:
        data = pd.read_csv(input_path, sep=' ', names=['smiles', 'zinc_id'], header=0)
    except pd.errors.EmptyDataError:
        print(f"Warning: {input_path} is empty. Skipping.")
        return

    molecules = []
    for smiles in tqdm(data['smiles'], desc=f"Processing {os.path.basename(input_path)}"):
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            # In a real scenario, you would extract features here.
            # For now, we'll just store the RDKit molecule object.
            molecules.append(mol)
    
    if molecules:
        output_filename = os.path.basename(input_path).replace('.smi', '.pt')
        output_filepath = os.path.join(output_dir, output_filename)
        torch.save(molecules, output_filepath)
        print(f"Successfully processed {len(molecules)} molecules and saved to {output_filepath}")
    else:
        print(f"No valid molecules found in {input_path}")


if __name__ == "__main__":
    input_files = glob.glob('data/**/*.smi', recursive=True)
    output_dir = 'data/processed'
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for smi_file in input_files:
        process_smi_file(smi_file, output_dir)