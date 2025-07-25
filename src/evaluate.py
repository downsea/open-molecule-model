import torch
import torch.nn.functional as F
import numpy as np
from torch_geometric.loader import DataLoader
from rdkit import Chem
from rdkit.Chem import Descriptors, QED
import selfies as sf
import os
import argparse
from tqdm import tqdm

from .data_loader import ZINC_Dataset
from .model import PanGuDrugModel
from .utils import SELFIESProcessor, create_condition_vector
from .config import Config

class PanGuEvaluator:
    def __init__(self, config=None):
        """Initialize evaluator with configuration."""
        if config is None:
            config = Config.from_args()
        
        self.config = config
        self.device = torch.device(
            config.evaluation.device if torch.cuda.is_available() else "cpu"
        )
        
        # Initialize components
        self.selfies_processor = SELFIESProcessor()
        self.config.model.output_dim = self.selfies_processor.get_vocab_size()
        
        # Load model
        self.model = self.load_model()
    
    def load_model(self):
        """Load trained model from checkpoint."""
        model = PanGuDrugModel(
            num_node_features=self.config.model.num_node_features,
            hidden_dim=self.config.model.hidden_dim,
            num_encoder_layers=self.config.model.num_encoder_layers,
            num_encoder_heads=self.config.model.num_encoder_heads,
            output_dim=self.config.model.output_dim,
            num_decoder_heads=self.config.model.num_decoder_heads,
            num_decoder_layers=self.config.model.num_decoder_layers,
            latent_dim=self.config.model.latent_dim,
            num_selected_layers=self.config.model.num_selected_layers
        ).to(self.device)
        
        if os.path.exists(self.config.paths.checkpoint_path):
            checkpoint = torch.load(
                self.config.paths.checkpoint_path, 
                map_location=self.device
            )
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded model from {self.config.paths.checkpoint_path}")
        else:
            print(f"Warning: No checkpoint found at {self.config.paths.checkpoint_path}. Using random weights.")
        
        model.eval()
        return model
    
    def encode_molecule(self, data):
        """Encode a molecular graph to latent space."""
        with torch.no_grad():
            data = data.to(self.device)
            
            # Get encoder output
            encoded_features = self.model.encoder(data)
            
            # Get latent parameters
            mean = self.model.fc_mean(encoded_features)
            log_var = self.model.fc_log_var(encoded_features)
            
            # Sample latent vector
            z = self.model.reparameterize(mean, log_var)
            
            return z, mean, log_var
    
    def decode_molecule(self, z, max_length=None):
        """Decode a latent vector to SELFIES string."""
        if max_length is None:
            max_length = self.config.data.max_length
            
        with torch.no_grad():
            batch_size = z.size(0)
            
            # Create condition vector (placeholder)
            condition_vector = create_condition_vector(
                batch_size, 
                self.config.model.latent_dim, 
                self.device
            )
            
            # Start with SOS token
            sos_token = torch.tensor([1], device=self.device)  # <sos>
            tgt_sequence = sos_token.unsqueeze(0).repeat(batch_size, 1)
            
            # Generate sequence greedily
            for i in range(max_length - 1):
                # One-hot encode current sequence
                tgt_one_hot = F.one_hot(
                    tgt_sequence, 
                    num_classes=self.config.model.output_dim
                ).float()
                tgt_one_hot = tgt_one_hot.transpose(1, 2)  # (batch, vocab, seq_len)
                
                # Forward pass
                output, _, _ = self.model(
                    torch.zeros(batch_size, 6, device=self.device),  # Dummy graph data
                    condition_vector,
                    tgt_one_hot
                )
                
                # Get next token probabilities
                next_token_logits = output[:, :, -1]  # Last position
                next_token = next_token_logits.argmax(dim=1)
                
                # Append to sequence
                tgt_sequence = torch.cat([tgt_sequence, next_token.unsqueeze(1)], dim=1)
                
                # Check for EOS token
                if (next_token == 2).all():  # <eos>
                    break
            
            return tgt_sequence
    
    def evaluate_reconstruction(self, dataset):
        """Evaluate reconstruction accuracy on dataset."""
        dataloader = DataLoader(
            dataset, 
            batch_size=self.config.data.batch_size, 
            shuffle=False,
            num_workers=self.config.system.num_workers
        )
        
        total_correct = 0
        total_tokens = 0
        exact_matches = 0
        valid_molecules = 0
        
        print("Evaluating reconstruction...")
        with torch.no_grad():
            for data in tqdm(dataloader):
                data = data.to(self.device)
                batch_size = data.num_graphs
                
                # Get original SELFIES
                original_selfies = [
                    sf.encoder(smiles) 
                    for smiles in data.smiles
                ]
                original_tensors = self.selfies_processor.encode_batch(
                    original_selfies, 
                    max_length=self.config.data.max_length
                )
                original_tensors = original_tensors.to(self.device)
                
                # Encode and decode
                z, mean, log_var = self.encode_molecule(data)
                reconstructed_tensors = self.decode_molecule(z)
                
                # Compare sequences
                for orig, recon in zip(original_tensors, reconstructed_tensors):
                    orig_str = self.selfies_processor.tensor_to_selfies(orig)
                    recon_str = self.selfies_processor.tensor_to_selfies(recon)
                    
                    # Exact match
                    if orig_str == recon_str:
                        exact_matches += 1
                    
                    # Valid molecule check
                    try:
                        mol = Chem.MolFromSmiles(sf.decoder(recon_str))
                        if mol is not None:
                            valid_molecules += 1
                    except:
                        pass
        
        total_samples = len(dataset)
        
        results = {
            'exact_match_rate': exact_matches / total_samples,
            'valid_molecule_rate': valid_molecules / total_samples,
        }
        
        return results
    
    def generate_molecules(self, num_samples=None):
        """Generate new molecules from random latent vectors."""
        if num_samples is None:
            num_samples = self.config.evaluation.num_samples
            
        print(f"Generating {num_samples} molecules...")
        generated_molecules = []
        valid_molecules = 0
        
        with torch.no_grad():
            for i in tqdm(range(num_samples)):
                # Sample from standard normal distribution
                z = torch.randn(1, self.config.model.latent_dim, device=self.device)
                
                # Decode to SELFIES
                sequence = self.decode_molecule(z)
                selfies_str = self.selfies_processor.tensor_to_selfies(sequence[0])
                
                # Convert to SMILES and check validity
                try:
                    smiles = sf.decoder(selfies_str)
                    mol = Chem.MolFromSmiles(smiles)
                    
                    if mol is not None:
                        valid_molecules += 1
                        
                        # Calculate properties
                        mw = Descriptors.MolWt(mol)
                        logp = Descriptors.MolLogP(mol)
                        qed = QED.qed(mol)
                        
                        generated_molecules.append({
                            'selfies': selfies_str,
                            'smiles': smiles,
                            'molecular_weight': mw,
                            'logp': logp,
                            'qed': qed
                        })
                        
                except Exception as e:
                    continue
        
        print(f"Generated {len(generated_molecules)} valid molecules out of {num_samples}")
        return generated_molecules
    
    def analyze_latent_space(self, dataset):
        """Analyze properties of the latent space."""
        dataloader = DataLoader(
            dataset, 
            batch_size=self.config.data.batch_size, 
            shuffle=False,
            num_workers=self.config.system.num_workers
        )
        
        latent_vectors = []
        properties = []
        
        print("Analyzing latent space...")
        with torch.no_grad():
            for data in tqdm(dataloader):
                data = data.to(self.device)
                
                # Encode to latent space
                z, mean, log_var = self.encode_molecule(data)
                latent_vectors.append(mean.cpu().numpy())
                
                # Get molecular properties
                for smiles in data.smiles:
                    mol = Chem.MolFromSmiles(smiles)
                    if mol is not None:
                        prop = {
                            'molecular_weight': Descriptors.MolWt(mol),
                            'logp': Descriptors.MolLogP(mol),
                            'qed': QED.qed(mol),
                            'num_atoms': mol.GetNumAtoms(),
                            'num_rings': Descriptors.RingCount(mol)
                        }
                        properties.append(prop)
        
        # Combine results
        latent_vectors = np.concatenate(latent_vectors, axis=0)
        
        return {
            'latent_vectors': latent_vectors,
            'properties': properties
        }

def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description='Evaluate PanGu Drug Model')
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--checkpoint', type=str, 
                        help='Override checkpoint path')
    parser.add_argument('--dataset', type=str,
                        help='Override dataset path')
    parser.add_argument('--mode', type=str, choices=['reconstruction', 'generation', 'latent_space'],
                        default='reconstruction', help='Evaluation mode')
    parser.add_argument('--num-samples', type=int,
                        help='Number of samples to generate (for generation mode)')
    parser.add_argument('--device', type=str,
                        help='Device to use')
    
    args = parser.parse_args()
    
    # Load configuration
    config = Config.from_yaml(args.config)
    
    # Override with command line arguments
    if args.checkpoint is not None:
        config.paths.checkpoint_path = args.checkpoint
    if args.dataset is not None:
        config.data.dataset_path = args.dataset
    if args.num_samples is not None:
        config.evaluation.num_samples = args.num_samples
    if args.device is not None:
        config.system.device = args.device
        config.evaluation.device = args.device
    
    # Initialize evaluator
    evaluator = PanGuEvaluator(config)
    
    # Load dataset for evaluation
    dataset = ZINC_Dataset(config.data.dataset_path)
    print(f"Loaded dataset with {len(dataset)} molecules")
    
    # Run evaluation based on mode
    if args.mode == 'reconstruction':
        results = evaluator.evaluate_reconstruction(dataset)
        print("Reconstruction Results:")
        for key, value in results.items():
            print(f"  {key}: {value:.4f}")
    
    elif args.mode == 'generation':
        molecules = evaluator.generate_molecules()
        print(f"\nGenerated {len(molecules)} valid molecules")
        
        # Print some examples
        print("\nExample generated molecules:")
        for i, mol in enumerate(molecules[:5]):
            print(f"  {i+1}. SMILES: {mol['smiles']}")
            print(f"      MW: {mol['molecular_weight']:.2f}, LogP: {mol['logp']:.2f}, QED: {mol['qed']:.3f}")
    
    elif args.mode == 'latent_space':
        results = evaluator.analyze_latent_space(dataset)
        print(f"Analyzed {len(results['latent_vectors'])} molecules in latent space")
        print(f"Latent vectors shape: {results['latent_vectors'].shape}")

if __name__ == "__main__":
    main()