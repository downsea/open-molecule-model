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
        print(f"Using device: {self.device}")
        
        # Initialize components - use train dataset path for consistent vocabulary
        vocab_file = os.path.join(self.config.data.train_dataset_path, "selfies_vocab.json")
        if os.path.exists(vocab_file):
            self.selfies_processor = SELFIESProcessor(vocab_file=vocab_file)
        else:
            self.selfies_processor = SELFIESProcessor()
        self.config.model.output_dim = self.selfies_processor.get_vocab_size()
        
        # Pre-cache dummy graphs for decoder
        self._dummy_graph_cache = {}
        
        # Load model
        self.model = self.load_model()
        
        # Enable optimizations if available
        if hasattr(torch, 'compile') and getattr(self.config, 'performance', {}).get('torch_compile', False):
            print("Compiling model for better performance...")
            self.model = torch.compile(self.model)
    
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
            num_selected_layers=self.config.model.num_selected_layers,
            max_seq_length=self.config.data.max_length
        ).to(self.device)
        
        if os.path.exists(self.config.paths.checkpoint_path):
            checkpoint = torch.load(
                self.config.paths.checkpoint_path, 
                map_location=self.device,
                weights_only=False
            )
            
            # Verify model configuration matches checkpoint
            if 'model_config' in checkpoint:
                saved_config = checkpoint['model_config']
                current_config = {
                    'num_node_features': self.config.model.num_node_features,
                    'hidden_dim': self.config.model.hidden_dim,
                    'num_encoder_layers': self.config.model.num_encoder_layers,
                    'num_selected_layers': self.config.model.num_selected_layers,
                    'output_dim': self.config.model.output_dim
                }
                
                for key, value in current_config.items():
                    if key in saved_config and saved_config[key] != value:
                        print(f"Warning: Config mismatch for {key}: saved={saved_config[key]}, current={value}")
                        # Update current config to match saved model
                        setattr(self.config.model, key, saved_config[key])
                
                # Recreate model with correct configuration
                # The saved model was trained with max_length=128 based on pos_encoder shape
                model = PanGuDrugModel(
                    num_node_features=saved_config['num_node_features'],
                    hidden_dim=saved_config['hidden_dim'],
                    num_encoder_layers=saved_config['num_encoder_layers'],
                    num_encoder_heads=saved_config['num_encoder_heads'],
                    output_dim=saved_config['output_dim'],
                    num_decoder_heads=saved_config['num_decoder_heads'],
                    num_decoder_layers=saved_config['num_decoder_layers'],
                    latent_dim=saved_config['latent_dim'],
                    num_selected_layers=saved_config['num_selected_layers'],
                    use_gradient_checkpointing=saved_config.get('use_gradient_checkpointing', False),
                    max_seq_length=128  # Fixed value based on checkpoint pos_encoder shape
                ).to(self.device)
            
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
    
    def _get_dummy_batch(self, batch_size):
        """Get cached dummy batch or create new one."""
        if batch_size not in self._dummy_graph_cache:
            from torch_geometric.data import Data, Batch
            
            if batch_size == 1:
                # Single node graph
                dummy_data = Data(
                    x=torch.ones(1, 6, device=self.device),
                    edge_index=torch.empty(2, 0, dtype=torch.long, device=self.device),
                )
                self._dummy_graph_cache[batch_size] = dummy_data
            else:
                # Multiple single-node graphs
                dummy_graphs = []
                for _ in range(batch_size):
                    dummy_data = Data(
                        x=torch.ones(1, 6, device=self.device),
                        edge_index=torch.empty(2, 0, dtype=torch.long, device=self.device),
                    )
                    dummy_graphs.append(dummy_data)
                self._dummy_graph_cache[batch_size] = Batch.from_data_list(dummy_graphs)
        
        return self._dummy_graph_cache[batch_size]
    
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
            sos_token = torch.tensor([1], device=self.device)
            tgt_sequence = sos_token.unsqueeze(0).repeat(batch_size, 1)
            
            # Pre-compute sequences in larger steps for efficiency
            eos_token = torch.tensor([2], device=self.device)
            
            # Generate sequence greedily
            for i in range(max_length - 1):
                # One-hot encode current sequence
                tgt_one_hot = F.one_hot(
                    tgt_sequence, 
                    num_classes=self.config.model.output_dim
                ).float()
                tgt_one_hot = tgt_one_hot.transpose(1, 2)
                
                # Use cached dummy batch
                dummy_batch = self._get_dummy_batch(batch_size)
                
                # Forward pass
                output, _, _ = self.model(
                    dummy_batch,
                    condition_vector,
                    tgt_one_hot
                )
                
                # Get next token probabilities
                next_token_logits = output[:, :, -1]
                next_token = next_token_logits.argmax(dim=1)
                
                # Append to sequence
                tgt_sequence = torch.cat([tgt_sequence, next_token.unsqueeze(1)], dim=1)
                
                # Check for EOS token (early termination)
                if (next_token == eos_token.item()).all():
                    break
            
            return tgt_sequence
    
    def evaluate_reconstruction(self, dataset):
        """Evaluate reconstruction accuracy on dataset."""
        # Use larger batch size for evaluation if specified
        eval_batch_size = getattr(self.config.evaluation, 'batch_size', self.config.data.batch_size * 2)
        dataloader = DataLoader(
            dataset, 
            batch_size=eval_batch_size, 
            shuffle=False,
            num_workers=self.config.system.num_workers,
            pin_memory=True,
            prefetch_factor=2
        )
        
        total_correct = 0
        total_tokens = 0
        exact_matches = 0
        valid_molecules = 0
        
        print("Evaluating reconstruction...")
        
        # Enable autocast for mixed precision if supported
        use_amp = getattr(self.config.system, 'mixed_precision', False) and self.device.type == 'cuda'
        scaler = torch.cuda.amp.GradScaler() if use_amp else None
        
        with torch.no_grad():
            if use_amp:
                torch.backends.cudnn.benchmark = True  # Optimize for consistent input sizes
            for data in tqdm(dataloader):
                data = data.to(self.device, non_blocking=True)
                batch_size = data.num_graphs
                
                # Use autocast context for mixed precision
                if use_amp:
                    with torch.cuda.amp.autocast():
                        z, mean, log_var = self.encode_molecule(data)
                        reconstructed_tensors = self.decode_molecule(z)
                else:
                    z, mean, log_var = self.encode_molecule(data)
                    reconstructed_tensors = self.decode_molecule(z)
                
                # Get original SELFIES from data objects
                original_selfies = [item.selfies if hasattr(item, 'selfies') and item.selfies is not None else "[C]" 
                                  for item in data.to_data_list()]
                original_tensors = self.selfies_processor.encode_batch(
                    original_selfies, 
                    max_length=self.config.data.max_length
                )
                original_tensors = original_tensors.to(self.device, non_blocking=True)
                
                # Note: Encoding/decoding moved above with AMP support
                
                # Batch comparison for efficiency
                orig_strs = [self.selfies_processor.tensor_to_selfies(orig) for orig in original_tensors]
                recon_strs = [self.selfies_processor.tensor_to_selfies(recon) for recon in reconstructed_tensors]
                
                # Count exact matches
                matches = sum(1 for orig, recon in zip(orig_strs, recon_strs) if orig == recon)
                exact_matches += matches
                
                # Parallel validity check
                valid_count = 0
                for recon_str in recon_strs:
                    try:
                        mol = Chem.MolFromSmiles(sf.decoder(recon_str))
                        if mol is not None:
                            valid_count += 1
                    except:
                        pass
                valid_molecules += valid_count
        
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
        
        # Generate in batches for better GPU utilization
        generation_batch_size = min(32, num_samples)  # Batch generation
        use_amp = getattr(self.config.system, 'mixed_precision', False) and self.device.type == 'cuda'
        
        with torch.no_grad():
            for batch_start in tqdm(range(0, num_samples, generation_batch_size)):
                batch_end = min(batch_start + generation_batch_size, num_samples)
                current_batch_size = batch_end - batch_start
                
                # Sample batch of latent vectors
                z = torch.randn(current_batch_size, self.config.model.latent_dim, device=self.device)
                
                # Decode batch to SELFIES
                if use_amp:
                    with torch.cuda.amp.autocast():
                        sequences = self.decode_molecule(z)
                else:
                    sequences = self.decode_molecule(z)
                
                # Process batch results
                for seq in sequences:
                    selfies_str = self.selfies_processor.tensor_to_selfies(seq)
                
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
        eval_batch_size = getattr(self.config.evaluation, 'batch_size', self.config.data.batch_size * 2)
        dataloader = DataLoader(
            dataset, 
            batch_size=eval_batch_size, 
            shuffle=False,
            num_workers=self.config.system.num_workers,
            pin_memory=True,
            prefetch_factor=2
        )
        
        latent_vectors = []
        properties = []
        
        print("Analyzing latent space...")
        
        # Enable mixed precision for GPU acceleration
        use_amp = getattr(self.config.system, 'mixed_precision', False) and self.device.type == 'cuda'
        
        with torch.no_grad():
            for data in tqdm(dataloader):
                data = data.to(self.device, non_blocking=True)
                
                # Encode to latent space with optional mixed precision
                if use_amp:
                    with torch.cuda.amp.autocast():
                        z, mean, log_var = self.encode_molecule(data)
                else:
                    z, mean, log_var = self.encode_molecule(data)
                
                # Move to CPU asynchronously
                latent_vectors.append(mean.cpu().numpy())
                
                # Batch molecular property calculation
                batch_props = []
                for smiles in data.smiles:
                    mol = Chem.MolFromSmiles(smiles)
                    if mol is not None:
                        # Calculate properties in batch for efficiency
                        prop = {
                            'molecular_weight': Descriptors.MolWt(mol),
                            'logp': Descriptors.MolLogP(mol),
                            'qed': QED.qed(mol),
                            'num_atoms': mol.GetNumAtoms(),
                            'num_rings': Descriptors.RingCount(mol)
                        }
                        batch_props.append(prop)
                properties.extend(batch_props)
        
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
    parser.add_argument('--use-train-data', action='store_true',
                        help='Use training data instead of validation data for evaluation')
    parser.add_argument('--mode', type=str, choices=['reconstruction', 'generation', 'latent_space'],
                        default='reconstruction', help='Evaluation mode')
    parser.add_argument('--num-samples', type=int,
                        help='Number of samples to generate (for generation mode)')
    parser.add_argument('--device', type=str,
                        help='Device to use')
    parser.add_argument('--max-samples', type=int,
                        help='Maximum number of samples to evaluate (for faster testing)')
    parser.add_argument('--quick', action='store_true',
                        help='Quick evaluation mode with reduced samples and larger batches')
    
    args = parser.parse_args()
    
    # Load configuration
    config = Config.from_yaml(args.config)
    
    # Override with command line arguments
    if args.checkpoint is not None:
        config.paths.checkpoint_path = args.checkpoint
    if args.dataset is not None:
        print(f"Using custom dataset path: {args.dataset}")
        # Don't modify config, just use the path directly
    elif args.use_train_data:
        print(f"Using training data instead of validation data for evaluation")
        # Don't modify config, just use training path directly
    if args.num_samples is not None:
        config.evaluation.num_samples = args.num_samples
    if args.device is not None:
        config.system.device = args.device
        config.evaluation.device = args.device
    
    # Quick mode optimizations
    if args.quick:
        config.evaluation.num_samples = min(50, config.evaluation.num_samples)
        # Increase batch sizes for quick mode
        if hasattr(config.evaluation, 'batch_size'):
            config.evaluation.batch_size = min(128, config.evaluation.batch_size * 4)
        else:
            config.evaluation.batch_size = 128
    
    # Apply max samples limit
    if args.max_samples is not None:
        if hasattr(config.evaluation, 'batch_size'):
            config.evaluation.batch_size = min(config.evaluation.batch_size, args.max_samples)
    
    # Initialize evaluator
    evaluator = PanGuEvaluator(config)
    
    # Determine evaluation dataset path
    if args.dataset is not None:
        eval_path = args.dataset
    elif args.use_train_data:
        eval_path = config.data.train_dataset_path
    else:
        # Use test data by default for evaluation
        eval_path = config.data.test_dataset_path
    
    print(f"Loading evaluation dataset from: {eval_path}")
    dataset = ZINC_Dataset(eval_path)
    print(f"Loaded evaluation dataset with {len(dataset)} molecules")
    
    # Apply max samples limit if specified
    if args.max_samples is not None and len(dataset) > args.max_samples:
        print(f"Limiting evaluation to first {args.max_samples} samples")
        from torch.utils.data import Subset
        dataset = Subset(dataset, range(args.max_samples))
    
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