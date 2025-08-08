import torch
import torch.nn.functional as F
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, QED
from typing import List, Dict, Any, Optional, Tuple
import json
import os
from tqdm import tqdm

from .model import GraphDiT, CriticModel
from .scheduler import CategoricalNoiseScheduler
from .data import create_featurizer, smiles_to_graph


class GraphDiTGenerator:
    """
    Generator for molecular graphs using trained GraphDiT model.
    """
    
    def __init__(
        self,
        model: GraphDiT,
        noise_scheduler: CategoricalNoiseScheduler,
        device: str = 'cuda'
    ):
        self.model = model.to(device)
        self.noise_scheduler = noise_scheduler
        self.device = device
        self.featurizer = create_featurizer()
        
        # Set model to evaluation mode
        self.model.eval()
    
    def generate_single_molecule(
        self,
        num_nodes: Optional[int] = None,
        node_range: Tuple[int, int] = (3, 50),
        temperature: float = 1.0,
        guidance_scale: float = 0.0,
        critic_model: Optional[CriticModel] = None,
        target_property: Optional[str] = None,
        property_target: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Generate a single molecule using the reverse diffusion process.
        
        Args:
            num_nodes: Fixed number of nodes (if None, sampled from range)
            node_range: Range for number of nodes
            temperature: Sampling temperature
            guidance_scale: Guidance scale for property optimization
            critic_model: Optional critic model for guided generation
            target_property: Target property for guidance
            property_target: Target property value
            
        Returns:
            Dictionary containing generated molecule information
        """
        with torch.no_grad():
            # Determine number of nodes
            if num_nodes is None:
                num_nodes = np.random.randint(node_range[0], node_range[1] + 1)
            
            # Initialize random graph
            x_t = torch.randint(
                0, self.noise_scheduler.num_node_classes,
                (num_nodes,), device=self.device
            )
            
            # Generate edge indices (fully connected initially)
            edge_index = self._generate_initial_edges(num_nodes)
            edge_attr = torch.randint(
                0, self.noise_scheduler.num_edge_classes,
                (edge_index.shape[1],), device=self.device
            )
            
            # Create batch tensor
            batch = torch.zeros(num_nodes, device=self.device, dtype=torch.long)
            
            # Reverse diffusion process
            for t in reversed(range(self.noise_scheduler.num_timesteps)):
                timesteps = torch.tensor([t], device=self.device)
                
                # Get model predictions
                node_logits, edge_logits = self.model(
                    x=F.one_hot(x_t, self.noise_scheduler.num_node_classes).float(),
                    edge_index=edge_index,
                    edge_attr=F.one_hot(edge_attr, self.noise_scheduler.num_edge_classes).float(),
                    batch=batch,
                    timesteps=timesteps
                )
                
                # Apply temperature scaling
                node_logits = node_logits / temperature
                edge_logits = edge_logits / temperature
                
                # Apply guidance if provided
                if guidance_scale > 0 and critic_model is not None:
                    node_logits, edge_logits = self._apply_guidance(
                        node_logits, edge_logits, x_t, edge_index, edge_attr, batch,
                        critic_model, target_property, property_target, guidance_scale
                    )
                
                # Sample next state
                node_probs = F.softmax(node_logits, dim=-1)
                x_t = torch.multinomial(node_probs, 1).squeeze(-1)
                
                edge_probs = F.softmax(edge_logits, dim=-1)
                edge_attr = torch.multinomial(edge_probs, 1).squeeze(-1)
                
                # Prune invalid edges (optional)
                if t % 100 == 0:  # Every 100 steps
                    edge_index, edge_attr = self._prune_edges(edge_index, edge_attr)
            
            # Convert to final molecule
            molecule_data = self._graph_to_molecule(x_t, edge_index, edge_attr)
            
            return {
                'atoms': x_t.cpu().numpy().tolist(),
                'edge_index': edge_index.cpu().numpy().tolist(),
                'edge_attr': edge_attr.cpu().numpy().tolist(),
                'smiles': molecule_data.get('smiles', ''),
                'valid': molecule_data.get('valid', False),
                'properties': molecule_data.get('properties', {}),
                'num_nodes': num_nodes
            }
    
    def generate_batch(
        self,
        num_samples: int,
        batch_size: int = 32,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Generate multiple molecules in batches.
        
        Args:
            num_samples: Number of molecules to generate
            batch_size: Batch size for generation
            **kwargs: Additional arguments for generation
            
        Returns:
            List of generated molecules
        """
        molecules = []
        
        with tqdm(total=num_samples, desc="Generating molecules") as pbar:
            for i in range(0, num_samples, batch_size):
                batch_size_actual = min(batch_size, num_samples - i)
                
                for _ in range(batch_size_actual):
                    molecule = self.generate_single_molecule(**kwargs)
                    molecules.append(molecule)
                    pbar.update(1)
        
        return molecules
    
    def _generate_initial_edges(self, num_nodes: int) -> torch.Tensor:
        """Generate initial edge connectivity."""
        # Create fully connected graph (excluding self-loops)
        edge_index = []
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                edge_index.extend([[i, j], [j, i]])
        
        if not edge_index:
            return torch.empty((2, 0), dtype=torch.long, device=self.device)
        
        return torch.tensor(edge_index, device=self.device).t().contiguous()
    
    def _prune_edges(self, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Prune edges based on bond type probabilities."""
        # Simple pruning: remove edges with bond type 0 (no bond)
        mask = edge_attr != 0
        return edge_index[:, mask], edge_attr[mask]
    
    def _graph_to_molecule(
        self,
        atoms: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor
    ) -> Dict[str, Any]:
        """
        Convert graph representation to RDKit molecule.
        
        Args:
            atoms: Atom types tensor
            edge_index: Edge indices tensor
            edge_attr: Edge attributes tensor
            
        Returns:
            Dictionary with molecule information
        """
        try:
            # Create empty molecule
            mol = Chem.RWMol()
            
            # Add atoms
            atom_symbols = list(self.featurizer.atom_to_idx.keys())
            for atom_type in atoms.cpu().numpy():
                if atom_type < len(atom_symbols):
                    symbol = atom_symbols[atom_type]
                    atom = Chem.Atom(symbol)
                    mol.AddAtom(atom)
            
            # Add bonds
            added_bonds = set()
            bond_types = list(self.featurizer.bond_to_idx.keys())
            
            for i, (u, v) in enumerate(edge_index.t().cpu().numpy()):
                bond_type_idx = edge_attr[i].cpu().item()
                if bond_type_idx > 0 and bond_type_idx < len(bond_types):
                    bond_type = bond_types[bond_type_idx]
                    
                    # Ensure u < v to avoid duplicate bonds
                    if u < v and (u, v) not in added_bonds:
                        mol.AddBond(int(u), int(v), bond_type)
                        added_bonds.add((u, v))
                        added_bonds.add((v, u))
            
            # Convert to mol and sanitize
            mol = mol.GetMol()
            if mol is None:
                return {'valid': False, 'error': 'Conversion failed'}
            
            try:
                Chem.SanitizeMol(mol)
                smiles = Chem.MolToSmiles(mol)
                
                # Calculate properties
                properties = {
                    'molecular_weight': float(Descriptors.MolWt(mol)),
                    'logp': float(Descriptors.MolLogP(mol)),
                    'tpsa': float(Descriptors.TPSA(mol)),
                    'qed': float(Chem.QED.qed(mol)),
                    'num_atoms': mol.GetNumAtoms(),
                    'num_bonds': mol.GetNumBonds()
                }
                
                return {
                    'valid': True,
                    'smiles': smiles,
                    'properties': properties,
                    'mol': mol
                }
                
            except Exception as e:
                return {'valid': False, 'error': str(e)}
                
        except Exception as e:
            return {'valid': False, 'error': str(e)}
    
    def _apply_guidance(
        self,
        node_logits: torch.Tensor,
        edge_logits: torch.Tensor,
        x_t: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        batch: torch.Tensor,
        critic_model: CriticModel,
        target_property: str,
        property_target: float,
        guidance_scale: float
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply guided generation using critic model."""
        
        # Enable gradients for guidance
        with torch.enable_grad():
            # Create graph representation
            x_onehot = F.one_hot(x_t, self.noise_scheduler.num_node_classes).float()
            edge_onehot = F.one_hot(edge_attr, self.noise_scheduler.num_edge_classes).float()
            
            # Get property prediction
            property_pred = critic_model(x_onehot, edge_index, edge_onehot, batch)
            
            # Compute gradient
            target_tensor = torch.tensor([property_target], device=self.device)
            guidance_loss = F.mse_loss(property_pred, target_tensor)
            
            # Compute gradients w.r.t. logits
            node_grad = torch.autograd.grad(guidance_loss, node_logits, retain_graph=True)[0]
            edge_grad = torch.autograd.grad(guidance_loss, edge_logits)[0]
            
            # Apply guidance
            node_logits = node_logits - guidance_scale * node_grad
            edge_logits = edge_logits - guidance_scale * edge_grad
        
        return node_logits, edge_logits


class MoleculeOptimizer:
    """
    Molecule optimizer using guided diffusion.
    """
    
    def __init__(
        self,
        generator: GraphDiTGenerator,
        critic_model: CriticModel,
        device: str = 'cuda'
    ):
        self.generator = generator
        self.critic_model = critic_model.to(device)
        self.device = device
    
    def optimize_molecule(
        self,
        smiles: str,
        target_property: str,
        property_target: float,
        num_steps: int = 50,
        guidance_scale: float = 1.0,
        noise_level: float = 0.5
    ) -> Dict[str, Any]:
        """
        Optimize a molecule for a specific property.
        
        Args:
            smiles: Input SMILES string
            target_property: Target property to optimize
            property_target: Target property value
            num_steps: Number of optimization steps
            guidance_scale: Strength of guidance
            noise_level: Initial noise level (0-1)
            
        Returns:
            Optimization results
        """
        # Convert SMILES to graph
        data = smiles_to_graph(smiles)
        if data is None:
            return {'error': 'Invalid SMILES'}
        
        # Move to device
        data = data.to(self.device)
        
        # Initialize with noise
        num_timesteps = int(self.generator.noise_scheduler.num_timesteps * noise_level)
        
        # Get optimized molecule
        optimized = self.generator.generate_single_molecule(
            num_nodes=data.num_nodes,
            guidance_scale=guidance_scale,
            critic_model=self.critic_model,
            target_property=target_property,
            property_target=property_target
        )
        
        return {
            'original': smiles,
            'optimized': optimized,
            'target_property': target_property,
            'target_value': property_target
        }
    
    def batch_optimize(
        self,
        smiles_list: List[str],
        target_property: str,
        property_target: float,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Optimize multiple molecules."""
        results = []
        
        for smiles in smiles_list:
            result = self.optimize_molecule(smiles, target_property, property_target, **kwargs)
            results.append(result)
        
        return results


def load_generator(checkpoint_path: str, config: Dict[str, Any], device: str = 'cuda') -> GraphDiTGenerator:
    """
    Load a trained generator from checkpoint.
    
    Args:
        checkpoint_path: Path to model checkpoint
        config: Model configuration
        device: Device to load on
        
    Returns:
        GraphDiTGenerator instance
    """
    from .scheduler import CategoricalNoiseScheduler
    
    # Create model and scheduler
    model = GraphDiT(
        node_dim=config['model']['node_dim'],
        edge_dim=config['model']['edge_dim'],
        hidden_dim=config['model']['hidden_dim'],
        num_layers=config['model']['num_layers'],
        num_heads=config['model']['num_heads']
    )
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Create scheduler
    scheduler = CategoricalNoiseScheduler(
        num_node_classes=config['model']['node_dim'],
        num_edge_classes=config['model']['edge_dim'],
        **config['scheduler']
    )
    
    return GraphDiTGenerator(model, scheduler, device)


def save_generation_results(results: List[Dict[str, Any]], output_path: str) -> None:
    """Save generation results to file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to {output_path}")


def validate_molecules(generated: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Validate generated molecules.
    
    Args:
        generated: List of generated molecules
        
    Returns:
        Validation metrics
    """
    valid_molecules = [m for m in generated if m.get('valid', False)]
    
    metrics = {
        'total_generated': len(generated),
        'valid_molecules': len(valid_molecules),
        'validity_rate': len(valid_molecules) / len(generated) if generated else 0,
        'unique_molecules': len(set([m.get('smiles', '') for m in valid_molecules])),
    }
    
    if valid_molecules:
        # Calculate average properties
        properties = [m.get('properties', {}) for m in valid_molecules]
        for prop in ['molecular_weight', 'logp', 'tpsa', 'qed']:
            values = [p.get(prop, 0) for p in properties if prop in p]
            if values:
                metrics[f'avg_{prop}'] = np.mean(values)
                metrics[f'std_{prop}'] = np.std(values)
    
    return metrics