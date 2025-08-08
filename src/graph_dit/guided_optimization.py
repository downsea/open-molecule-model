import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Callable
from rdkit import Chem
from rdkit.Chem import Descriptors, QED
import json
import os
from tqdm import tqdm
import copy

from .model import GraphDiT, CriticModel
from .scheduler import CategoricalNoiseScheduler
from .generator import GraphDiTGenerator
from .data import smiles_to_graph, create_featurizer


class GuidedDiffusionOptimizer:
    """
    Guided diffusion optimizer for molecule optimization.
    Uses gradient-based guidance during the reverse diffusion process.
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
        self.featurizer = create_featurizer()
        
        # Set models to evaluation mode
        self.generator.model.eval()
        self.critic_model.eval()
    
    def optimize_molecule(
        self,
        smiles: str,
        property_function: Callable,
        property_target: float,
        num_steps: int = 50,
        guidance_scale: float = 1.0,
        noise_start: float = 1.0,
        noise_end: float = 0.0,
        temperature: float = 1.0,
        similarity_constraint: float = 0.7,
        max_attempts: int = 10
    ) -> Dict[str, Any]:
        """
        Optimize a molecule for a specific property using guided diffusion.
        
        Args:
            smiles: Input SMILES string
            property_function: Function to compute molecular property
            property_target: Target property value
            num_steps: Number of optimization steps
            guidance_scale: Strength of guidance (0-2)
            noise_start: Starting noise level (0-1)
            noise_end: Ending noise level (0-1)
            temperature: Sampling temperature
            similarity_constraint: Minimum Tanimoto similarity to original
            max_attempts: Maximum optimization attempts
            
        Returns:
            Optimization results dictionary
        """
        # Load original molecule
        original_data = smiles_to_graph(smiles)
        if original_data is None:
            return {'error': 'Invalid SMILES input'}
        
        original_data = original_data.to(self.device)
        original_mol = Chem.MolFromSmiles(smiles)
        if original_mol is None:
            return {'error': 'Cannot parse original molecule'}
        
        # Get original fingerprint for similarity calculation
        original_fp = Chem.RDKFingerprint(original_mol)
        
        # Initialize optimization state
        best_molecule = None
        best_score = float('inf')  # For minimization
        best_similarity = 1.0
        
        optimization_history = []
        
        for attempt in range(max_attempts):
            # Initialize from noisy version
            current_mol = self._initialize_noisy_molecule(
                original_data, noise_start
            )
            
            # Guided optimization
            optimized_mol = self._guided_optimization_step(
                current_mol,
                property_function,
                property_target,
                num_steps,
                guidance_scale,
                noise_end,
                temperature
            )
            
            # Validate and score
            result = self._validate_and_score(
                optimized_mol,
                original_fp,
                property_function,
                property_target,
                similarity_constraint
            )
            
            optimization_history.append({
                'attempt': attempt + 1,
                'molecule': result,
                'property_value': result['property_value'],
                'similarity': result['similarity'],
                'score': result['score']
            })
            
            # Update best molecule
            if result['score'] < best_score and result['valid']:
                best_score = result['score']
                best_molecule = result
                best_similarity = result['similarity']
        
        return {
            'original': {
                'smiles': smiles,
                'property_value': float(property_function(original_mol)),
                'properties': self._get_molecular_properties(original_mol)
            },
            'optimized': best_molecule,
            'history': optimization_history,
            'target': property_target,
            'parameters': {
                'guidance_scale': guidance_scale,
                'num_steps': num_steps,
                'temperature': temperature,
                'similarity_constraint': similarity_constraint
            }
        }
    
    def _initialize_noisy_molecule(
        self,
        original_data,
        noise_level: float
    ) -> Dict[str, torch.Tensor]:
        """Initialize molecule with controlled noise."""
        num_timesteps = int(self.generator.noise_scheduler.num_timesteps * noise_level)
        
        # Add noise to original molecule
        x_noisy = self.generator.noise_scheduler.q_sample(
            original_data.x, num_timesteps
        )
        
        edge_attr_noisy = self.generator.noise_scheduler.q_sample(
            original_data.edge_attr, num_timesteps
        )
        
        return {
            'x': x_noisy,
            'edge_index': original_data.edge_index,
            'edge_attr': edge_attr_noisy,
            'batch': torch.zeros(original_data.num_nodes, device=self.device, dtype=torch.long)
        }
    
    def _guided_optimization_step(
        self,
        molecule: Dict[str, torch.Tensor],
        property_function: Callable,
        property_target: float,
        num_steps: int,
        guidance_scale: float,
        noise_end: float,
        temperature: float
    ) -> Dict[str, torch.Tensor]:
        """Perform guided optimization step."""
        x_t = molecule['x'].clone()
        edge_index = molecule['edge_index']
        edge_attr = molecule['edge_attr'].clone()
        batch = molecule['batch']
        
        # Calculate noise schedule
        noise_start = 1.0
        noise_end_timestep = int(self.generator.noise_scheduler.num_timesteps * noise_end)
        
        for step in range(num_steps):
            # Current timestep
            current_t = int(noise_start * (1 - step / num_steps) * self.generator.noise_scheduler.num_timesteps)
            current_t = max(current_t, noise_end_timestep)
            
            # Enable gradients for guidance
            x_t.requires_grad = True
            edge_attr.requires_grad = True
            
            # Get model predictions
            timesteps = torch.tensor([current_t], device=self.device)
            
            node_logits, edge_logits = self.generator.model(
                x=F.one_hot(x_t, self.generator.noise_scheduler.num_node_classes).float(),
                edge_index=edge_index,
                edge_attr=F.one_hot(edge_attr, self.generator.noise_scheduler.num_edge_classes).float(),
                batch=batch,
                timesteps=timesteps
            )
            
            # Convert to molecule and compute property
            mol = self._graph_to_molecule(x_t, edge_index, edge_attr)
            if mol is None or not mol.get('valid', False):
                continue
            
            # Compute property and guidance loss
            current_property = torch.tensor([property_function(mol['mol'])], device=self.device, dtype=torch.float32)
            guidance_loss = F.mse_loss(current_property, torch.tensor([property_target], device=self.device))
            
            # Compute gradients
            node_grad = torch.autograd.grad(guidance_loss, node_logits, retain_graph=True)[0]
            edge_grad = torch.autograd.grad(guidance_loss, edge_logits)[0]
            
            # Apply guidance
            guided_node_logits = node_logits - guidance_scale * node_grad
            guided_edge_logits = edge_logits - guidance_scale * edge_grad
            
            # Sample next state
            node_probs = F.softmax(guided_node_logits / temperature, dim=-1)
            x_t = torch.multinomial(node_probs, 1).squeeze(-1)
            
            edge_probs = F.softmax(guided_edge_logits / temperature, dim=-1)
            edge_attr = torch.multinomial(edge_probs, 1).squeeze(-1)
            
            # Detach for next iteration
            x_t = x_t.detach()
            edge_attr = edge_attr.detach()
        
        return {
            'x': x_t,
            'edge_index': edge_index,
            'edge_attr': edge_attr,
            'batch': batch
        }
    
    def _validate_and_score(
        self,
        molecule: Dict[str, torch.Tensor],
        original_fp,
        property_function: Callable,
        property_target: float,
        similarity_constraint: float
    ) -> Dict[str, Any]:
        """Validate molecule and compute optimization score."""
        
        mol = self._graph_to_molecule(
            molecule['x'], molecule['edge_index'], molecule['edge_attr']
        )
        
        if mol is None or not mol.get('valid', False):
            return {'valid': False, 'error': 'Invalid molecule'}
        
        # Calculate property value
        property_value = float(property_function(mol['mol']))
        
        # Calculate similarity
        optimized_fp = Chem.RDKFingerprint(mol['mol'])
        similarity = Chem.DataStructs.TanimotoSimilarity(original_fp, optimized_fp)
        
        # Calculate optimization score
        property_error = abs(property_value - property_target)
        similarity_penalty = max(0, similarity_constraint - similarity) * 10
        
        score = property_error + similarity_penalty
        
        return {
            'valid': True,
            'smiles': mol['smiles'],
            'property_value': property_value,
            'similarity': similarity,
            'score': score,
            'properties': mol['properties']
        }
    
    def _graph_to_molecule(self, x, edge_index, edge_attr) -> Optional[Dict[str, Any]]:
        """Convert graph to molecule."""
        try:
            # Create RDKit molecule
            mol = self.generator._graph_to_molecule(x, edge_index, edge_attr)
            return mol
        except Exception as e:
            return {'valid': False, 'error': str(e)}


class PropertyFunctions:
    """Collection of property functions for optimization."""
    
    @staticmethod
    def molecular_weight(mol: Chem.Mol) -> float:
        """Calculate molecular weight."""
        return float(Descriptors.MolWt(mol))
    
    @staticmethod
    def logp(mol: Chem.Mol) -> float:
        """Calculate LogP."""
        return float(Descriptors.MolLogP(mol))
    
    @staticmethod
    def tpsa(mol: Chem.Mol) -> float:
        """Calculate TPSA."""
        return float(Descriptors.TPSA(mol))
    
    @staticmethod
    def qed(mol: Chem.Mol) -> float:
        """Calculate QED."""
        return float(Chem.QED.qed(mol))
    
    @staticmethod
    def sascore(mol: Chem.Mol) -> float:
        """Calculate synthetic accessibility score."""
        try:
            from rdkit.Chem import SAscore
            return float(SAscore.calculateScore(mol))
        except ImportError:
            return 0.0
    
    @staticmethod
    def hbd(mol: Chem.Mol) -> float:
        """Calculate hydrogen bond donors."""
        return float(Descriptors.NumHDonors(mol))
    
    @staticmethod
    def hba(mol: Chem.Mol) -> float:
        """Calculate hydrogen bond acceptors."""
        return float(Descriptors.NumHAcceptors(mol))


class MultiObjectiveOptimizer:
    """
    Multi-objective optimization for molecules.
    """
    
    def __init__(
        self,
        generator: GraphDiTGenerator,
        critic_model: CriticModel,
        device: str = 'cuda'
    ):
        self.guided_optimizer = GuidedDiffusionOptimizer(generator, critic_model, device)
        self.device = device
    
    def optimize_multiple_properties(
        self,
        smiles: str,
        property_functions: List[Callable],
        property_targets: List[float],
        property_weights: List[float],
        **kwargs
    ) -> Dict[str, Any]:
        """
        Optimize for multiple properties simultaneously.
        
        Args:
            smiles: Input SMILES
            property_functions: List of property functions
            property_targets: List of target values
            property_weights: List of weights for each property
            **kwargs: Additional optimization parameters
            
        Returns:
            Multi-objective optimization results
        """
        
        def combined_property_function(mol):
            """Combined property function for multi-objective optimization."""
            total_score = 0.0
            for func, target, weight in zip(property_functions, property_targets, property_weights):
                value = func(mol)
                score = abs(value - target)
                total_score += weight * score
            return total_score
        
        return self.guided_optimizer.optimize_molecule(
            smiles,
            combined_property_function,
            0.0,  # Target is minimum score
            **kwargs
        )


def create_optimization_pipeline(
    generator: GraphDiTGenerator,
    critic_model: CriticModel,
    device: str = 'cuda'
) -> GuidedDiffusionOptimizer:
    """Create optimization pipeline."""
    return GuidedDiffusionOptimizer(generator, critic_model, device)


def batch_optimize_molecules(
    optimizer: GuidedDiffusionOptimizer,
    molecules: List[str],
    property_function: Callable,
    property_target: float,
    **kwargs
) -> List[Dict[str, Any]]:
    """Optimize multiple molecules in batch."""
    results = []
    
    for smiles in tqdm(molecules, desc="Optimizing molecules"):
        result = optimizer.optimize_molecule(smiles, property_function, property_target, **kwargs)
        results.append(result)
    
    return results


def save_optimization_results(results: List[Dict[str, Any]], output_path: str) -> None:
    """Save optimization results to file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Optimization results saved to {output_path}")


def analyze_optimization_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze optimization results."""
    if not results:
        return {'error': 'No results to analyze'}
    
    valid_results = [r for r in results if 'optimized' in r and r['optimized'].get('valid', False)]
    
    if not valid_results:
        return {'error': 'No valid optimization results'}
    
    # Calculate improvement metrics
    improvements = []
    for result in valid_results:
        original_value = result['original']['property_value']
        optimized_value = result['optimized']['property_value']
        target = result['target']
        
        improvement = {
            'original': original_value,
            'optimized': optimized_value,
            'target': target,
            'improvement': optimized_value - original_value,
            'target_error': abs(optimized_value - target),
            'similarity': result['optimized']['similarity']
        }
        improvements.append(improvement)
    
    # Summary statistics
    summary = {
        'total_molecules': len(results),
        'successful_optimizations': len(valid_results),
        'success_rate': len(valid_results) / len(results),
        'avg_improvement': np.mean([i['improvement'] for i in improvements]),
        'avg_target_error': np.mean([i['target_error'] for i in improvements]),
        'avg_similarity': np.mean([i['similarity'] for i in improvements])
    }
    
    return {
        'summary': summary,
        'detailed_improvements': improvements
    }