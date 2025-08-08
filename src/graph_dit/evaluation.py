import torch
import numpy as np
import json
import os
from typing import List, Dict, Any, Optional, Tuple
from rdkit import Chem
from rdkit.Chem import Descriptors, QED, rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem.rdMolDescriptors import GetMorganFingerprint
from rdkit import DataStructs
from sklearn.metrics import mean_squared_error, r2_score, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import pandas as pd
from tqdm import tqdm


class MoleculeEvaluator:
    """
    Comprehensive evaluation framework for generated molecules.
    """
    
    def __init__(self, reference_dataset: Optional[List[str]] = None):
        self.reference_dataset = reference_dataset or []
        
    def evaluate_generation(
        self,
        generated_molecules: List[Dict[str, Any]],
        reference_molecules: Optional[List[str]] = None,
        save_path: Optional[str] = None
    ) -> Dict[str, float]:
        """
        Evaluate generated molecules against MOSES and other benchmarks.
        
        Args:
            generated_molecules: List of generated molecules with SMILES
            reference_molecules: Reference dataset (optional)
            save_path: Path to save evaluation results
            
        Returns:
            Dictionary of evaluation metrics
        """
        if reference_molecules is None:
            reference_molecules = self.reference_dataset
        
        # Extract SMILES strings
        generated_smiles = []
        valid_molecules = []
        
        for mol in generated_molecules:
            smiles = mol.get('smiles', '')
            if smiles and self._is_valid_smiles(smiles):
                generated_smiles.append(smiles)
                valid_molecules.append(mol)
        
        # Calculate metrics
        metrics = {}
        
        # Basic validity metrics
        metrics.update(self._calculate_validity_metrics(generated_smiles))
        
        # Chemical validity metrics
        metrics.update(self._calculate_chemical_validity_metrics(generated_smiles))
        
        # Uniqueness and novelty
        metrics.update(self._calculate_uniqueness_novelty(generated_smiles, reference_molecules))
        
        # Molecular property distributions
        metrics.update(self._calculate_property_distributions(generated_smiles, reference_molecules))
        
        # Scaffold analysis
        metrics.update(self._calculate_scaffold_metrics(generated_smiles, reference_molecules))
        
        # Save results if requested
        if save_path:
            self._save_evaluation_results(metrics, save_path)
        
        return metrics
    
    def _is_valid_smiles(self, smiles: str) -> bool:
        """Check if SMILES is valid."""
        try:
            mol = Chem.MolFromSmiles(smiles)
            return mol is not None
        except:
            return False
    
    def _calculate_validity_metrics(self, smiles_list: List[str]) -> Dict[str, float]:
        """Calculate validity metrics."""
        total = len(smiles_list)
        
        # Valid molecules
        valid_mols = [Chem.MolFromSmiles(s) for s in smiles_list if self._is_valid_smiles(s)]
        valid_count = len(valid_mols)
        validity_rate = valid_count / total if total > 0 else 0.0
        
        # Chemically valid (syntactically correct + chemically reasonable)
        chemically_valid = []
        for mol in valid_mols:
            if mol is not None:
                try:
                    Chem.SanitizeMol(mol)
                    chemically_valid.append(mol)
                except:
                    pass
        
        chemical_validity_rate = len(chemically_valid) / total if total > 0 else 0.0
        
        return {
            'validity': validity_rate,
            'chemical_validity': chemical_validity_rate,
            'valid_molecules': len(chemically_valid),
            'total_molecules': total
        }
    
    def _calculate_chemical_validity_metrics(self, smiles_list: List[str]) -> Dict[str, float]:
        """Calculate chemical validity metrics."""
        valid_mols = [Chem.MolFromSmiles(s) for s in smiles_list if self._is_valid_smiles(s)]
        valid_mols = [mol for mol in valid_mols if mol is not None]
        
        if not valid_mols:
            return {'passes_filters': 0.0, 'kekulizable': 0.0, 'charge_balanced': 0.0}
        
        passes_filters = 0
        kekulizable = 0
        charge_balanced = 0
        
        for mol in valid_mols:
            try:
                # Basic filters
                if self._pass_basic_filters(mol):
                    passes_filters += 1
                
                # Kekulization
                Chem.Kekulize(mol)
                kekulizable += 1
                
                # Charge balance
                if self._is_charge_balanced(mol):
                    charge_balanced += 1
                    
            except:
                continue
        
        total = len(valid_mols)
        return {
            'passes_filters': passes_filters / total,
            'kekulizable': kekulizable / total,
            'charge_balanced': charge_balanced / total
        }
    
    def _pass_basic_filters(self, mol: Chem.Mol) -> bool:
        """Apply basic chemical filters."""
        # Molecular weight filter
        mw = Descriptors.MolWt(mol)
        if mw < 50 or mw > 1000:
            return False
        
        # Heavy atom count filter
        heavy_atoms = mol.GetNumHeavyAtoms()
        if heavy_atoms < 3 or heavy_atoms > 50:
            return False
        
        # Element filter (common elements)
        allowed_elements = {6, 7, 8, 9, 15, 16, 17, 35, 53, 1}  # C, N, O, F, P, S, Cl, Br, I, H
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() not in allowed_elements:
                return False
        
        return True
    
    def _is_charge_balanced(self, mol: Chem.Mol) -> bool:
        """Check if molecule is charge balanced."""
        total_charge = sum(atom.GetFormalCharge() for atom in mol.GetAtoms())
        return total_charge == 0
    
    def _calculate_uniqueness_novelty(
        self,
        generated_smiles: List[str],
        reference_molecules: List[str]
    ) -> Dict[str, float]:
        """Calculate uniqueness and novelty metrics."""
        valid_mols = [Chem.MolFromSmiles(s) for s in generated_smiles if self._is_valid_smiles(s)]
        valid_mols = [mol for mol in valid_mols if mol is not None]
        
        if not valid_mols:
            return {'uniqueness': 0.0, 'novelty': 0.0}
        
        # Canonical SMILES for uniqueness
        canonical_smiles = []
        for mol in valid_mols:
            try:
                Chem.SanitizeMol(mol)
                canonical = Chem.MolToSmiles(mol, canonical=True)
                canonical_smiles.append(canonical)
            except:
                continue
        
        # Uniqueness
        unique_smiles = list(set(canonical_smiles))
        uniqueness = len(unique_smiles) / len(canonical_smiles) if canonical_smiles else 0.0
        
        # Novelty (not in reference)
        reference_set = set(reference_molecules)
        novel_count = sum(1 for s in unique_smiles if s not in reference_set)
        novelty = novel_count / len(unique_smiles) if unique_smiles else 0.0
        
        return {
            'uniqueness': uniqueness,
            'novelty': novelty,
            'unique_molecules': len(unique_smiles),
            'novel_molecules': novel_count
        }
    
    def _calculate_property_distributions(
        self,
        generated_smiles: List[str],
        reference_molecules: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Calculate molecular property distributions."""
        valid_mols = [Chem.MolFromSmiles(s) for s in generated_smiles if self._is_valid_smiles(s)]
        valid_mols = [mol for mol in valid_mols if mol is not None]
        
        if not valid_mols:
            return {}
        
        properties = []
        for mol in valid_mols:
            try:
                Chem.SanitizeMol(mol)
                props = {
                    'molecular_weight': Descriptors.MolWt(mol),
                    'logp': Descriptors.MolLogP(mol),
                    'qed': QED.qed(mol),
                    'tpsa': Descriptors.TPSA(mol),
                    'num_atoms': mol.GetNumAtoms(),
                    'num_rings': rdMolDescriptors.CalcNumRings(mol),
                    'num_aromatic_rings': rdMolDescriptors.CalcNumAromaticRings(mol),
                    'num_rotatable_bonds': rdMolDescriptors.CalcNumRotatableBonds(mol),
                    'hbd': Descriptors.NumHDonors(mol),
                    'hba': Descriptors.NumHAcceptors(mol)
                }
                properties.append(props)
            except:
                continue
        
        if not properties:
            return {}
        
        # Calculate statistics
        stats = {}
        for prop_name in properties[0].keys():
            values = [p[prop_name] for p in properties]
            stats[prop_name] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values)),
                'median': float(np.median(values))
            }
        
        return stats
    
    def _calculate_scaffold_metrics(
        self,
        generated_smiles: List[str],
        reference_molecules: List[str]
    ) -> Dict[str, Any]:
        """Calculate scaffold-based metrics."""
        valid_mols = [Chem.MolFromSmiles(s) for s in generated_smiles if self._is_valid_smiles(s)]
        valid_mols = [mol for mol in valid_mols if mol is not None]
        
        if not valid_mols:
            return {'scaffold_diversity': 0.0}
        
        # Get scaffolds
        scaffolds = []
        for mol in valid_mols:
            try:
                scaffold = MurckoScaffold.GetScaffoldForMol(mol)
                scaffold_smiles = Chem.MolToSmiles(scaffold)
                scaffolds.append(scaffold_smiles)
            except:
                scaffolds.append('no_scaffold')
        
        # Scaffold diversity
        unique_scaffolds = len(set(scaffolds))
        scaffold_diversity = unique_scaffolds / len(scaffolds) if scaffolds else 0.0
        
        # Novel scaffolds
        reference_scaffolds = set()
        for smiles in reference_molecules:
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                try:
                    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
                    scaffold_smiles = Chem.MolToSmiles(scaffold)
                    reference_scaffolds.add(scaffold_smiles)
                except:
                    pass
        
        novel_scaffolds = sum(1 for s in set(scaffolds) if s not in reference_scaffolds)
        
        return {
            'scaffold_diversity': scaffold_diversity,
            'unique_scaffolds': unique_scaffolds,
            'novel_scaffolds': novel_scaffolds,
            'novel_scaffold_ratio': novel_scaffolds / unique_scaffolds if unique_scaffolds > 0 else 0.0
        }
    
    def _save_evaluation_results(self, metrics: Dict[str, Any], save_path: str) -> None:
        """Save evaluation results to file."""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        with open(save_path, 'w') as f:
            json.dump(metrics, f, indent=2)


class PropertyPredictionEvaluator:
    """
    Evaluator for property prediction models.
    """
    
    def __init__(self):
        pass
    
    def evaluate_regression(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        save_path: Optional[str] = None
    ) -> Dict[str, float]:
        """Evaluate regression model performance."""
        metrics = {
            'mse': float(mean_squared_error(targets, predictions)),
            'rmse': float(np.sqrt(mean_squared_error(targets, predictions))),
            'mae': float(np.mean(np.abs(targets - predictions))),
            'r2': float(r2_score(targets, predictions)),
            'pearson_r': float(np.corrcoef(targets, predictions)[0, 1]),
            'pearson_r2': float(np.corrcoef(targets, predictions)[0, 1] ** 2),
            'spearman_r': float(self._spearman_correlation(targets, predictions))
        }
        
        if save_path:
            self._save_evaluation_results(metrics, save_path)
        
        return metrics
    
    def evaluate_classification(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        save_path: Optional[str] = None
    ) -> Dict[str, float]:
        """Evaluate classification model performance."""
        # Handle binary classification
        if predictions.ndim == 1 or predictions.shape[1] == 1:
            predicted_labels = (predictions > 0.5).astype(int)
            
            metrics = {
                'accuracy': float(np.mean(predicted_labels == targets)),
                'auc': float(roc_auc_score(targets, predictions)),
                'f1': float(self._calculate_f1(targets, predicted_labels)),
                'precision': float(self._calculate_precision(targets, predicted_labels)),
                'recall': float(self._calculate_recall(targets, predicted_labels))
            }
        else:
            # Multi-class classification
            predicted_labels = np.argmax(predictions, axis=1)
            
            metrics = {
                'accuracy': float(np.mean(predicted_labels == targets))
            }
        
        if save_path:
            self._save_evaluation_results(metrics, save_path)
        
        return metrics
    
    def _spearman_correlation(self, x: np.ndarray, y: np.ndarray) -> float:
        """Calculate Spearman correlation."""
        x_rank = np.argsort(np.argsort(x))
        y_rank = np.argsort(np.argsort(y))
        return float(np.corrcoef(x_rank, y_rank)[0, 1])
    
    def _calculate_f1(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate F1 score."""
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        
        if tp + fp + fn == 0:
            return 0.0
        
        precision = tp / (tp + fp) if tp + fp > 0 else 0.0
        recall = tp / (tp + fn) if tp + fn > 0 else 0.0
        
        if precision + recall == 0:
            return 0.0
        
        return 2 * precision * recall / (precision + recall)
    
    def _calculate_precision(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate precision."""
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        
        return tp / (tp + fp) if tp + fp > 0 else 0.0
    
    def _calculate_recall(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate recall."""
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        
        return tp / (tp + fn) if tp + fn > 0 else 0.0
    
    def _save_evaluation_results(self, metrics: Dict[str, float], save_path: str) -> None:
        """Save evaluation results to file."""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        with open(save_path, 'w') as f:
            json.dump(metrics, f, indent=2)


class BenchmarkSuite:
    """
    Comprehensive benchmarking suite for GraphDiT models.
    """
    
    def __init__(self):
        self.molecule_evaluator = MoleculeEvaluator()
        self.property_evaluator = PropertyPredictionEvaluator()
    
    def run_generation_benchmark(
        self,
        generator,
        num_samples: int = 1000,
        reference_dataset: Optional[List[str]] = None,
        save_dir: str = 'benchmarks/generation'
    ) -> Dict[str, Any]:
        """Run generation benchmark."""
        os.makedirs(save_dir, exist_ok=True)
        
        # Generate molecules
        print(f"Generating {num_samples} molecules...")
        generated = generator.generate_batch(num_samples)
        
        # Evaluate
        metrics = self.molecule_evaluator.evaluate_generation(
            generated,
            reference_dataset,
            save_path=os.path.join(save_dir, 'generation_metrics.json')
        )
        
        # Create visualizations
        self._create_generation_plots(generated, save_dir)
        
        return {
            'metrics': metrics,
            'generated_count': len(generated),
            'save_dir': save_dir
        }
    
    def run_property_prediction_benchmark(
        self,
        predictor,
        test_loader,
        task_type: str,
        save_dir: str = 'benchmarks/property_prediction'
    ) -> Dict[str, Any]:
        """Run property prediction benchmark."""
        os.makedirs(save_dir, exist_ok=True)
        
        # Get predictions
        predictions = []
        targets = []
        
        for batch in test_loader:
            batch = batch.to(next(predictor.parameters()).device)
            pred = predictor.predict(batch)
            predictions.extend(pred)
            targets.extend(batch.y.cpu().numpy())
        
        predictions = np.array(predictions)
        targets = np.array(targets)
        
        # Evaluate based on task type
        if task_type == 'regression':
            metrics = self.property_evaluator.evaluate_regression(
                predictions, targets,
                save_path=os.path.join(save_dir, 'regression_metrics.json')
            )
        else:
            metrics = self.property_evaluator.evaluate_classification(
                predictions, targets,
                save_path=os.path.join(save_dir, 'classification_metrics.json')
            )
        
        # Create visualizations
        self._create_property_prediction_plots(predictions, targets, task_type, save_dir)
        
        return {
            'metrics': metrics,
            'predictions': predictions.tolist(),
            'targets': targets.tolist(),
            'save_dir': save_dir
        }
    
    def run_optimization_benchmark(
        self,
        optimizer,
        test_molecules: List[str],
        property_function,
        property_target: float,
        save_dir: str = 'benchmarks/optimization'
    ) -> Dict[str, Any]:
        """Run optimization benchmark."""
        os.makedirs(save_dir, exist_ok=True)
        
        # Optimize molecules
        results = []
        for smiles in tqdm(test_molecules, desc="Optimizing molecules"):
            result = optimizer.optimize_molecule(
                smiles, property_function, property_target
            )
            results.append(result)
        
        # Analyze results
        analysis = self._analyze_optimization_results(results)
        
        # Save results
        with open(os.path.join(save_dir, 'optimization_results.json'), 'w') as f:
            json.dump(results, f, indent=2)
        
        with open(os.path.join(save_dir, 'optimization_analysis.json'), 'w') as f:
            json.dump(analysis, f, indent=2)
        
        # Create visualizations
        self._create_optimization_plots(results, save_dir)
        
        return {
            'results': results,
            'analysis': analysis,
            'save_dir': save_dir
        }
    
    def create_benchmark_report(
        self,
        results: Dict[str, Any],
        save_path: str
    ) -> None:
        """Create comprehensive benchmark report."""
        report = {
            'timestamp': pd.Timestamp.now().isoformat(),
            'generation_results': results.get('generation', {}),
            'property_prediction_results': results.get('property_prediction', {}),
            'optimization_results': results.get('optimization', {}),
            'summary': self._create_summary(results)
        }
        
        with open(save_path, 'w') as f:
            json.dump(report, f, indent=2)
    
    def _create_generation_plots(self, generated_molecules: List[Dict[str, Any]], save_dir: str) -> None:
        """Create generation analysis plots."""
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Extract properties
        properties = []
        for mol in generated_molecules:
            if mol.get('valid', False) and 'properties' in mol:
                properties.append(mol['properties'])
        
        if properties:
            df = pd.DataFrame(properties)
            
            # Molecular weight distribution
            axes[0, 0].hist(df['molecular_weight'], bins=30, alpha=0.7)
            axes[0, 0].set_title('Molecular Weight Distribution')
            axes[0, 0].set_xlabel('Molecular Weight')
            axes[0, 0].set_ylabel('Count')
            
            # LogP distribution
            axes[0, 1].hist(df['logp'], bins=30, alpha=0.7)
            axes[0, 1].set_title('LogP Distribution')
            axes[0, 1].set_xlabel('LogP')
            axes[0, 1].set_ylabel('Count')
            
            # QED distribution
            axes[0, 2].hist(df['qed'], bins=30, alpha=0.7)
            axes[0, 2].set_title('QED Distribution')
            axes[0, 2].set_xlabel('QED')
            axes[0, 2].set_ylabel('Count')
            
            # TPSA vs LogP scatter
            axes[1, 0].scatter(df['tpsa'], df['logp'], alpha=0.6)
            axes[1, 0].set_title('TPSA vs LogP')
            axes[1, 0].set_xlabel('TPSA')
            axes[1, 0].set_ylabel('LogP')
            
            # Molecular weight vs QED scatter
            axes[1, 1].scatter(df['molecular_weight'], df['qed'], alpha=0.6)
            axes[1, 1].set_title('Molecular Weight vs QED')
            axes[1, 1].set_xlabel('Molecular Weight')
            axes[1, 1].set_ylabel('QED')
            
            # Property correlation heatmap
            corr_matrix = df[['molecular_weight', 'logp', 'qed', 'tpsa']].corr()
            sns.heatmap(corr_matrix, annot=True, fmt='.2f', ax=axes[1, 2])
            axes[1, 2].set_title('Property Correlations')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'generation_analysis.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_property_prediction_plots(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        task_type: str,
        save_dir: str
    ) -> None:
        """Create property prediction analysis plots."""
        plt.style.use('seaborn-v0_8')
        
        if task_type == 'regression':
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            
            # Scatter plot
            axes[0].scatter(targets, predictions, alpha=0.6)
            axes[0].plot([targets.min(), targets.max()], [targets.min(), targets.max()], 'r--', lw=2)
            axes[0].set_xlabel('True Values')
            axes[0].set_ylabel('Predicted Values')
            axes[0].set_title('True vs Predicted Values')
            
            # Residuals plot
            residuals = targets - predictions
            axes[1].scatter(predictions, residuals, alpha=0.6)
            axes[1].axhline(y=0, color='r', linestyle='--')
            axes[1].set_xlabel('Predicted Values')
            axes[1].set_ylabel('Residuals')
            axes[1].set_title('Residual Plot')
        else:
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            
            # ROC curve
            fpr, tpr, _ = roc_curve(targets, predictions)
            axes[0].plot(fpr, tpr, label='ROC curve')
            axes[0].plot([0, 1], [0, 1], 'k--', label='Random')
            axes[0].set_xlabel('False Positive Rate')
            axes[0].set_ylabel('True Positive Rate')
            axes[0].set_title('ROC Curve')
            axes[0].legend()
            
            # Prediction distribution
            axes[1].hist(predictions, bins=30, alpha=0.7, label='Predictions')
            axes[1].axvline(x=0.5, color='r', linestyle='--', label='Threshold')
            axes[1].set_xlabel('Prediction Probability')
            axes[1].set_ylabel('Count')
            axes[1].set_title('Prediction Distribution')
            axes[1].legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'property_prediction_analysis.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_optimization_plots(self, results: List[Dict[str, Any]], save_dir: str) -> None:
        """Create optimization analysis plots."""
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Extract data
        improvements = []
        similarities = []
        property_values = []
        
        for result in results:
            if 'optimized' in result and result['optimized'].get('valid', False):
                original = result['original']['property_value']
                optimized = result['optimized']['property_value']
                target = result['target']
                
                improvements.append(optimized - original)
                similarities.append(result['optimized']['similarity'])
                property_values.append(optimized)
        
        if improvements:
            # Improvement distribution
            axes[0, 0].hist(improvements, bins=30, alpha=0.7)
            axes[0, 0].set_title('Property Improvement Distribution')
            axes[0, 0].set_xlabel('Property Change')
            axes[0, 0].set_ylabel('Count')
            
            # Similarity vs improvement
            axes[0, 1].scatter(similarities, improvements, alpha=0.6)
            axes[0, 1].set_title('Similarity vs Improvement')
            axes[0, 1].set_xlabel('Tanimoto Similarity')
            axes[0, 1].set_ylabel('Property Change')
            
            # Final property distribution
            axes[1, 0].hist(property_values, bins=30, alpha=0.7)
            axes[1, 0].axvline(np.mean([r['target'] for r in results]), color='r', linestyle='--')
            axes[1, 0].set_title('Final Property Distribution')
            axes[1, 0].set_xlabel('Property Value')
            axes[1, 0].set_ylabel('Count')
            
            # Success rate by similarity
            success_by_similarity = {}
            for sim, imp in zip(similarities, improvements):
                sim_bin = int(sim * 10) / 10  # 0.1 bins
                if sim_bin not in success_by_similarity:
                    success_by_similarity[sim_bin] = []
                success_by_similarity[sim_bin].append(1 if abs(imp) > 0.1 else 0)
            
            sim_bins = list(success_by_similarity.keys())
            success_rates = [np.mean(success_by_similarity[bin]) for bin in sim_bins]
            
            axes[1, 1].plot(sim_bins, success_rates, 'o-')
            axes[1, 1].set_title('Success Rate by Similarity')
            axes[1, 1].set_xlabel('Similarity Bin')
            axes[1, 1].set_ylabel('Success Rate')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'optimization_analysis.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_summary(self, results: Dict[str, Any]) -> Dict[str, str]:
        """Create summary of benchmark results."""
        summary = {}
        
        if 'generation' in results:
            gen_metrics = results['generation']['metrics']
            summary['generation'] = f"Validity: {gen_metrics['validity']:.3f}, " \
                                   f"Uniqueness: {gen_metrics['uniqueness']:.3f}, " \
                                   f"Novelty: {gen_metrics['novelty']:.3f}"
        
        if 'property_prediction' in results:
            prop_metrics = results['property_prediction']['metrics']
            summary['property_prediction'] = f"R²: {prop_metrics['r2']:.3f}, " \
                                           f"RMSE: {prop_metrics['rmse']:.3f}"
        
        if 'optimization' in results:
            opt_analysis = results['optimization']['analysis']
            summary['optimization'] = f"Success rate: {opt_analysis['summary']['success_rate']:.3f}, " \
                                    f"Avg improvement: {opt_analysis['summary']['avg_improvement']:.3f}"
        
        return summary


def create_benchmark_suite() -> BenchmarkSuite:
    """Create benchmark suite instance."""
    return BenchmarkSuite()


def run_full_benchmark(
    generator=None,
    property_predictor=None,
    optimizer=None,
    save_dir: str = 'benchmarks/full'
) -> Dict[str, Any]:
    """Run comprehensive benchmark across all tasks."""
    suite = create_benchmark_suite()
    
    results = {}
    
    if generator:
        results['generation'] = suite.run_generation_benchmark(
            generator, save_dir=os.path.join(save_dir, 'generation')
        )
    
    if property_predictor:
        # Note: This would need test_loader parameter
        pass
    
    if optimizer:
        # Note: This would need test_molecules parameter
        pass
    
    # Create comprehensive report
    suite.create_benchmark_report(results, os.path.join(save_dir, 'benchmark_report.json'))
    
    return results