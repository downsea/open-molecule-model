
import torch
import torch.nn as nn
from .encoder import GraphTransformerEncoder
from .decoder import TransformerDecoder

class PanGuDrugModel(nn.Module):
    def __init__(self, num_node_features, hidden_dim, num_encoder_layers, num_encoder_heads,
                 output_dim, num_decoder_heads, num_decoder_layers, latent_dim, num_selected_layers,
                 use_gradient_checkpointing=False, max_seq_length=512):
        super(PanGuDrugModel, self).__init__()
        
        # Store configuration for debugging and optimization
        self.config = {
            'num_node_features': num_node_features,
            'hidden_dim': hidden_dim,
            'num_encoder_layers': num_encoder_layers,
            'num_encoder_heads': num_encoder_heads,
            'output_dim': output_dim,
            'num_decoder_heads': num_decoder_heads,
            'num_decoder_layers': num_decoder_layers,
            'latent_dim': latent_dim,
            'num_selected_layers': num_selected_layers,
            'use_gradient_checkpointing': use_gradient_checkpointing
        }
        
        # Initialize optimized encoder
        self.encoder = GraphTransformerEncoder(
            num_node_features=num_node_features,
            hidden_dim=hidden_dim,
            num_layers=num_encoder_layers,
            num_heads=num_encoder_heads,
            num_selected_layers=num_selected_layers,
            use_gradient_checkpointing=use_gradient_checkpointing
        )
        
        # Calculate encoder output dimension: num_selected_layers * hidden_dim * num_heads
        encoder_output_dim = num_selected_layers * hidden_dim * num_encoder_heads
        
        # Project to latent space dimensions with proper initialization
        self.fc_mean = nn.Linear(encoder_output_dim, latent_dim)
        self.fc_log_var = nn.Linear(encoder_output_dim, latent_dim)
        
        # Initialize latent projection layers
        nn.init.xavier_uniform_(self.fc_mean.weight)
        nn.init.zeros_(self.fc_mean.bias)
        nn.init.xavier_uniform_(self.fc_log_var.weight)
        nn.init.zeros_(self.fc_log_var.bias)

        # Initialize optimized decoder
        self.decoder = TransformerDecoder(
            latent_dim=latent_dim,
            output_dim=output_dim,
            hidden_dim=hidden_dim,
            num_heads=num_decoder_heads,
            num_layers=num_decoder_layers,
            use_gradient_checkpointing=use_gradient_checkpointing,
            max_seq_length=max_seq_length
        )
        
        # Add dropout for regularization
        self.latent_dropout = nn.Dropout(0.1)

    def reparameterize(self, mean, log_var):
        """Reparameterization trick with numerical stability."""
        # Clamp log_var for numerical stability
        log_var = torch.clamp(log_var, min=-20, max=10)
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mean + eps * std

    def forward(self, data, condition_vector, tgt):
        """
        Optimized forward pass with better memory management.
        
        Args:
            data: PyG Data object with molecular graph
            condition_vector: Conditioning information (currently unused)
            tgt: Target sequence in one-hot format (batch_size, vocab_size, seq_len)
        
        Returns:
            output: Decoded logits (batch_size, vocab_size, seq_len)
            mean: Latent mean (batch_size, latent_dim)
            log_var: Latent log variance (batch_size, latent_dim)
        """
        # Encode the input graph
        encoded_features = self.encoder(data)
        
        # Get mean and log_var from the encoded features
        mean = self.fc_mean(encoded_features)
        log_var = self.fc_log_var(encoded_features)
        
        # Reparameterization trick with dropout for regularization
        z = self.reparameterize(mean, log_var)
        z = self.latent_dropout(z)
        
        # Decode the latent vector
        output = self.decoder(z, condition_vector, tgt)
        
        return output, mean, log_var
    
    def encode(self, data):
        """Encode molecular graph to latent space (for evaluation)."""
        with torch.no_grad():
            encoded_features = self.encoder(data)
            mean = self.fc_mean(encoded_features)
            log_var = self.fc_log_var(encoded_features)
            return mean, log_var
    
    def decode(self, z, max_length=128, temperature=1.0):
        """Decode latent vector to sequence (for generation)."""
        with torch.no_grad():
            batch_size = z.size(0)
            device = z.device
            
            # Start with SOS token (assuming index 1)
            current_seq = torch.zeros(batch_size, self.config['output_dim'], 1, device=device)
            current_seq[:, 1, 0] = 1.0  # SOS token
            
            generated_sequence = []
            
            for _ in range(max_length - 1):
                # Forward pass through decoder
                output = self.decoder(z, None, current_seq)
                
                # Get next token probabilities
                next_logits = output[:, :, -1] / temperature  # (batch_size, vocab_size)
                next_probs = torch.softmax(next_logits, dim=-1)
                
                # Sample next token
                next_token = torch.multinomial(next_probs, 1)  # (batch_size, 1)
                
                # Create one-hot encoding
                next_one_hot = torch.zeros(batch_size, self.config['output_dim'], 1, device=device)
                next_one_hot.scatter_(1, next_token.unsqueeze(-1), 1.0)
                
                # Append to sequence
                current_seq = torch.cat([current_seq, next_one_hot], dim=-1)
                generated_sequence.append(next_token.squeeze(-1))
                
                # Check for EOS token (assuming index 2)
                if (next_token.squeeze(-1) == 2).all():
                    break
            
            return torch.stack(generated_sequence, dim=1)  # (batch_size, seq_len)

def vae_loss(recon_x, x, mean, log_var, beta=1.0, reduction='mean'):
    """
    Optimized VAE loss for sequence generation with numerical stability.
    
    Args:
        recon_x: Reconstructed logits, shape [batch_size, vocab_size, seq_len]
        x: Target token indices, shape [batch_size, seq_len]
        mean: Mean of latent distribution, shape [batch_size, latent_dim]
        log_var: Log variance of latent distribution, shape [batch_size, latent_dim]
        beta: Weight for KL divergence term
        reduction: 'mean', 'sum', or 'none'
    
    Returns:
        Dictionary containing:
            - total_loss: Combined reconstruction + KL loss
            - recon_loss: Reconstruction loss component
            - kl_loss: KL divergence component
            - perplexity: Perplexity metric for monitoring
    """
    batch_size, vocab_size, seq_len = recon_x.shape
    
    # OPTIMIZATION: More efficient reshaping
    # Transpose and reshape in one operation
    recon_x_flat = recon_x.permute(0, 2, 1).contiguous().view(-1, vocab_size)
    x_flat = x.contiguous().view(-1)
    
    # Create mask for non-padding tokens (more efficient than ignore_index)
    mask = (x_flat != 0).float()
    num_valid_tokens = mask.sum()
    
    # Cross-entropy reconstruction loss with manual masking (more efficient)
    log_probs = torch.log_softmax(recon_x_flat, dim=-1)
    nll_loss = -log_probs.gather(1, x_flat.unsqueeze(1)).squeeze(1)
    
    # Apply mask and compute mean over valid tokens only
    if reduction == 'mean':
        recon_loss = (nll_loss * mask).sum() / (num_valid_tokens + 1e-8)
    elif reduction == 'sum':
        recon_loss = (nll_loss * mask).sum()
    else:
        recon_loss = nll_loss * mask
    
    # KL divergence with numerical stability and proper scaling
    # Clamp log_var to prevent numerical issues
    log_var_clamped = torch.clamp(log_var, min=-20, max=10)
    
    # More numerically stable KL computation
    kl_div = -0.5 * (1 + log_var_clamped - mean.pow(2) - log_var_clamped.exp())
    
    if reduction == 'mean':
        kl_loss = kl_div.mean()
    elif reduction == 'sum':
        kl_loss = kl_div.sum()
    else:
        kl_loss = kl_div.sum(dim=1)
    
    # Total loss
    total_loss = recon_loss + beta * kl_loss
    
    # Calculate perplexity for monitoring (only for mean reduction)
    if reduction == 'mean':
        perplexity = torch.exp(recon_loss)
    else:
        perplexity = None
    
    return {
        'total_loss': total_loss,
        'recon_loss': recon_loss,
        'kl_loss': kl_loss,
        'perplexity': perplexity,
        'num_valid_tokens': num_valid_tokens
    }

def compute_molecular_metrics(generated_smiles, target_smiles=None):
    """
    Compute molecular validity and diversity metrics.
    
    Args:
        generated_smiles: List of generated SMILES strings
        target_smiles: Optional list of target SMILES for comparison
    
    Returns:
        Dictionary with molecular metrics
    """
    from rdkit import Chem
    from rdkit.Chem import Descriptors
    import numpy as np
    
    valid_mols = []
    valid_smiles = []
    
    # Check validity
    for smiles in generated_smiles:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            valid_mols.append(mol)
            valid_smiles.append(Chem.MolToSmiles(mol))  # Canonicalize
    
    validity = len(valid_mols) / len(generated_smiles) if generated_smiles else 0.0
    uniqueness = len(set(valid_smiles)) / len(valid_smiles) if valid_smiles else 0.0
    
    # Compute molecular properties
    properties = {
        'molecular_weights': [],
        'logp_values': [],
        'qed_scores': []
    }
    
    for mol in valid_mols:
        try:
            properties['molecular_weights'].append(Descriptors.MolWt(mol))
            properties['logp_values'].append(Descriptors.MolLogP(mol))
            properties['qed_scores'].append(Descriptors.qed(mol))
        except:
            continue
    
    # Compute diversity (average pairwise Tanimoto distance)
    diversity = 0.0
    if len(valid_mols) > 1:
        from rdkit.Chem import rdMolDescriptors
        from rdkit import DataStructs
        
        fps = [rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, 2) for mol in valid_mols]
        similarities = []
        
        for i in range(len(fps)):
            for j in range(i + 1, len(fps)):
                sim = DataStructs.TanimotoSimilarity(fps[i], fps[j])
                similarities.append(sim)
        
        diversity = 1.0 - np.mean(similarities) if similarities else 0.0
    
    return {
        'validity': validity,
        'uniqueness': uniqueness,
        'diversity': diversity,
        'num_valid': len(valid_mols),
        'num_unique': len(set(valid_smiles)),
        'properties': properties
    }
