
import torch
import torch.nn as nn
from .encoder import GraphTransformerEncoder
from .decoder import TransformerDecoder

class PanGuDrugModel(nn.Module):
    def __init__(self, num_node_features, hidden_dim, num_encoder_layers, num_encoder_heads,
                 output_dim, num_decoder_heads, num_decoder_layers, latent_dim, num_selected_layers):
        super(PanGuDrugModel, self).__init__()
        self.encoder = GraphTransformerEncoder(num_node_features, hidden_dim, num_encoder_layers, num_encoder_heads)
        
        # Calculate encoder output dimension: 8 layers * hidden_dim * num_heads
        encoder_output_dim = num_selected_layers * hidden_dim * num_encoder_heads
        
        # Project to latent space dimensions
        self.fc_mean = nn.Linear(encoder_output_dim, latent_dim)
        self.fc_log_var = nn.Linear(encoder_output_dim, latent_dim)

        self.decoder = TransformerDecoder(latent_dim, output_dim, hidden_dim, num_decoder_heads, num_decoder_layers)

    def reparameterize(self, mean, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mean + eps * std

    def forward(self, data, condition_vector, tgt):
        # Encode the input graph
        encoded_features = self.encoder(data)
        
        # Get mean and log_var from the encoded features
        mean = self.fc_mean(encoded_features)
        log_var = self.fc_log_var(encoded_features)
        
        # Reparameterization trick
        z = self.reparameterize(mean, log_var)
        
        # Decode the latent vector
        output = self.decoder(z, condition_vector, tgt)
        
        return output, mean, log_var

def vae_loss(recon_x, x, mean, log_var, beta=1.0):
    """
    VAE loss for sequence generation using cross-entropy.
    
    Args:
        recon_x: Reconstructed logits, shape [batch_size, vocab_size, seq_len]
        x: Target token indices, shape [batch_size, seq_len]
        mean: Mean of latent distribution, shape [batch_size, latent_dim]
        log_var: Log variance of latent distribution, shape [batch_size, latent_dim]
        beta: Weight for KL divergence term
    
    Returns:
        Total loss = reconstruction_loss + beta * kl_div
    """
    batch_size = recon_x.size(0)
    
    # Cross-entropy reconstruction loss for sequences
    # recon_x: [batch_size, vocab_size, seq_len]
    # x: [batch_size, seq_len]
    
    # Ensure recon_x is [batch_size, vocab_size, seq_len] and x is [batch_size, seq_len]
    batch_size, vocab_size, seq_len = recon_x.shape
    
    # Reshape for cross-entropy: [batch_size * seq_len, vocab_size] and [batch_size * seq_len]
    recon_x_flat = recon_x.transpose(1, 2).contiguous().view(-1, vocab_size)
    x_flat = x.contiguous().view(-1)
    
    recon_loss = nn.functional.cross_entropy(
        recon_x_flat, 
        x_flat, 
        ignore_index=0,  # Ignore padding tokens
        reduction='mean'
    )
    
    # KL divergence with proper scaling
    kl_div = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp(), dim=1)
    kl_div = kl_div.mean()  # Average over batch
    
    return recon_loss + beta * kl_div
