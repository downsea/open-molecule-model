
import torch
import torch.nn as nn
import math
import torch.nn.functional as F

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class RelativePositionalEncoding(nn.Module):
    def __init__(self, d_model, max_relative_position=128):
        super(RelativePositionalEncoding, self).__init__()
        self.d_model = d_model
        self.max_relative_position = max_relative_position
        
        # Relative position embeddings
        self.relative_position_bias = nn.Embedding(
            2 * max_relative_position + 1, d_model
        )
        
    def forward(self, length):
        """Generate relative position embeddings for sequences of given length."""
        device = next(self.parameters()).device
        range_vec = torch.arange(length, dtype=torch.long, device=device)
        distance_mat = range_vec.unsqueeze(0) - range_vec.unsqueeze(1)
        distance_mat_clipped = torch.clamp(
            distance_mat, -self.max_relative_position, self.max_relative_position
        )
        final_mat = distance_mat_clipped + self.max_relative_position
        
        return self.relative_position_bias(final_mat)

class TransformerDecoder(nn.Module):
    def __init__(self, latent_dim, output_dim, hidden_dim, num_heads, num_layers,
                 use_gradient_checkpointing=False, max_seq_length=512):
        super(TransformerDecoder, self).__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.max_seq_length = max_seq_length

        # Optimized positional encoding with caching
        self.pos_encoder = PositionalEncoding(hidden_dim, max_len=max_seq_length)
        
        # Use standard transformer decoder with optimizations
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,  # Standard 4x expansion
            dropout=0.1,
            activation='gelu',  # GELU often works better than ReLU
            batch_first=True,  # More efficient for modern PyTorch
            norm_first=True    # Pre-norm for better training stability
        )
        
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_layers
        )
        
        # Optimized projection layers with proper initialization
        self.input_proj = nn.Linear(output_dim, hidden_dim)
        self.fc_latent = nn.Linear(latent_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, output_dim)
        
        # Layer normalization for stability
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
        # Initialize weights properly
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Xavier/Glorot initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, latent_representation, condition_vector, tgt):
        """
        Optimized forward pass with better memory management.
        
        Args:
            latent_representation: (batch_size, latent_dim)
            condition_vector: (batch_size, latent_dim) - currently unused but kept for compatibility
            tgt: (batch_size, vocab_size, seq_len) - one-hot encoded target sequence
        
        Returns:
            output: (batch_size, vocab_size, seq_len) - logits for each position
        """
        batch_size, vocab_size, seq_len = tgt.shape
        
        # OPTIMIZATION: Convert one-hot to indices to save memory during processing
        # tgt: (batch_size, vocab_size, seq_len) -> (batch_size, seq_len)
        tgt_indices = tgt.argmax(dim=1)  # More memory efficient than transpose
        
        # Create proper embedding using the input projection layer
        # Convert indices to one-hot for proper linear transformation
        tgt_one_hot_for_embedding = F.one_hot(tgt_indices, num_classes=vocab_size).float()
        
        # Apply input projection to get embeddings
        # tgt_one_hot_for_embedding: (batch_size, seq_len, vocab_size)
        # input_proj expects: (*, vocab_size) -> (*, hidden_dim)
        tgt_embedded = self.input_proj(tgt_one_hot_for_embedding)
        
        # Apply layer normalization for stability
        tgt_embedded = self.layer_norm(tgt_embedded)
        
        # Add positional encoding (batch_first=True format)
        tgt_embedded = self.pos_encoder(tgt_embedded)
        
        # Project latent representation to memory
        # Expand to sequence length for attention
        memory = self.fc_latent(latent_representation)  # (batch_size, hidden_dim)
        memory = memory.unsqueeze(1).expand(-1, seq_len, -1)  # (batch_size, seq_len, hidden_dim)
        
        # Create causal mask for autoregressive generation
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(tgt.device)
        
        # Apply transformer decoder with gradient checkpointing if enabled
        if self.use_gradient_checkpointing and self.training:
            from torch.utils.checkpoint import checkpoint
            output = checkpoint(
                self.transformer_decoder,
                tgt_embedded,
                memory,
                tgt_mask
            )
        else:
            output = self.transformer_decoder(
                tgt_embedded,
                memory,
                tgt_mask=tgt_mask
            )
        
        # Final projection to vocabulary size
        output = self.fc_out(output)  # (batch_size, seq_len, vocab_size)
        
        # Transpose to match expected output format: (batch_size, vocab_size, seq_len)
        output = output.transpose(1, 2)
        
        return output
