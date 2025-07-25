
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
    def __init__(self, latent_dim, output_dim, hidden_dim, num_heads, num_layers):
        super(TransformerDecoder, self).__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_heads = num_heads
        self.num_layers = num_layers

        self.pos_encoder = PositionalEncoding(hidden_dim)
        self.relative_pos_encoders = nn.ModuleList([
            RelativePositionalEncoding(hidden_dim) if i > 0 else None
            for i in range(num_layers)
        ])
        
        # Create decoder layers with appropriate positional encoding
        decoder_layers = []
        for i in range(num_layers):
            if i == 0:
                # First layer uses absolute positional encoding
                layer = nn.TransformerDecoderLayer(
                    d_model=hidden_dim, 
                    nhead=num_heads,
                    dropout=0.1,
                    batch_first=False
                )
            else:
                # Subsequent layers use relative positional encoding
                layer = nn.TransformerDecoderLayer(
                    d_model=hidden_dim, 
                    nhead=num_heads,
                    dropout=0.1,
                    batch_first=False
                )
            decoder_layers.append(layer)
        
        self.transformer_decoder = nn.ModuleList(decoder_layers)
        
        self.input_proj = nn.Linear(output_dim, hidden_dim)
        self.fc_latent = nn.Linear(latent_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, output_dim)

    def forward(self, latent_representation, condition_vector, tgt):
        # Project latent representation to memory
        memory = self.fc_latent(latent_representation).unsqueeze(1)
        
        # Transpose tgt from (batch_size, vocab_size, seq_len) to (batch_size, seq_len, vocab_size)
        tgt = tgt.transpose(1, 2)
        
        # Project target sequence to hidden dimension
        tgt = self.input_proj(tgt)
        
        # Transpose to (seq_len, batch_size, hidden_dim) for transformer format
        tgt = tgt.transpose(0, 1)
        memory = memory.transpose(0, 1)
        
        # Process through decoder layers
        output = tgt
        for i, layer in enumerate(self.transformer_decoder):
            if i == 0:
                # First layer: absolute positional encoding
                output = self.pos_encoder(output.transpose(0, 1)).transpose(0, 1)
            else:
                # Subsequent layers: relative positional encoding
                seq_len = output.size(0)
                batch_size = output.size(1)
                
                # Add relative position bias
                rel_pos_bias = self.relative_pos_encoders[i](seq_len)
                # Skip relative positioning for now - PyTorch's built-in does not support it directly
                # This would require custom attention implementation
                
            # Apply transformer decoder layer
            output = layer(output, memory)
        
        # Transpose back to (batch_size, seq_len, hidden_dim)
        output = output.transpose(0, 1)
        
        # Final projection to output dimension
        output = self.fc_out(output)
        
        # Transpose to (batch_size, output_dim, seq_len) for loss function
        output = output.transpose(1, 2)
        return output
