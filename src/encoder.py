
import torch
import torch.nn as nn
from torch_geometric.nn import TransformerConv, global_mean_pool
from torch.utils.checkpoint import checkpoint

class GraphTransformerEncoder(nn.Module):
    def __init__(self, num_node_features, hidden_dim, num_layers, num_heads,
                 num_selected_layers=8, use_gradient_checkpointing=False):
        super(GraphTransformerEncoder, self).__init__()
        self.num_layers = num_layers
        self.num_selected_layers = num_selected_layers
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        
        # Validate configuration
        if num_layers < num_selected_layers:
            raise ValueError(f"num_layers ({num_layers}) must be >= num_selected_layers ({num_selected_layers})")
        
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.dropouts = nn.ModuleList()

        for i in range(num_layers):
            in_channels = num_node_features if i == 0 else hidden_dim * num_heads
            self.convs.append(TransformerConv(in_channels, hidden_dim, heads=num_heads, dropout=0.1))
            self.bns.append(nn.BatchNorm1d(hidden_dim * num_heads))
            self.dropouts.append(nn.Dropout(0.1))
        
        # Pre-compute layer selection indices for efficiency
        # As per architecture.md: layers 1, 2, 3, 4, 5, 6, 8, 10 (1-indexed)
        # Convert to 0-indexed: [0, 1, 2, 3, 4, 5, 7, 9]
        if num_layers >= 10:
            self.selected_layer_indices = [0, 1, 2, 3, 4, 5, 7, 9]
        else:
            # For fewer layers, select evenly distributed layers
            step = max(1, num_layers // num_selected_layers)
            self.selected_layer_indices = list(range(0, min(num_layers, num_selected_layers * step), step))
            # Ensure we have exactly num_selected_layers
            while len(self.selected_layer_indices) < num_selected_layers:
                self.selected_layer_indices.append(num_layers - 1)
            self.selected_layer_indices = self.selected_layer_indices[:num_selected_layers]
        
        print(f"Encoder using layers: {self.selected_layer_indices} (0-indexed)")

    def _forward_layer(self, layer_idx, x, edge_index):
        """Forward pass through a single layer with optional checkpointing."""
        def layer_forward():
            x_out = self.convs[layer_idx](x, edge_index)
            x_out = self.bns[layer_idx](x_out)
            x_out = torch.relu(x_out)
            x_out = self.dropouts[layer_idx](x_out)
            return x_out
        
        if self.use_gradient_checkpointing and self.training:
            return checkpoint(layer_forward)
        else:
            return layer_forward()

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # Store outputs from selected layers only (memory optimization)
        selected_outputs = []
        
        for i in range(self.num_layers):
            x = self._forward_layer(i, x, edge_index)
            
            # Only store outputs from selected layers
            if i in self.selected_layer_indices:
                # Pool immediately to save memory
                pooled = global_mean_pool(x, batch)
                selected_outputs.append(pooled)

        # CRITICAL OPTIMIZATION: Use direct concatenation instead of stack+view
        # This reduces memory usage and improves performance significantly
        x = torch.cat(selected_outputs, dim=1)  # (batch_size, num_selected_layers * hidden_dim * num_heads)
        
        return x
