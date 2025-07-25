
import torch
import torch.nn as nn
from torch_geometric.nn import TransformerConv, global_mean_pool

class GraphTransformerEncoder(nn.Module):
    def __init__(self, num_node_features, hidden_dim, num_layers, num_heads):
        super(GraphTransformerEncoder, self).__init__()
        self.num_layers = num_layers
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        for i in range(num_layers):
            in_channels = num_node_features if i == 0 else hidden_dim * num_heads
            self.convs.append(TransformerConv(in_channels, hidden_dim, heads=num_heads))
            self.bns.append(nn.BatchNorm1d(hidden_dim * num_heads))

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        layer_outputs = []
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            x = self.bns[i](x)
            x = torch.relu(x)
            layer_outputs.append(x)

        # Concatenate representations from specified layers
        # As per architecture.md: layers 1, 2, 3, 4, 5, 6, 8, 10
        # Using 0-based indexing: 0, 1, 2, 3, 4, 5, 7, 9
        selected_layers = [0, 1, 2, 3, 4, 5, 7, 9]
        
        # Ensure we have at least the required number of layers
        # If not, use available layers and repeat the last one
        available_layers = len(layer_outputs)
        if available_layers < 10:
            # Use available layers and pad with the last layer
            selected_outputs = []
            for i in range(8):  # We need 8 layers
                if i < available_layers:
                    selected_outputs.append(layer_outputs[i])
                else:
                    selected_outputs.append(layer_outputs[-1])
        else:
            selected_outputs = [layer_outputs[i] for i in selected_layers]

        # Create 8x256 latent matrix by pooling each layer's output separately
        # Each layer output is (num_nodes, hidden_dim * num_heads)
        # We need to pool each layer to get (batch_size, hidden_dim * num_heads)
        pooled_outputs = []
        for layer_output in selected_outputs:
            # Pool each layer to graph-level representation
            pooled = global_mean_pool(layer_output, batch)
            pooled_outputs.append(pooled)
        
        # Stack to create 8x256 matrix: (batch_size, 8, hidden_dim * num_heads)
        # Then reshape to (batch_size, 8 * hidden_dim * num_heads)
        latent_matrix = torch.stack(pooled_outputs, dim=1)  # (batch_size, 8, hidden_dim * num_heads)
        
        # Flatten to get final representation
        x = latent_matrix.view(latent_matrix.size(0), -1)  # (batch_size, 8 * hidden_dim * num_heads)
        
        return x
