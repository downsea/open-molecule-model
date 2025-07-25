
import torch
import torch.nn as nn
from torch_geometric.nn import TransformerConv

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
        
        # We need to handle the case where num_layers is less than 10
        # For now, we assume num_layers is at least 10
        if self.num_layers < 10:
            # Simple case: just use all layers
            selected_outputs = layer_outputs
        else:
            selected_outputs = [layer_outputs[i] for i in selected_layers]


        # The output should be a latent matrix of size 8x256.
        # This requires more specific pooling and concatenation logic.
        # For now, we will just concatenate and pool.
        x = torch.cat(selected_outputs, dim=1)
        
        # This is a placeholder for the correct pooling logic
        # to get to the 8x256 latent matrix.
        # We will need to implement a more sophisticated pooling strategy.
        # For now, we will just do a global add pool.
        from torch_geometric.nn import global_add_pool
        x = global_add_pool(x, batch)

        return x
