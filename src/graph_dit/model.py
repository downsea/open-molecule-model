import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_mean_pool
from torch_geometric.data import Batch
import math


class SinusoidalPositionEmbedding(nn.Module):
    """Sinusoidal position embedding for timestep encoding."""
    
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, timesteps):
        """
        Args:
            timesteps: Tensor of shape [batch_size] containing timesteps
        Returns:
            Tensor of shape [batch_size, dim] containing positional embeddings
        """
        device = timesteps.device
        half_dim = self.dim // 2
        
        # Clamp timesteps to prevent out-of-bounds issues
        timesteps = torch.clamp(timesteps, min=0, max=999)
        
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device, dtype=torch.float32) * -embeddings)
        embeddings = timesteps.float()[:, None] * embeddings[None, :]
        embeddings = torch.cat([embeddings.sin(), embeddings.cos()], dim=-1)
        return embeddings


class GraphTransformerBlock(nn.Module):
    """Graph Transformer block using GATv2Conv with edge features."""
    
    def __init__(self, hidden_dim, num_heads, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        
        # Multi-head graph attention
        self.attention = GATv2Conv(
            hidden_dim, 
            hidden_dim // num_heads, 
            heads=num_heads, 
            edge_dim=hidden_dim,
            dropout=dropout,
            concat=True
        )
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout)
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
    def forward(self, x, edge_index, edge_attr, batch):
        """
        Args:
            x: Node features [num_nodes, hidden_dim]
            edge_index: Graph connectivity [2, num_edges]
            edge_attr: Edge features [num_edges, hidden_dim]
            batch: Batch vector [num_nodes]
        Returns:
            Updated node features [num_nodes, hidden_dim]
        """
        # Multi-head attention with residual connection
        attn_out = self.attention(x, edge_index, edge_attr)
        x = self.norm1(x + attn_out)
        
        # Feed-forward with residual connection
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        
        return x


class GraphDiT(nn.Module):
    """Graph Diffusion Transformer for molecular generation."""
    
    def __init__(
        self,
        node_dim,
        edge_dim,
        hidden_dim=256,
        num_layers=8,
        num_heads=8,
        max_time_steps=1000,
        dropout=0.1
    ):
        super().__init__()
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.max_time_steps = max_time_steps
        
        # Input embeddings for categorical features
        self.node_embed = nn.Embedding(node_dim, hidden_dim)
        self.edge_embed = nn.Embedding(edge_dim, hidden_dim)
        
        # Timestep embedding
        self.time_embed = SinusoidalPositionEmbedding(hidden_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
        
        # Transformer layers
        self.transformer_layers = nn.ModuleList([
            GraphTransformerBlock(hidden_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])
        
        # Output heads for atom and bond prediction
        self.node_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, node_dim)
        )
        
        self.edge_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, edge_dim)
        )
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        """Initialize weights using Xavier uniform."""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
                
    def forward(self, x, edge_index, edge_attr, batch, timesteps):
        """
        Forward pass through the GraphDiT model.
        
        Args:
            x: Node indices [num_nodes] (categorical)
            edge_index: Graph connectivity [2, num_edges]
            edge_attr: Edge indices [num_edges] (categorical)
            batch: Batch vector [num_nodes]
            timesteps: Timestep tensor [batch_size]
            
        Returns:
            pred_node_logits: Predicted node logits [num_nodes, node_dim]
            pred_edge_logits: Predicted edge logits [num_edges, edge_dim]
        """
        # Clamp inputs to prevent out-of-bounds embedding access
        x = torch.clamp(x.long(), min=0, max=self.node_dim - 1)
        edge_attr = torch.clamp(edge_attr.long(), min=0, max=self.edge_dim - 1)
        timesteps = torch.clamp(timesteps.long(), min=0, max=self.max_time_steps - 1)
        
        # Embed inputs
        node_emb = self.node_embed(x)
        edge_emb = self.edge_embed(edge_attr)
        
        # Embed timesteps
        time_emb = self.time_embed(timesteps)  # [batch_size, hidden_dim]
        time_emb = self.time_mlp(time_emb)     # [batch_size, hidden_dim]
        
        # Add timestep embedding to nodes
        time_emb_expanded = time_emb[batch]  # [num_nodes, hidden_dim]
        node_emb = node_emb + time_emb_expanded
        
        # Pass through transformer layers
        for layer in self.transformer_layers:
            node_emb = layer(node_emb, edge_index, edge_emb, batch)
            
        # Predict outputs
        pred_node_logits = self.node_head(node_emb)
        
        # For edge prediction, we need to handle edge features
        # Simple approach: use node embeddings to predict edge features
        edge_start = node_emb[edge_index[0]]
        edge_end = node_emb[edge_index[1]]
        edge_features = (edge_start + edge_end) / 2
        pred_edge_logits = self.edge_head(edge_features)
        
        return pred_node_logits, pred_edge_logits
    
    def get_graph_embedding(self, x, edge_index, edge_attr, batch, timesteps):
        """
        Extract graph-level embeddings for downstream tasks.
        
        Args:
            x: Node features [num_nodes, node_dim]
            edge_index: Graph connectivity [2, num_edges]
            edge_attr: Edge features [num_edges, edge_dim]
            batch: Batch vector [num_nodes]
            timesteps: Timestep tensor [batch_size]
            
        Returns:
            Graph embeddings [batch_size, hidden_dim]
        """
        # Get final node embeddings
        with torch.no_grad():
            node_emb = self.node_embed(x)
            edge_emb = self.edge_embed(edge_attr)
            
            time_emb = self.time_embed(timesteps)
            time_emb = self.time_mlp(time_emb)
            time_emb_expanded = time_emb[batch]
            node_emb = node_emb + time_emb_expanded
            
            for layer in self.transformer_layers:
                node_emb = layer(node_emb, edge_index, edge_emb, batch)
                
        # Global mean pooling to get graph-level embedding
        graph_emb = global_mean_pool(node_emb, batch)
        return graph_emb


class PropertyPredictionHead(nn.Module):
    """MLP head for molecule property prediction using frozen GraphDiT."""
    
    def __init__(self, input_dim, hidden_dims, output_dim, dropout=0.1):
        super().__init__()
        layers = []
        dims = [input_dim] + hidden_dims + [output_dim]
        
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:  # Not the last layer
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))
                
        self.mlp = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.mlp(x)


class CriticModel(nn.Module):
    """Critic model for guided molecule optimization."""
    
    def __init__(self, node_dim, edge_dim, hidden_dim=128, num_layers=3):
        super().__init__()
        self.gnn = nn.ModuleList([
            GATv2Conv(node_dim, hidden_dim // 4, heads=4, edge_dim=edge_dim)
        ])
        
        for _ in range(num_layers - 1):
            self.gnn.append(
                GATv2Conv(hidden_dim, hidden_dim // 4, heads=4, edge_dim=edge_dim)
            )
            
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
    def forward(self, x, edge_index, edge_attr, batch):
        for layer in self.gnn:
            x = layer(x, edge_index, edge_attr)
            x = F.relu(x)
            
        # Global mean pooling
        x = global_mean_pool(x, batch)
        return self.predictor(x).squeeze(-1)