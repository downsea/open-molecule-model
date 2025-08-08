import torch
import torch.nn.functional as F
import numpy as np


class CategoricalNoiseScheduler:
    """
    Noise scheduler for categorical features (atoms and bonds) in molecular graphs.
    Uses transition matrices for discrete diffusion on categorical variables.
    """
    
    def __init__(self, num_node_classes, num_edge_classes, num_timesteps=1000, beta_start=1e-4, beta_end=0.02, schedule='cosine'):
        self.num_node_classes = num_node_classes
        self.num_edge_classes = num_edge_classes
        self.num_timesteps = num_timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.schedule = schedule
        
        # Generate noise schedule
        self.betas = self._get_noise_schedule()
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        
        # Pre-compute transition matrices for nodes and edges
        self.node_transition_matrices = self._compute_transition_matrices(self.num_node_classes)
        self.edge_transition_matrices = self._compute_transition_matrices(self.num_edge_classes)
        
    def _get_noise_schedule(self):
        """Generate noise schedule (beta values)."""
        if self.schedule == 'linear':
            return torch.linspace(self.beta_start, self.beta_end, self.num_timesteps)
        elif self.schedule == 'cosine':
            return self._cosine_beta_schedule()
        elif self.schedule == 'quadratic':
            return torch.linspace(self.beta_start**0.5, self.beta_end**0.5, self.num_timesteps) ** 2
        else:
            raise ValueError(f"Unknown schedule: {self.schedule}")
    
    def _cosine_beta_schedule(self, s=0.008):
        """Cosine beta schedule as proposed in improved DDPM."""
        steps = self.num_timesteps + 1
        x = torch.linspace(0, self.num_timesteps, steps)
        alphas_cumprod = torch.cos(((x / self.num_timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)
    
    def _compute_transition_matrices(self, num_classes):
        """
        Compute transition matrices for categorical diffusion.
        
        Args:
            num_classes: Number of categorical classes
            
        Returns:
            Transition matrices for each timestep [num_timesteps, num_classes, num_classes]
        """
        transition_matrices = []
        
        for t in range(self.num_timesteps):
            beta_t = self.betas[t]
            
            # Create transition matrix: high probability of staying same, small uniform probability of changing
            transition_matrix = torch.eye(num_classes) * (1 - beta_t) + torch.ones(num_classes, num_classes) * (beta_t / num_classes)
            
            # Ensure rows sum to 1 (probabilities)
            transition_matrix = transition_matrix / transition_matrix.sum(dim=1, keepdim=True)
            transition_matrices.append(transition_matrix)
            
        return torch.stack(transition_matrices)
    
    def q_sample(self, x_0, t, noise=None):
        """
        Sample from q(x_t | x_0) - add noise to clean data.
        
        Args:
            x_0: Clean categorical data [batch_size, num_features] or [num_nodes]
            t: Timestep [batch_size] or scalar
            noise: Optional noise tensor
            
        Returns:
            x_t: Noised data at timestep t
        """
        if isinstance(t, int):
            t = torch.tensor([t])
        
        device = x_0.device
        batch_size = x_0.shape[0]
        
        # Determine if this is node or edge data based on shape
        num_classes = self.num_node_classes if len(x_0.shape) == 1 else self.num_edge_classes
        transition_matrices = self.node_transition_matrices if len(x_0.shape) == 1 else self.edge_transition_matrices
        
        x_t = torch.zeros_like(x_0)
        
        for i in range(batch_size):
            t_i = t[i] if t.numel() > 1 else t.item()
            transition_matrix = transition_matrices[t_i].to(device)
            
            # Convert to one-hot
            x_0_onehot = F.one_hot(x_0[i], num_classes=num_classes).float()
            
            # Apply transition matrix
            x_t_probs = torch.matmul(x_0_onehot, transition_matrix.T)
            
            # Sample from categorical distribution
            x_t[i] = torch.multinomial(x_t_probs, 1).squeeze(-1)
            
        return x_t
    
    def q_posterior(self, x_0, x_t, t):
        """
        Compute q(x_{t-1} | x_0, x_t) for reverse process.
        
        Args:
            x_0: Original clean data
            x_t: Noisy data at timestep t
            t: Current timestep
            
        Returns:
            Posterior probabilities
        """
        device = x_0.device
        num_classes = self.num_node_classes if len(x_0.shape) == 1 else self.num_edge_classes
        transition_matrices = self.node_transition_matrices if len(x_0.shape) == 1 else self.edge_transition_matrices
        
        if t == 0:
            return x_0
            
        # Get transition matrices
        Q_t = transition_matrices[t].to(device)
        Q_t_minus_1 = transition_matrices[t-1].to(device)
        
        # Compute posterior using Bayes' theorem
        x_0_onehot = F.one_hot(x_0, num_classes=num_classes).float()
        x_t_onehot = F.one_hot(x_t, num_classes=num_classes).float()
        
        # Posterior = P(x_{t-1} | x_0, x_t) ∝ P(x_t | x_{t-1}) * P(x_{t-1} | x_0)
        posterior = torch.matmul(x_t_onehot, Q_t.T) * torch.matmul(x_0_onehot, Q_t_minus_1.T)
        posterior = posterior / posterior.sum(dim=-1, keepdim=True)
        
        return posterior
    
    def sample_timesteps(self, batch_size, device):
        """Sample random timesteps for training."""
        return torch.randint(0, self.num_timesteps, (batch_size,), device=device)
    
    def sample(self, model, num_samples, num_nodes_range=(10, 50), device='cuda'):
        """
        Generate samples using the reverse diffusion process.
        
        Args:
            model: Trained GraphDiT model
            num_samples: Number of molecules to generate
            num_nodes_range: Range for number of nodes in generated graphs
            device: Device to run generation on
            
        Returns:
            Generated samples
        """
        model.eval()
        samples = []
        
        with torch.no_grad():
            for _ in range(num_samples):
                # Initialize random graph
                num_nodes = np.random.randint(num_nodes_range[0], num_nodes_range[1])
                
                # Start with random categorical features
                x_t = torch.randint(0, self.num_node_classes, (num_nodes,), device=device)
                
                # Generate random edge indices (this is a simplified approach)
                # In practice, you might want to use a more sophisticated edge generation strategy
                edge_index = self._generate_random_edges(num_nodes, device)
                edge_attr = torch.randint(0, self.num_edge_classes, (edge_index.shape[1],), device=device)
                
                # Reverse diffusion process
                for t in reversed(range(self.num_timesteps)):
                    # Prepare batch
                    batch = torch.zeros(num_nodes, device=device, dtype=torch.long)
                    timesteps = torch.tensor([t], device=device)
                    
                    # Get model predictions
                    node_logits, edge_logits = model(x_t.unsqueeze(-1).float(), 
                                                     edge_index, 
                                                     edge_attr.unsqueeze(-1).float(), 
                                                     batch, 
                                                     timesteps)
                    
                    # Sample next state
                    node_probs = F.softmax(node_logits, dim=-1)
                    x_t = torch.multinomial(node_probs, 1).squeeze(-1)
                    
                    edge_probs = F.softmax(edge_logits, dim=-1)
                    edge_attr = torch.multinomial(edge_probs, 1).squeeze(-1)
                
                samples.append({
                    'x': x_t.cpu().numpy(),
                    'edge_index': edge_index.cpu().numpy(),
                    'edge_attr': edge_attr.cpu().numpy()
                })
        
        return samples
    
    def _generate_random_edges(self, num_nodes, device, max_edges_per_node=4):
        """Generate random edges for initialization."""
        edges = []
        for i in range(num_nodes):
            num_edges = np.random.randint(0, min(max_edges_per_node, num_nodes))
            targets = np.random.choice([j for j in range(num_nodes) if j != i], 
                                     size=num_edges, replace=False)
            for target in targets:
                edges.append([i, target])
        
        if not edges:
            edges = [[0, 0]]  # Handle empty graphs
            
        edge_index = torch.tensor(edges, device=device).T
        return edge_index


class DiscreteDiffusionLoss:
    """Loss function for discrete diffusion training."""
    
    @staticmethod
    def __call__(pred_logits, target, mask=None):
        """
        Compute cross-entropy loss for categorical features.
        
        Args:
            pred_logits: Predicted logits [batch_size, num_classes]
            target: Target categorical values [batch_size]
            mask: Optional mask [batch_size]
            
        Returns:
            Loss value
        """
        loss = F.cross_entropy(pred_logits, target, reduction='none')
        
        if mask is not None:
            loss = loss * mask
            loss = loss.sum() / mask.sum()
        else:
            loss = loss.mean()
            
        return loss