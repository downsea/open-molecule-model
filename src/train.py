import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import os
import selfies as sf

from .data_loader import ZINC_Dataset
from .model import PanGuDrugModel, vae_loss
from .utils import SELFIESProcessor, create_condition_vector
from .config import Config

def save_checkpoint(model, optimizer, epoch, loss, checkpoint_path):
    """Save model checkpoint."""
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, checkpoint_path)

def load_checkpoint(model, optimizer, checkpoint_path):
    """Load model checkpoint."""
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        print(f"Resuming from epoch {epoch} with loss {loss}")
        return epoch, loss
    else:
        print("No checkpoint found, starting from scratch.")
        return 0, 0.0

def train(config=None):
    """Train the PanGu Drug Model."""
    # Load configuration
    if config is None:
        config = Config.from_args()
    
    # Override device based on availability
    device = torch.device(config.system.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    config.system.device = str(device)
    config.evaluation.device = str(device)

    # --- TensorBoard ---
    writer = SummaryWriter(config.paths.log_dir)

    # --- Dataset and DataLoader ---
    dataset = ZINC_Dataset(config.data.dataset_path)
    dataloader = DataLoader(
        dataset, 
        batch_size=config.data.batch_size, 
        shuffle=True,
        num_workers=config.system.num_workers
    )

    # Initialize SELFIES processor
    selfies_processor = SELFIESProcessor()
    config.model.output_dim = selfies_processor.get_vocab_size()
    
    # --- Model ---
    model = PanGuDrugModel(
        num_node_features=config.model.num_node_features,
        hidden_dim=config.model.hidden_dim,
        num_encoder_layers=config.model.num_encoder_layers,
        num_encoder_heads=config.model.num_encoder_heads,
        output_dim=config.model.output_dim,
        num_decoder_heads=config.model.num_decoder_heads,
        num_decoder_layers=config.model.num_decoder_layers,
        latent_dim=config.model.latent_dim,
        num_selected_layers=config.model.num_selected_layers
    ).to(device)

    # --- Optimizer ---
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=config.training.learning_rate,
        weight_decay=config.optimization.weight_decay
    )

    # --- Load Checkpoint ---
    start_epoch, last_loss = load_checkpoint(model, optimizer, config.paths.checkpoint_path)

    # --- Training Loop ---
    print("Starting training...")
    print("Configuration:")
    print(config)
    
    for epoch in range(start_epoch, config.training.num_epochs):
        model.train()
        total_loss = 0
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{config.training.num_epochs}")
        for i, data in enumerate(progress_bar):
            data = data.to(device)

            # --- Prepare target sequences ---
            batch_size = data.num_graphs
            
            # Get SMILES strings from batch and convert to SELFIES
            smiles_strings = data.smiles
            selfies_strings = [sf.encoder(smiles) for smiles in smiles_strings]
            
            # Encode SELFIES strings to tensors
            target_tensors = selfies_processor.encode_batch(
                selfies_strings, 
                max_length=config.data.max_length
            )
            target_tensors = target_tensors.to(device)
            
            # Create one-hot encoded targets for decoder
            tgt_one_hot = F.one_hot(
                target_tensors[:, :-1], 
                num_classes=config.model.output_dim
            ).float()
            tgt_one_hot = tgt_one_hot.transpose(1, 2)  # (batch, vocab, seq_len)
            
            # --- Forward Pass ---
            optimizer.zero_grad()
            
            # Create condition vectors (placeholder for now)
            condition_vector = create_condition_vector(
                batch_size, 
                config.model.latent_dim, 
                device
            )

            output, mean, log_var = model(data, condition_vector, tgt_one_hot)
            
            # Prepare targets for loss calculation
            targets = target_tensors[:, 1:]  # Shifted targets
            loss = vae_loss(
                output, 
                targets, 
                mean, 
                log_var, 
                beta=config.training.beta
            )
            
            # --- Backward Pass ---
            loss.backward()
            
            # Gradient clipping
            if config.training.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), 
                    config.training.gradient_clip
                )
            
            optimizer.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())

            # --- TensorBoard Logging (per batch) ---
            global_step = epoch * len(dataloader) + i
            writer.add_scalar('Loss/train_batch', loss.item(), global_step)

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{config.training.num_epochs}, Average Loss: {avg_loss:.4f}")
        
        # --- TensorBoard Logging (per epoch) ---
        writer.add_scalar('Loss/train_epoch_avg', avg_loss, epoch)
        
        # --- Save Checkpoint ---
        save_checkpoint(
            model, 
            optimizer, 
            epoch + 1, 
            avg_loss, 
            config.paths.checkpoint_path
        )

    writer.close()
    print("Training complete.")
    return config

if __name__ == "__main__":
    train()