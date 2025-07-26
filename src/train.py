import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import os
import selfies as sf
from torch.cuda.amp import GradScaler, autocast

from .data_loader import ZINC_Dataset, StreamingZINCDataset
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
    use_streaming = getattr(config.data, 'use_streaming', True)
    cache_in_memory = getattr(config.data, 'cache_in_memory', False)
    
    if use_streaming:
        from torch.utils.data import DataLoader as TorchDataLoader
        dataset = StreamingZINCDataset(
            config.data.dataset_path,
            max_length=config.data.max_length,
            shuffle_files=True
        )
        dataloader = TorchDataLoader(
            dataset, 
            batch_size=config.data.batch_size,
            num_workers=config.system.num_workers
        )
    else:
        dataset = ZINC_Dataset(
            config.data.dataset_path,
            max_length=config.data.max_length,
            cache_in_memory=cache_in_memory
        )
        dataloader = DataLoader(
            dataset, 
            batch_size=config.data.batch_size, 
            shuffle=not cache_in_memory,  # Shuffle if not streaming
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

    # --- Mixed Precision Training ---
    use_mixed_precision = getattr(config.system, 'mixed_precision', True)
    scaler = GradScaler() if use_mixed_precision and device.type == 'cuda' else None

    # --- Gradient Accumulation ---
    gradient_accumulation_steps = getattr(config.training, 'gradient_accumulation_steps', 1)
    effective_batch_size = config.data.batch_size * gradient_accumulation_steps
    print(f"Effective batch size: {effective_batch_size} (batch_size={config.data.batch_size}, accumulation_steps={gradient_accumulation_steps})")

    # --- Load Checkpoint ---
    start_epoch, last_loss = load_checkpoint(model, optimizer, config.paths.checkpoint_path)

    # --- Memory Optimization ---
    def get_gpu_memory_usage():
        if torch.cuda.is_available():
            return {
                'allocated': torch.cuda.memory_allocated() / 1024**3,  # GB
                'cached': torch.cuda.memory_reserved() / 1024**3,  # GB
                'max_allocated': torch.cuda.max_memory_allocated() / 1024**3,  # GB
            }
        return {}

    # Clear GPU cache before training
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"Initial GPU memory: {get_gpu_memory_usage()}")

    # --- Comprehensive Training Information ---
    def format_memory(bytes_value):
        """Format bytes to human readable string."""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if bytes_value < 1024.0:
                return f"{bytes_value:.2f} {unit}"
            bytes_value /= 1024.0
        return f"{bytes_value:.2f} TB"

    def get_model_size(model):
        """Calculate model size in memory."""
        param_size = 0
        param_count = 0
        buffer_size = 0
        buffer_count = 0
        
        for param in model.parameters():
            param_count += param.nelement()
            param_size += param.nelement() * param.element_size()
        
        for buffer in model.buffers():
            buffer_count += buffer.nelement()
            buffer_size += buffer.nelement() * buffer.element_size()
        
        total_size = param_size + buffer_size
        return {
            'parameters': param_count,
            'param_size': param_size,
            'buffers': buffer_count,
            'buffer_size': buffer_size,
            'total_size': total_size,
            'total_size_str': format_memory(total_size)
        }

    # Calculate batch information
    actual_batch_size = config.data.batch_size
    effective_batch_size = actual_batch_size * gradient_accumulation_steps
    
    # Calculate dataset information
    dataset_info = ""
    if use_streaming:
        dataset_info = f"Streaming mode enabled - processing {len(dataset.processed_files)} data files on-the-fly"
        total_samples = "Unknown (streaming)"
        # For streaming datasets, we estimate steps based on file sizes
        estimated_samples = 0
        for file_path in dataset.processed_files:
            try:
                data = torch.load(file_path, weights_only=False)
                estimated_samples += len(data)
            except:
                pass
        total_steps_per_epoch = estimated_samples // config.data.batch_size // gradient_accumulation_steps
    else:
        total_samples = len(dataset)
        dataset_info = f"{total_samples:,} total samples loaded from {len(dataset.processed_files)} files"
        total_steps_per_epoch = len(dataloader) // gradient_accumulation_steps
    
    # Model size calculation
    model_info = get_model_size(model)
    
    # Memory information
    gpu_memory_info = ""
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(device)
        gpu_memory_total = torch.cuda.get_device_properties(device).total_memory
        gpu_memory_info = f"""
    📊 GPU Information:
       Device: {gpu_name}
       Total Memory: {format_memory(gpu_memory_total)}
       Model Size: {model_info['total_size_str']}
       Parameters: {model_info['parameters']:,}
       Buffers: {model_info['buffers']:,}"""

    print(f"""
    ╔══════════════════════════════════════════════════════════════════════════════╗
    ║                        🧬 PanGu Drug Model Training                         ║
    ╚══════════════════════════════════════════════════════════════════════════════╝
    
    📁 DATASET INFORMATION:
       Dataset Path: {config.data.dataset_path}
       {dataset_info}
       Max Sequence Length: {config.data.max_length}
       
    ⚙️  BATCH CONFIGURATION:
       Actual Batch Size: {actual_batch_size}
       Gradient Accumulation Steps: {gradient_accumulation_steps}
       Effective Batch Size: {effective_batch_size}
       Steps per Epoch: {total_steps_per_epoch:,}
       Total Epochs: {config.training.num_epochs}
       
    🧮 MODEL ARCHITECTURE:
       Encoder Layers: {config.model.num_encoder_layers}
       Encoder Heads: {config.model.num_encoder_heads}
       Decoder Layers: {config.model.num_decoder_layers}
       Decoder Heads: {config.model.num_decoder_heads}
       Hidden Dimension: {config.model.hidden_dim}
       Latent Dimension: {config.model.latent_dim}
       Vocabulary Size: {config.model.output_dim}
       {gpu_memory_info}
    
    🎯 OPTIMIZATION SETTINGS:
       Learning Rate: {config.training.learning_rate}
       Weight Decay: {config.optimization.weight_decay}
       Gradient Clipping: {config.training.gradient_clip}
       Beta (KL Weight): {config.training.beta}
       
    ⚡ PERFORMANCE OPTIMIZATIONS:
       Mixed Precision: {use_mixed_precision}
       Number of Workers: {config.system.num_workers}
       Pin Memory: {getattr(config.system, 'pin_memory', True)}
       Streaming Mode: {use_streaming}
       
    📊 TRAINING STATISTICS:
       Total Training Steps: {total_steps_per_epoch * config.training.num_epochs:,}
       Estimated Training Time: ~{(total_steps_per_epoch * config.training.num_epochs * 0.5):.1f} minutes (assuming 0.5s/step)
    
    ╔══════════════════════════════════════════════════════════════════════════════╗
    ║                        Starting Training Process...                          ║
    ╚══════════════════════════════════════════════════════════════════════════════╝
    """)

    # Log initial memory usage
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        initial_memory = get_gpu_memory_usage()
        print(f"    💾 Initial GPU Memory Usage: {initial_memory}")

    # --- Training Loop ---
    
    # --- Simplified Training Information ---
    print("=" * 80)
    print("🧬 PanGu Drug Model - Training Information")
    print("=" * 80)
    print(f"📊 DATASET: {'Streaming' if use_streaming else 'In-memory'}")
    print(f"📦 BATCH: {config.data.batch_size} × {gradient_accumulation_steps} = {effective_batch_size} effective")
    print(f"🧠 MODEL: {config.model.hidden_dim} hidden, {config.model.num_encoder_layers} encoder layers")
    print(f"💾 DEVICE: {device}")
    print(f"📁 PATHS: logs={config.paths.log_dir}, checkpoints={config.paths.checkpoint_path}")
    print("=" * 80)
    
    for epoch in range(start_epoch, config.training.num_epochs):
        model.train()
        total_loss = 0
        optimizer.zero_grad()
        
        # For streaming datasets, we can't get length, so we use a counter
        step_count = 0
        batch_count = 0
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{config.training.num_epochs}")
        for i, data in enumerate(progress_bar):
            if data is None:
                continue
                
            data = data.to(device)

            # --- Prepare target sequences ---
            batch_size = data.num_graphs if hasattr(data, 'num_graphs') else len(data)
            
            # Get SMILES strings from batch and convert to SELFIES
            if hasattr(data, 'smiles'):
                smiles_strings = data.smiles
            else:
                # Handle streaming dataset
                smiles_strings = [item.smiles for item in data]
            
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
            
            # --- Forward Pass with Mixed Precision ---
            if use_mixed_precision and scaler is not None:
                with autocast():
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
                    loss = loss / gradient_accumulation_steps
                
                scaler.scale(loss).backward()
                
                if (i + 1) % gradient_accumulation_steps == 0:
                    # Gradient clipping
                    if config.training.gradient_clip > 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(), 
                            config.training.gradient_clip
                        )
                    
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
            else:
                # Standard training
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
                loss = loss / gradient_accumulation_steps
                
                loss.backward()
                
                if (i + 1) % gradient_accumulation_steps == 0:
                    # Gradient clipping
                    if config.training.gradient_clip > 0:
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(), 
                            config.training.gradient_clip
                        )
                    
                    optimizer.step()
                    optimizer.zero_grad()
            
            total_loss += loss.item() * gradient_accumulation_steps
            progress_bar.set_postfix(loss=loss.item() * gradient_accumulation_steps)

            # Update counters
            step_count += 1
            
            # --- TensorBoard Logging (per batch) ---
            global_step = epoch * total_steps_per_epoch + batch_count
            if (i + 1) % gradient_accumulation_steps == 0:
                writer.add_scalar('Loss/train_batch', loss.item() * gradient_accumulation_steps, global_step)
                batch_count += 1

        avg_loss = total_loss / max(step_count, 1)
        print(f"Epoch {epoch+1}/{config.training.num_epochs}, Average Loss: {avg_loss:.4f}")
        
        # --- Memory Logging ---
        if torch.cuda.is_available():
            memory_usage = get_gpu_memory_usage()
            writer.add_scalar('Memory/allocated_gb', memory_usage.get('allocated', 0), epoch)
            writer.add_scalar('Memory/cached_gb', memory_usage.get('cached', 0), epoch)
            writer.add_scalar('Memory/max_allocated_gb', memory_usage.get('max_allocated', 0), epoch)
            
            # Clear cache periodically
            if epoch % 5 == 0:
                torch.cuda.empty_cache()
        
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
    
    # Final memory report
    if torch.cuda.is_available():
        final_memory = get_gpu_memory_usage()
        print(f"Final GPU memory usage: {final_memory}")
    
    return config

if __name__ == "__main__":
    train()