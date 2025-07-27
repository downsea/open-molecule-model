import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import os
import json
import selfies as sf
from torch.cuda.amp import GradScaler
from torch.amp import autocast
import numpy as np
import time
import math
from collections import defaultdict, deque

from .data_loader import ZINC_Dataset, StreamingZINCDataset, create_optimized_dataloader
from .model import PanGuDrugModel, vae_loss, compute_molecular_metrics
from .utils import SELFIESProcessor, create_condition_vector
from .config import Config

class AdvancedLRScheduler:
    """Advanced learning rate scheduler with warmup and multiple decay strategies."""
    
    def __init__(self, optimizer, warmup_steps=1000, total_steps=10000,
                 scheduler_type='cosine', min_lr_ratio=0.01):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.scheduler_type = scheduler_type
        self.min_lr_ratio = min_lr_ratio
        self.base_lr = optimizer.param_groups[0]['lr']
        self.current_step = 0
        
    def step(self):
        """Update learning rate."""
        self.current_step += 1
        
        if self.current_step <= self.warmup_steps:
            # Warmup phase
            lr = self.base_lr * (self.current_step / self.warmup_steps)
        else:
            # Main scheduling phase
            progress = (self.current_step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            progress = min(progress, 1.0)
            
            if self.scheduler_type == 'cosine':
                lr = self.min_lr_ratio * self.base_lr + (self.base_lr - self.min_lr_ratio * self.base_lr) * \
                     0.5 * (1 + math.cos(math.pi * progress))
            elif self.scheduler_type == 'linear':
                lr = self.base_lr * (1 - progress * (1 - self.min_lr_ratio))
            elif self.scheduler_type == 'exponential':
                lr = self.base_lr * (self.min_lr_ratio ** progress)
            else:
                lr = self.base_lr
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        return lr

class EarlyStopping:
    """Early stopping with patience and best model saving."""
    
    def __init__(self, patience=10, min_delta=1e-4, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = float('inf')
        self.counter = 0
        self.best_weights = None
        
    def __call__(self, val_loss, model):
        """Check if training should stop."""
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            if self.restore_best_weights:
                self.best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            self.counter += 1
            
        if self.counter >= self.patience:
            if self.restore_best_weights and self.best_weights:
                model.load_state_dict(self.best_weights)
            return True
        return False

class TrainingMetrics:
    """Advanced training metrics tracking."""
    
    def __init__(self, window_size=100):
        self.window_size = window_size
        self.metrics = defaultdict(lambda: deque(maxlen=window_size))
        self.epoch_metrics = defaultdict(list)
        
    def update(self, **kwargs):
        """Update metrics."""
        for key, value in kwargs.items():
            if isinstance(value, torch.Tensor):
                value = value.item()
            self.metrics[key].append(value)
    
    def get_average(self, key):
        """Get moving average of metric."""
        if key in self.metrics and len(self.metrics[key]) > 0:
            return np.mean(list(self.metrics[key]))
        return 0.0
    
    def log_epoch(self, epoch):
        """Log epoch averages."""
        for key, values in self.metrics.items():
            if len(values) > 0:
                self.epoch_metrics[key].append(np.mean(list(values)))
        self.metrics.clear()

def save_checkpoint(model, optimizer, scheduler, epoch, loss, metrics, checkpoint_path):
    """Enhanced checkpoint saving with more information."""
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'metrics': metrics.epoch_metrics if metrics else {},
        'model_config': getattr(model, 'config', {}),
        'timestamp': time.time()
    }
    
    if scheduler:
        checkpoint['scheduler_state'] = {
            'current_step': scheduler.current_step,
            'base_lr': scheduler.base_lr
        }
    
    torch.save(checkpoint, checkpoint_path)
    
    # Save best model separately
    best_path = checkpoint_path.replace('.pt', '_best.pt')
    if not os.path.exists(best_path) or loss < torch.load(best_path, weights_only=True)['loss']:
        torch.save(checkpoint, best_path)

def load_checkpoint(model, optimizer, scheduler, checkpoint_path):
    """Enhanced checkpoint loading."""
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, weights_only=False)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        metrics = checkpoint.get('metrics', {})
        
        if scheduler and 'scheduler_state' in checkpoint:
            scheduler.current_step = checkpoint['scheduler_state']['current_step']
            scheduler.base_lr = checkpoint['scheduler_state']['base_lr']
        
        print(f"Resuming from epoch {epoch} with loss {loss:.4f}")
        return epoch, loss, metrics
    else:
        print("No checkpoint found, starting from scratch.")
        return 0, 0.0, {}

def train(config=None):
    """Enhanced training function with advanced optimizations."""
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
    
    # --- Initialize Training Metrics ---
    metrics = TrainingMetrics(window_size=100)

    # --- Dataset and DataLoader ---
    use_streaming = getattr(config.data, 'use_streaming', True)
    cache_in_memory = getattr(config.data, 'cache_in_memory', False)
    
    print("🔄 Setting up optimized data loading...")
    
    # Load training dataset
    if use_streaming:
        train_dataset = StreamingZINCDataset(
            config.data.train_dataset_path,
            max_length=config.data.max_length,
            shuffle_files=True,
            buffer_size=getattr(config.data, 'buffer_size', 1000)
        )
        train_dataloader = create_optimized_dataloader(
            train_dataset,
            batch_size=config.data.batch_size,
            num_workers=0,  # Disable multiprocessing for streaming on Windows
            shuffle=False,  # Streaming datasets handle their own shuffling
            pin_memory=getattr(config.system, 'pin_memory', True),
            persistent_workers=False
        )
    else:
        train_dataset = ZINC_Dataset(
            config.data.train_dataset_path,
            max_length=config.data.max_length,
            cache_in_memory=cache_in_memory,
            precompute_features=True
        )
        train_dataloader = create_optimized_dataloader(
            train_dataset,
            batch_size=config.data.batch_size,
            shuffle=not cache_in_memory,
            num_workers=config.system.num_workers,
            pin_memory=getattr(config.system, 'pin_memory', True),
            persistent_workers=getattr(config.system, 'persistent_workers', True)
        )
    
    # Load validation dataset
    if use_streaming:
        val_dataset = StreamingZINCDataset(
            config.data.val_dataset_path,
            max_length=config.data.max_length,
            shuffle_files=False,  # No shuffling for validation
            buffer_size=getattr(config.data, 'buffer_size', 1000)
        )
        val_dataloader = create_optimized_dataloader(
            val_dataset,
            batch_size=config.data.batch_size,
            num_workers=0,
            shuffle=False,
            pin_memory=getattr(config.system, 'pin_memory', True),
            persistent_workers=False
        )
    else:
        val_dataset = ZINC_Dataset(
            config.data.val_dataset_path,
            max_length=config.data.max_length,
            cache_in_memory=cache_in_memory,
            precompute_features=True
        )
        val_dataloader = create_optimized_dataloader(
            val_dataset,
            batch_size=config.data.batch_size,
            shuffle=False,
            num_workers=config.system.num_workers,
            pin_memory=getattr(config.system, 'pin_memory', True),
            persistent_workers=getattr(config.system, 'persistent_workers', True)
        )

    # Initialize SELFIES processor with dynamic vocabulary
    print("📝 Initializing SELFIES processor...")
    selfies_processor = SELFIESProcessor(max_vocab_size=2000)
    
    # Build vocabulary from data if needed
    vocab_file = os.path.join(config.data.train_dataset_path, "selfies_vocab.json")
    if not os.path.exists(vocab_file):
        print("🔄 Building vocabulary from training dataset...")
        # Sample some data to build vocabulary
        sample_selfies = []
        sample_count = 0
        for batch in train_dataloader:
            if batch is not None and hasattr(batch, 'smiles'):
                for smiles in batch.smiles[:10]:  # Sample 10 per batch
                    try:
                        selfies = sf.encoder(smiles)
                        sample_selfies.append(selfies)
                        sample_count += 1
                        if sample_count >= 10000:  # Enough for vocab building
                            break
                    except:
                        continue
                if sample_count >= 10000:
                    break
        
        if sample_selfies:
            selfies_processor.update_vocab_from_data(sample_selfies)
            selfies_processor.save_vocab(vocab_file)
    else:
        selfies_processor = SELFIESProcessor(vocab_file=vocab_file)
    
    config.model.output_dim = selfies_processor.get_vocab_size()
    print(f"✅ Vocabulary size: {config.model.output_dim}")
    
    # Load data analysis results if available
    analysis_path = "data/data_report/analysis_results.json"
    if os.path.exists(analysis_path):
        print("📊 Loading data analysis results...")
        with open(analysis_path, 'r') as f:
            analysis = json.load(f)
            
        # Use analysis recommendations if available
        recommendations = analysis.get('recommendations', {})
        if recommendations:
            original_batch = config.data.batch_size
            config.data.batch_size = recommendations.get('batch_size', config.data.batch_size)
            config.data.max_length = recommendations.get('max_sequence_length', config.data.max_length)
            config.training.num_epochs = recommendations.get('num_epochs', config.training.num_epochs)
            
            if original_batch != config.data.batch_size:
                print(f"⚙️  Updated batch size from analysis: {original_batch} → {config.data.batch_size}")
            if config.data.max_length != 128:
                print(f"⚙️  Updated max length from analysis: {config.data.max_length}")
            if config.training.num_epochs != 10:
                print(f"⚙️  Updated epochs from analysis: {config.training.num_epochs}")
    else:
        print("⚠️  No data analysis found. Run './bootstrap.sh --analyze' to generate analysis.")
    
    # --- Enhanced Model Initialization ---
    print("🧠 Initializing optimized model...")
    use_gradient_checkpointing = getattr(config.system, 'gradient_checkpointing', False)
    
    model = PanGuDrugModel(
        num_node_features=config.model.num_node_features,
        hidden_dim=config.model.hidden_dim,
        num_encoder_layers=config.model.num_encoder_layers,
        num_encoder_heads=config.model.num_encoder_heads,
        output_dim=config.model.output_dim,
        num_decoder_heads=config.model.num_decoder_heads,
        num_decoder_layers=config.model.num_decoder_layers,
        latent_dim=config.model.latent_dim,
        num_selected_layers=config.model.num_selected_layers,
        use_gradient_checkpointing=use_gradient_checkpointing,
        max_seq_length=config.data.max_length
    ).to(device)

    # --- Advanced Optimizer Setup ---
    print("⚙️  Setting up advanced optimizer...")
    
    # Separate parameter groups for different learning rates
    encoder_params = list(model.encoder.parameters())
    decoder_params = list(model.decoder.parameters())
    latent_params = list(model.fc_mean.parameters()) + list(model.fc_log_var.parameters())
    
    param_groups = [
        {'params': encoder_params, 'lr': config.training.learning_rate, 'name': 'encoder'},
        {'params': decoder_params, 'lr': config.training.learning_rate * 1.2, 'name': 'decoder'},  # Slightly higher for decoder
        {'params': latent_params, 'lr': config.training.learning_rate * 0.8, 'name': 'latent'}  # Lower for latent space
    ]
    
    optimizer = torch.optim.AdamW(
        param_groups,
        weight_decay=config.optimization.weight_decay,
        betas=(0.9, 0.999),
        eps=1e-8
    )

    # --- Early Stopping ---
    early_stopping = EarlyStopping(
        patience=getattr(config.training, 'early_stopping_patience', 15),
        min_delta=1e-5,
        restore_best_weights=True
    )
    
    # --- Advanced Learning Rate Scheduling ---
    # Calculate total steps after we have dataset information
    total_steps_per_epoch = 1000  # Default fallback
    val_steps_per_epoch = 100     # Default fallback
    total_steps = max(1, total_steps_per_epoch * config.training.num_epochs)
    warmup_steps = getattr(config.optimization, 'warmup_steps', min(1000, total_steps // 10))
    
    scheduler = AdvancedLRScheduler(
        optimizer,
        warmup_steps=warmup_steps,
        total_steps=total_steps,
        scheduler_type=getattr(config.optimization, 'scheduler', 'cosine'),
        min_lr_ratio=0.01
    )

    # --- Mixed Precision Training ---
    use_mixed_precision = getattr(config.system, 'mixed_precision', True)
    scaler = GradScaler() if use_mixed_precision and device.type == 'cuda' else None

    # --- Gradient Accumulation ---
    gradient_accumulation_steps = getattr(config.training, 'gradient_accumulation_steps', 1)
    effective_batch_size = config.data.batch_size * gradient_accumulation_steps
    print(f"Effective batch size: {effective_batch_size} (batch_size={config.data.batch_size}, accumulation_steps={gradient_accumulation_steps})")

    # --- Load Checkpoint ---
    start_epoch, last_loss, _ = load_checkpoint(model, optimizer, scheduler, config.paths.checkpoint_path)

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
    train_dataset_info = ""
    val_dataset_info = ""

    
    try:
        if use_streaming:
            train_dataset_info = f"Training: Streaming mode - processing {len(train_dataset.processed_files)} files"
            val_dataset_info = f"Validation: Streaming mode - processing {len(val_dataset.processed_files)} files"
            
            # Use estimated values for streaming mode
            total_steps_per_epoch = max(1, 670071 // config.data.batch_size // gradient_accumulation_steps)
            val_steps_per_epoch = max(1, 83759 // config.data.batch_size)
        else:
            train_samples = len(train_dataset)
            val_samples = len(val_dataset)
            train_dataset_info = f"Training: {train_samples:,} samples from {len(train_dataset.processed_files)} files"
            val_dataset_info = f"Validation: {val_samples:,} samples from {len(val_dataset.processed_files)} files"
            total_steps_per_epoch = max(1, len(train_dataloader) // gradient_accumulation_steps)
            val_steps_per_epoch = max(1, len(val_dataloader))
    except Exception as e:
        print(f"Warning: Could not calculate exact steps, using defaults: {e}")
        total_steps_per_epoch = 1000
        val_steps_per_epoch = 100
    
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
       Training Path: {config.data.train_dataset_path}
       Validation Path: {config.data.val_dataset_path}
       {train_dataset_info}
       {val_dataset_info}
       Max Sequence Length: {config.data.max_length}
       
    ⚙️  BATCH CONFIGURATION:
       Actual Batch Size: {actual_batch_size}
       Gradient Accumulation Steps: {gradient_accumulation_steps}
       Effective Batch Size: {effective_batch_size}
       Training Steps per Epoch: {total_steps_per_epoch:,}
       Validation Steps per Epoch: {val_steps_per_epoch:,}
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
        
        # Initialize counters for this epoch
        step_count = 0
        batch_count = 0
        
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{config.training.num_epochs}")
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
                with autocast('cuda'):
                    condition_vector = create_condition_vector(
                        batch_size, 
                        config.model.latent_dim, 
                        device
                    )
                    output, mean, log_var = model(data, condition_vector, tgt_one_hot)
                    
                    # Prepare targets for loss calculation
                    targets = target_tensors[:, 1:]  # Shifted targets
                    loss_dict = vae_loss(
                        output,
                        targets,
                        mean,
                        log_var,
                        beta=config.training.beta
                    )
                    loss = loss_dict['total_loss'] / gradient_accumulation_steps
                
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
                loss_dict = vae_loss(
                    output,
                    targets,
                    mean,
                    log_var,
                    beta=config.training.beta
                )
                loss = loss_dict['total_loss'] / gradient_accumulation_steps
                
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
        
        # --- Validation ---
        model.eval()
        val_loss = 0.0
        val_step_count = 0
        
        print(f"Running validation...")
        with torch.no_grad():
            val_progress_bar = tqdm(val_dataloader, desc="Validation")
            for data in val_progress_bar:
                if data is None:
                    continue
                    
                data = data.to(device)
                batch_size = data.num_graphs if hasattr(data, 'num_graphs') else len(data)
                
                # Get SMILES strings from batch and convert to SELFIES
                if hasattr(data, 'smiles'):
                    smiles_strings = data.smiles
                else:
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
                tgt_one_hot = tgt_one_hot.transpose(1, 2)
                
                # Forward pass
                condition_vector = create_condition_vector(
                    batch_size,
                    config.model.latent_dim,
                    device
                )
                output, mean, log_var = model(data, condition_vector, tgt_one_hot)
                
                # Prepare targets for loss calculation
                targets = target_tensors[:, 1:]
                loss_dict = vae_loss(
                    output,
                    targets,
                    mean,
                    log_var,
                    beta=config.training.beta
                )
                val_loss += loss_dict['total_loss'].item()
                val_step_count += 1
                
                val_progress_bar.set_postfix(val_loss=loss_dict['total_loss'].item())
        
        avg_val_loss = val_loss / max(val_step_count, 1)
        print(f"Validation Loss: {avg_val_loss:.4f}")
        
        # Log validation loss
        writer.add_scalar('Loss/val_epoch_avg', avg_val_loss, epoch)
        
        # Early stopping check
        if early_stopping(avg_val_loss, model):
            print(f"Early stopping triggered at epoch {epoch+1}")
            break
        
        # --- Save Checkpoint ---
        save_checkpoint(
            model,
            optimizer,
            scheduler,
            epoch + 1,
            avg_val_loss,  # Use validation loss for checkpointing
            metrics,
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