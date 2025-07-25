import torch
from torch_geometric.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import os

from .data_loader import ZINC_Dataset
from .model import PanGuDrugModel, vae_loss

# --- Configuration ---
DATASET_PATH = "data/processed"
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
NUM_EPOCHS = 10
LOG_DIR = "runs/pangu_drug_model"
CHECKPOINT_PATH = "checkpoints/pangu_drug_model.pt"

# Model Parameters (placeholders, need to be set based on the data)
# These should be refined based on the actual data and paper
NUM_NODE_FEATURES = 6  # Based on the features extracted in data_loader.py
HIDDEN_DIM = 256
NUM_ENCODER_LAYERS = 6
NUM_ENCODER_HEADS = 4
OUTPUT_DIM = 128  # Placeholder for vocab size of SELFIES
NUM_DECODER_HEADS = 4
NUM_DECODER_LAYERS = 6
LATENT_DIM = 128

def save_checkpoint(model, optimizer, epoch, loss):
    os.makedirs(os.path.dirname(CHECKPOINT_PATH), exist_ok=True)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, CHECKPOINT_PATH)

def load_checkpoint(model, optimizer):
    if os.path.exists(CHECKPOINT_PATH):
        print(f"Loading checkpoint from {CHECKPOINT_PATH}")
        checkpoint = torch.load(CHECKPOINT_PATH)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        print(f"Resuming from epoch {epoch} with loss {loss}")
        return epoch, loss
    else:
        print("No checkpoint found, starting from scratch.")
        return 0, 0.0

def train():
    # --- Device ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- TensorBoard ---
    writer = SummaryWriter(LOG_DIR)

    # --- Dataset and DataLoader ---
    dataset = ZINC_Dataset(DATASET_PATH)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # --- Model ---
    num_selected_layers = NUM_ENCODER_LAYERS if NUM_ENCODER_LAYERS < 10 else 8
    model = PanGuDrugModel(
        num_node_features=NUM_NODE_FEATURES,
        hidden_dim=HIDDEN_DIM,
        num_encoder_layers=NUM_ENCODER_LAYERS,
        num_encoder_heads=NUM_ENCODER_HEADS,
        output_dim=OUTPUT_DIM,
        num_decoder_heads=NUM_DECODER_HEADS,
        num_decoder_layers=NUM_DECODER_LAYERS,
        latent_dim=LATENT_DIM,
        num_selected_layers=num_selected_layers
    ).to(device)

    # --- Optimizer ---
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # --- Load Checkpoint ---
    start_epoch, last_loss = load_checkpoint(model, optimizer)

    # --- Training Loop ---
    print("Starting training...")
    for epoch in range(start_epoch, NUM_EPOCHS):
        model.train()
        total_loss = 0
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")
        for i, data in enumerate(progress_bar):
            data = data.to(device)

            # --- Forward Pass ---
            optimizer.zero_grad()
            
            # Placeholder for target and condition vectors
            condition_vector = torch.randn(data.num_graphs, LATENT_DIM).to(device)
            tgt = torch.randn(data.num_graphs, 10, OUTPUT_DIM).to(device)

            output, mean, log_var = model(data, condition_vector, tgt)
            
            loss = vae_loss(output, tgt, mean, log_var)
            
            # --- Backward Pass ---
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())

            # --- TensorBoard Logging (per batch) ---
            global_step = epoch * len(dataloader) + i
            writer.add_scalar('Loss/train_batch', loss.item(), global_step)

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Average Loss: {avg_loss:.4f}")
        
        # --- TensorBoard Logging (per epoch) ---
        writer.add_scalar('Loss/train_epoch_avg', avg_loss, epoch)
        
        # --- Save Checkpoint ---
        save_checkpoint(model, optimizer, epoch + 1, avg_loss)

    writer.close()
    print("Training complete.")

if __name__ == "__main__":
    train()
