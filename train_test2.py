import xarray as xr
import numpy as np
import torch
import torch.nn as nn
import os
import time
from utils import *
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from collections import deque
from sklearn.metrics import r2_score


############################################################   TRAINING   ####################################################################

def train_model(model, train_loader, val_loader,
                criterion, optimizer, device,
                save_path ='/home/jovyan/SSH/B_data/updated_dm/test3/model.pth', n_epochs=2000, patience=50, stop_crit='R2', lr_sched_crit='valLoss'):

    """
    This function trains a deep learning model.
    
    Parameters:
    - model: The neural network model to be trained
    - train_loader: DataLoader for training data
    - val_loader: DataLoader for validation data
    - criterion: Loss function
    - optimizer: Optimization algorithm
    - device: Device to run the model on (CPU or GPU)
    - save_path: Path to save the trained model
    - n_epochs: Number of training epochs
    - stop_crit: Stopping criteria ('R2' or 'valLoss')
    - lr_sched_crit: Stopping criteria lr scheduler is based on ('R2' or 'valLoss')
    
    Returns:
    - train_losses (np.ndarray): Array of training loss per epoch
    - val_losses (np.ndarray): Array of validation loss per epoch
    - val_r2_history (np.ndarray): Array of r2 values per epoch
    """
    
    def print_numberofparameters(model):
        """Print the number of trainable parameters in the model"""
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Trainable Parameters: {trainable_params}")
    
    # Create a learning rate scheduler that reduces the learning rate when the validation loss stops improving 
    
    if lr_sched_crit == "valLoss":
        scheduler_mode = "min"
    elif lr_sched_crit == "R2":
        scheduler_mode = "max"

    scheduler = ReduceLROnPlateau(optimizer, mode=scheduler_mode, factor=0.85, patience=4, verbose=True)

    # Move the model to the specified device (CPU or GPU)
    model.to(device)
    print_numberofparameters(model)
    
    # Initialize variables for early stopping
    k = 5  # Number of best validation losses or R2 to keep track of
    best_val_loss = deque(maxlen=k)
    best_val_r2 = deque(maxlen=k)
    patience_counter = 0

    # === Initialize NumPy arrays for loss history ===
    train_losses = np.array([])
    val_losses = np.array([])
    val_r2_history = np.array([])
    
    # Check if a saved model exists and load it if it does
    if os.path.isfile(save_path):
        checkpoint = torch.load(save_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint['best_val_loss']
        best_val_r2 = checkpoint['best_val_r2']
        patience_counter = checkpoint['patience_counter']
        # Load previous loss arrays if present
        train_losses = checkpoint.get('train_losses', np.array([]))
        val_losses = checkpoint.get('val_losses', np.array([]))
        val_r2_history = checkpoint.get('val_r2_history', val_r2_history)
        print(f"Resuming from epoch {start_epoch} with best val losses {list(best_val_loss)} and best R2 values {list(best_val_r2)}")
    else:
        start_epoch = 0

    # Main training loop
    for epoch in range(start_epoch, n_epochs):
        
        start_time = time.time()  
        model.train()  # Set model to training mode
        train_running_loss = 0.0

        # Training phase
        for batch_x, batch_y in train_loader:
            # Move batch to device
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            # Forward pass
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_running_loss += loss.item() * batch_x.size(0)
            
        # Calculate average training loss for the epoch
        epoch_loss = train_running_loss / len(train_loader.dataset)
        
        # Validation phase
        model.eval()   # Set model to evaluation mode
        val_running_loss = 0.0
        all_outputs = []
        all_targets = []
    
        with torch.no_grad():   # Disable gradient calculation for validation
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                val_running_loss += loss.item() * batch_x.size(0)

                all_outputs.append(outputs.cpu())
                all_targets.append(batch_y.cpu())
                
        # Calculate average validation loss
        val_loss = val_running_loss / len(val_loader.dataset)

        # Calculate R2
        # === Flatten outputs for r2_score ===
        all_outputs = torch.cat(all_outputs, dim=0).view(torch.cat(all_outputs, dim=0).size(0), -1).numpy()
        all_targets = torch.cat(all_targets, dim=0).view(torch.cat(all_targets, dim=0).size(0), -1).numpy()
        val_r2 = r2_score(all_targets, all_outputs)

        # === Append to NumPy arrays ===
        train_losses = np.append(train_losses, epoch_loss)
        val_losses = np.append(val_losses, val_loss)
        val_r2_history = np.append(val_r2_history, val_r2)
        
        # Update learning rate based on validation loss
        if lr_sched_crit == "valLoss":
            scheduler_metric = val_loss
        elif lr_sched_crit == "R2":
            scheduler_metric = val_r2
        scheduler.step(scheduler_metric)

        # Calculate epoch duration and peak memory usage
        end_time = time.time()
        epoch_duration = end_time - start_time
        peak_memory = torch.cuda.max_memory_allocated(device=device) / (1024 ** 2)
        
        print(f'Epoch {epoch+1}, Train Loss: {epoch_loss:.3e}, Val Loss: {val_loss:.3e}, Val R²: {val_r2:.4f}, Epoch Time: {epoch_duration:.2f}s')
        
        # Save model for first 30 epochs regardless of performance
        if epoch < 30:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_loss': best_val_loss,
                'best_val_r2': best_val_r2,
                'patience_counter': patience_counter,
                # Save losses
                'train_losses': train_losses,
                'val_losses': val_losses,
                'val_r2_history': val_r2_history
            }
            torch.save(checkpoint, save_path)
            print(f'Model saved at epoch {epoch+1}')
        else:
            if stop_crit == 'valLoss':
                # After 30 epochs, use early stopping logic
                if len(best_val_loss) < k or val_loss < max(best_val_loss):
                    if len(best_val_loss) == k:
                        best_val_loss.remove(max(best_val_loss))
                    best_val_loss.append(val_loss)
                    patience_counter = 0
                else:
                    patience_counter += 1
                    print(f'Patience counter: {patience_counter}/{patience}')
            
                # Save checkpoint only if it's the best model so far
                if val_loss == min(best_val_loss):
                    checkpoint = {
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'best_val_loss': best_val_loss,
                        'best_val_r2': best_val_r2,
                        'patience_counter': patience_counter,
                        # Save losses
                        'train_losses': train_losses,
                        'val_losses': val_losses,
                        'val_r2_history': val_r2_history
                    }
                    torch.save(checkpoint, save_path)
                    print(f'Best model so far saved to {save_path}')
            
                # Early stopping check
                if patience_counter >= patience:
                    print('Early stopping triggered')
                    break

            if stop_crit == 'R2':
                    # After 30 epochs, use early stopping logic
                if len(best_val_r2) < k or val_r2 > min(best_val_r2):
                    if len(best_val_r2) == k:
                        best_val_r2.remove(min(best_val_r2))
                    best_val_r2.append(val_r2)
                    patience_counter = 0
                else:
                    patience_counter += 1
                    print(f'Patience counter: {patience_counter}/{patience}')
            
                # Save checkpoint only if it's the best model so far
                if val_r2 == max(best_val_r2):
                    checkpoint = {
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'best_val_loss': best_val_loss,
                        'best_val_r2': best_val_r2,
                        'patience_counter': patience_counter,
                        # Save losses
                        'train_losses': train_losses,
                        'val_losses': val_losses,
                        'val_r2_history': val_r2_history
                    }
                    torch.save(checkpoint, save_path)
                    print(f'Best model so far saved to {save_path}')
            
                # Early stopping check
                if patience_counter >= patience:
                    print('Early stopping triggered')
                    break

    print('Training complete')
    return train_losses, val_losses, val_r2_history
