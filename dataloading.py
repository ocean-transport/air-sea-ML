import torch
from torch.utils.data import TensorDataset, DataLoader
import numpy as np


def prepare_dataloaders(ds_HR_reg, ds_LR_reg, var='u_relative', batch_size=32, normalization_type=None, normalization_stats=True):
    """
    Prepare train/val/test DataLoaders from high- and low-resolution datasets.

    Args:
        ds_HR_reg: xarray.Dataset - high-res dataset
        ds_LR_reg: xarray.Dataset - low-res dataset
        var: str - variable to extract (e.g., 'u_relative', 'v_relative')
        batch_size: int - batch size for DataLoaders
        normalization_type: str or None - 'standardization', 'normalization', or None
        normalization_stats: bool - if True, return normalization statistics

    Returns:
        train_loader, val_loader, test_loader
        HR_test (xarray.DataArray), LR_test (xarray.DataArray)
    """

    # Split data into train/val/test
    # Total number of samples
    n_samples = len(ds_HR_reg.time)  # Assuming 'time' is the dimension

    # Generate random indices
    indices = np.arange(n_samples)
    np.random.shuffle(indices)

    # Split indices for training, validation, and testing
    train_size = int(0.7 * n_samples)
    val_size = int(0.15 * n_samples)

    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]

    # Select data using the indices
    # HR
    train_data_HR = ds_HR_reg.isel(time=train_indices)
    val_data_HR = ds_HR_reg.isel(time=val_indices)
    test_data_HR = ds_HR_reg.isel(time=test_indices)
    # LR
    train_data_LR = ds_LR_reg.isel(time=train_indices)
    val_data_LR = ds_LR_reg.isel(time=val_indices)
    test_data_LR = ds_LR_reg.isel(time=test_indices)

    print("Training set size:", len(train_data_HR.time))
    print("Validation set size:", len(val_data_HR.time))
    print("Testing set size:", len(test_data_HR.time))

    # Select data for specific variable
    # Extract HR data
    HR_train = getattr(train_data_HR, var).values # training
    HR_val   = getattr(val_data_HR, var).values   # validation
    HR_test  = getattr(test_data_HR, var)         # testing

    # Extract LR data
    LR_train = getattr(train_data_LR, var).values # training
    LR_val   = getattr(val_data_LR, var).values   # validation
    LR_test  = getattr(test_data_LR, var).values  # testing

    # Always define norm_stats
    norm_stats = None

    # Apply optional normalization using stats from training data
    if normalization_type == 'standardization':
        # Step 1: Get training stats
        mean = HR_train.mean()
        std = HR_train.std()

        # Step 2: Normalize w.r.t. HR_train stats
        HR_train = (HR_train - mean) / std
        HR_val   = (HR_val   - mean) / std
        HR_test  = (HR_test - mean) / std  

        LR_train = (LR_train - mean) / std
        LR_val   = (LR_val   - mean) / std
        LR_test  = (LR_test - mean) / std

        if normalization_stats:
            norm_stats = (mean, std)

    elif normalization_type == 'normalization':
        # Step 1: Get training stats        
        min_val = HR_train.min()
        max_val = HR_train.max()

        # Step 2: Normalize w.r.t. HR_train stats
        HR_train = (HR_train - min_val) / (max_val - min_val)
        HR_val   = (HR_val - min_val) / (max_val - min_val)
        HR_test  = (HR_test - min_val) / (max_val - min_val)

        LR_train = (LR_train - min_val) / (max_val - min_val)
        LR_val   = (LR_val - min_val) / (max_val - min_val)
        LR_test  = (LR_test - min_val) / (max_val - min_val)

        if normalization_stats:
            norm_stats = (min_val, max_val)

    elif normalization_type is not None:
        raise ValueError("normalization_type must be 'standardization', 'normalization', or None.")

    # Convert train/val/test data to Torch Tensor
    x_train = torch.tensor(LR_train, dtype=torch.float32).unsqueeze(1)  # (N, 1, H, W)
    y_train = torch.tensor(HR_train, dtype=torch.float32).unsqueeze(1)  # (N, 1, H, W)

    x_val = torch.tensor(LR_val, dtype=torch.float32).unsqueeze(1)  # (N, 1, H, W)
    y_val = torch.tensor(HR_val, dtype=torch.float32).unsqueeze(1)  # (N, 1, H, W)

    x_test = torch.tensor(LR_test, dtype=torch.float32).unsqueeze(1)  # (N, 1, H, W)
    # y_test = torch.tensor(HR_test, dtype=torch.float32).unsqueeze(1)  # (N, 1, H, W)

    # Create train/val/test DataLoaders
    train_dataset = TensorDataset(x_train, y_train)
    val_dataset = TensorDataset(x_val, y_val)
    test_dataset = TensorDataset(x_test, torch.zeros(len(x_test)))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    if normalization_stats:
        return train_loader, val_loader, test_loader, HR_test, LR_test, norm_stats
    else:
        return train_loader, val_loader, test_loader, HR_test, LR_test