"""
Dissipative Hamiltonian Neural Network (DHNN) for Time Series Forecasting
Implements a physics-inspired neural network with energy conservation and dissipation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict, Optional, List
import math
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from typing import Any, Union

# Type aliases
Tensor = torch.Tensor
Module = nn.Module

class HamiltonianFunction(nn.Module):
    """Learnable Hamiltonian function H(q,p) = K(p) + V(q)"""
    def __init__(self, input_dim: int, hidden_dims: List[int] = None):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [64, 64]
            
        # Build the potential energy network V(q)
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.Softplus()  # Ensures positive-definite energy
            ])
            prev_dim = h_dim
        
        # Output is a single scalar (total energy)
        layers.append(nn.Linear(prev_dim, 1))
        self.net = nn.Sequential(*layers)
        
    def forward(self, q: Tensor, p: Tensor) -> Tensor:
        # Concatenate position (q) and momentum (p) terms
        x = torch.cat([q, p], dim=-1)
        return self.net(x).squeeze(-1)

class DissipativeHamiltonianNN(pl.LightningModule):
    """
    Dissipative Hamiltonian Neural Network for time series forecasting.
    Implements the dynamics:
    dq/dt = ∂H/∂p - γq
    dp/dt = -∂H/∂q - γp + F_ext
    """
    def __init__(
        self,
        input_dim: int = 1,
        hidden_dims: List[int] = None,
        gamma: float = 0.1,  # Damping coefficient
        dt: float = 0.1,     # Time step
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # Learnable Hamiltonian function
        self.H = HamiltonianFunction(input_dim * 2, hidden_dims)
        
        # Damping coefficient (can be made learnable)
        self.gamma = nn.Parameter(torch.tensor(gamma), requires_grad=True)
        
        # Time step for numerical integration
        self.dt = dt
        
        # Optimization parameters
        self.lr = lr
        self.weight_decay = weight_decay
        
    def forward(self, x: Tensor, n_steps: int = 1) -> Tuple[Tensor, Tensor]:
        """
        Forward pass through the DHNN
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
            n_steps: Number of steps to forecast
            
        Returns:
            Tuple of (q, p) where q are the predicted positions (forecasts)
            and p are the predicted momenta
        """
        batch_size, seq_len, _ = x.shape
        device = x.device
        
        # Initialize q (position) and p (momentum)
        q = x.clone()
        p = torch.zeros_like(x)
        
        # Store predictions
        q_pred = []
        
        # Initial state
        q_pred.append(q[:, -1:, :])  # Last observed point
        
        # Autoregressive forecasting
        for _ in range(n_steps):
            # Compute gradients of H w.r.t. q and p
            q.requires_grad_(True)
            p.requires_grad_(True)
            
            # Compute Hamiltonian and its gradients
            H = self.H(q, p)
            dH_dq = torch.autograd.grad(H.sum(), q, create_graph=True)[0]
            dH_dp = torch.autograd.grad(H.sum(), p, create_graph=True)[0]
            
            # Update equations with dissipation
            dq_dt = dH_dp - self.gamma * q
            dp_dt = -dH_dq - self.gamma * p
            
            # Update q and p using Euler's method
            q_new = q + self.dt * dq_dt
            p_new = p + self.dt * dp_dt
            
            # Store prediction
            q_pred.append(q_new[:, -1:, :])
            
            # Update for next step
            q = torch.cat([q[:, 1:, :], q_new[:, -1:, :]], dim=1)
            p = torch.cat([p[:, 1:, :], p_new[:, -1:, :]], dim=1)
        
        # Stack predictions (batch_size, n_steps + 1, input_dim)
        return torch.cat(q_pred, dim=1)
    
    def training_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        x, y = batch
        y_hat = self(x, n_steps=y.size(1))
        
        # Compute loss (MSE + energy conservation + regularization)
        mse_loss = F.mse_loss(y_hat, y)
        
        # Add energy conservation loss
        q = torch.cat([x, y_hat], dim=1)
        p = torch.zeros_like(q, requires_grad=True)
        H = self.H(q, p)
        dH_dq = torch.autograd.grad(H.sum(), q, create_graph=True)[0]
        dH_dp = torch.autograd.grad(H.sum(), p, create_graph=True)[0]
        
        # Hamiltonian equations of motion
        dq_dt = dH_dp - self.gamma * q
        dp_dt = -dH_dq - self.gamma * p
        
        # Compute violation of Hamilton's equations
        ham_loss = (dq_dt.pow(2).mean() + dp_dt.pow(2).mean()) * 0.1
        
        # Total loss
        loss = mse_loss + ham_loss
        
        # Log metrics
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_mse', mse_loss, prog_bar=True)
        self.log('hamiltonian_loss', ham_loss, prog_bar=True)
        self.log('gamma', self.gamma, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> None:
        x, y = batch
        y_hat = self(x, n_steps=y.size(1))
        val_loss = F.mse_loss(y_hat, y)
        self.log('val_loss', val_loss, prog_bar=True)
        return val_loss
    
    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )
    
    def predict(
        self, 
        x: Union[np.ndarray, Tensor], 
        n_steps: int = 1,
        batch_size: int = 32,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ) -> np.ndarray:
        """Generate predictions for input sequence"""
        self.eval()
        if isinstance(x, np.ndarray):
            x = torch.FloatTensor(x).to(device)
        
        # Handle batch prediction
        if len(x.shape) == 2:  # Single sequence
            x = x.unsqueeze(0)
        
        with torch.no_grad():
            preds = self(x.to(device), n_steps=n_steps)
        
        return preds.cpu().numpy()


class TimeSeriesDataset(Dataset):
    """PyTorch Dataset for time series forecasting"""
    def __init__(
        self, 
        data: np.ndarray, 
        seq_len: int = 30, 
        pred_len: int = 1,
        stride: int = 1
    ):
        """
        Args:
            data: Time series data of shape (n_samples, n_features)
            seq_len: Length of input sequences
            pred_len: Length of prediction horizon
            stride: Stride for sliding window
        """
        self.data = data
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.stride = stride
        
    def __len__(self) -> int:
        return (len(self.data) - self.seq_len - self.pred_len) // self.stride + 1
    
    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        start = idx * self.stride
        end = start + self.seq_len
        
        x = self.data[start:end]
        y = self.data[end:end + self.pred_len]
        
        return torch.FloatTensor(x), torch.FloatTensor(y)


def train_dhnn(
    train_data: np.ndarray,
    val_data: Optional[np.ndarray] = None,
    seq_len: int = 30,
    pred_len: int = 1,
    batch_size: int = 32,
    hidden_dims: List[int] = None,
    max_epochs: int = 100,
    patience: int = 10,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    gpus: int = 1 if torch.cuda.is_available() else 0,
    **kwargs
) -> DissipativeHamiltonianNN:
    """
    Train a DHNN model on time series data
    
    Args:
        train_data: Training data of shape (n_samples, n_features)
        val_data: Optional validation data (same shape as train_data)
        seq_len: Length of input sequences
        pred_len: Length of prediction horizon
        batch_size: Batch size for training
        hidden_dims: List of hidden layer dimensions for the Hamiltonian network
        max_epochs: Maximum number of training epochs
        patience: Number of epochs to wait before early stopping
        lr: Learning rate
        weight_decay: Weight decay for L2 regularization
        gpus: Number of GPUs to use for training
        
    Returns:
        Trained DHNN model
    """
    # Create datasets
    train_dataset = TimeSeriesDataset(train_data, seq_len, pred_len)
    val_dataset = TimeSeriesDataset(val_data, seq_len, pred_len) if val_data is not None else None
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size * 2,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
    else:
        val_loader = None
    
    # Initialize model
    input_dim = train_data.shape[-1]
    model = DissipativeHamiltonianNN(
        input_dim=input_dim,
        hidden_dims=hidden_dims or [64, 64],
        lr=lr,
        weight_decay=weight_decay,
        **kwargs
    )
    
    # Set up callbacks
    callbacks = [
        EarlyStopping(
            monitor='val_loss' if val_loader else 'train_loss',
            patience=patience,
            verbose=True,
            mode='min'
        ),
        ModelCheckpoint(
            monitor='val_loss' if val_loader else 'train_loss',
            save_top_k=1,
            mode='min',
            save_last=True
        )
    ]
    
    # Train the model
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        callbacks=callbacks,
        gpus=gpus,
        progress_bar_refresh_rate=20,
        logger=True,
        log_every_n_steps=10
    )
    
    trainer.fit(model, train_loader, val_loader if val_loader else None)
    
    return model
