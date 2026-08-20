# Import standard libraries
import os
import numpy as np
import pandas as pd

from torchdiffeq import odeint
from src.sample.utils.plotting_utils import *

# import machine learning modules
from src.sample.decorators.general_decorators import *
from sample.utils.saving_and_loading_utils import *
from src.sample.config import *
from src.sample.utils.plotting_utils import plot_signals
import torch
from src.sample.classes.controllers.MambaInverseController import MambaInverseController
import matplotlib.pyplot as plt
import random

plt.style.use("src/sample/style.mplstyle")

import pickle


def create_inverse_controller_dataset(Y_trajectories, U_trajectories, n_y, n_u):
    """
    Slices raw batch continuous MIMO trajectories into history-windowed features 
    and targets for an inverse controller.
    
    Parameters:
        Y_trajectories: Tensor or NumPy array of shape [Num_Traces, Seq_Len, input_dim] (Plant Outputs)
        U_trajectories: Tensor or NumPy array of shape [Num_Traces, Seq_Len, output_dim] (Control Inputs)
        n_y: Number of past plant output lookbacks (excluding current y_k)
        n_u: Number of past control action lookbacks
        
    Returns:
        X_raw: NumPy array of shape [Num_Traces, Sliding_Seq_Len, Feature_Dim]
        Y_raw: NumPy array of shape [Num_Traces, Sliding_Seq_Len, output_dim]
    """
    # Convert PyTorch tensors to NumPy arrays if necessary
    if torch.is_tensor(Y_trajectories):
        Y_trajectories = Y_trajectories.detach().cpu().numpy()
    if torch.is_tensor(U_trajectories):
        U_trajectories = U_trajectories.detach().cpu().numpy()
        
    # --- FIXED DIMENSION UNPACKING HERE ---
    num_traces, total_seq_len, output_dim = Y_trajectories.shape
    input_dim = U_trajectories.shape[-1]
    
    start_idx = max(n_y, n_u)
    end_idx = total_seq_len - 1
    sliding_seq_len = end_idx - start_idx
    
    # Calculate total feature dimension for verification
    # y_{k+1} (input_dim) + y_k...y_{k-n_y} (input_dim * (n_y + 1)) + u_{k-1}...u_{k-n_u} (output_dim * n_u)
    feature_dim = n_u * input_dim + (n_y+2) * output_dim
    
    print(f"📦 Slicing {num_traces} traces. Window metrics:")
    print(f"   ↳ Clean Rollout Steps per Trace: {sliding_seq_len}")
    print(f"   ↳ Total Feature vector size (dim_v): {feature_dim}")

    X_list = []
    Y_list = []
    
    for t_idx in range(num_traces):
        y_trace = Y_trajectories[t_idx]  # Shape: [Total_Seq_Len, input_dim]
        #print("y_trace.shape", y_trace.shape)
        u_trace = U_trajectories[t_idx]  # Shape: [Total_Seq_Len, output_dim]
        #print("u_trace.shape", u_trace.shape)
        trace_features = []
        trace_targets = []
        
        for k in range(start_idx, end_idx):
            # 1. Future target trajectory point: y_{k+1}
            y_next = y_trace[k + 1]
            #print("y_next: ", y_next.shape)
            # 2. Plant output history: [y_k, y_{k-1}, ..., y_{k-n_y}]
            y_hist = y_trace[k - n_y : k + 1].flatten()
            #print("y_hist: ", y_hist.shape) 
            #y_hist_reversed = y_hist[::-1].flatten()
            
            # 3. Control input history: [u_{k-1}, u_{k-2}, ..., u_{k-n_u}]
            u_hist = u_trace[k - n_u : k].flatten()
            #print("u_hist: ", u_hist.shape) 
            #u_hist_reversed = u_hist[::-1].flatten()
            
            # Combine into a single feature row v_k
            v_k = np.concatenate([y_next, y_hist, u_hist])

            #print("v_k: ", v_k.shape)
            
            trace_features.append(v_k)
            trace_targets.append(u_trace[k])  # Target is the control action u_k
            
        X_list.append(np.array(trace_features))  # Shape: [Sliding_Seq_Len, feature_dim]
        Y_list.append(np.array(trace_targets))   # Shape: [Sliding_Seq_Len, output_dim]
        
    # Stack back to 3D arrays matching your train_controller layout expectations
    X_raw = np.stack(X_list, axis=0)  # [Num_Traces, Sliding_Seq_Len, feature_dim]
    Y_raw = np.stack(Y_list, axis=0)  # [Num_Traces, Sliding_Seq_Len, output_dim]

    print("X_raw shape after slicing:", X_raw.shape)
    
    return X_raw, Y_raw





def compute_trajectory_metrics(y_true, y_pred, dt, eps=1e-8):
    """
    Computes trajectory error metrics averaged across time steps and features.
    
    Args:
        y_true: np.ndarray of shape (N_samples, seq_len, dim) or (seq_len, dim)
        y_pred: np.ndarray of shape (N_samples, seq_len, dim) or (seq_len, dim)
        dt: float, time step interval
        eps: small constant to avoid division by zero
        
    Returns:
        dict containing MSE, RMSE, MAE, MAPE, NRMSE, IAE, and ISE
    """
    error = y_true - y_pred
    abs_error = np.abs(error)
    sq_error = error ** 2

    # Pointwise trajectory metrics (full float precision)
    mse = float(np.mean(sq_error))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(abs_error))
    mape = float(np.mean(abs_error / (np.abs(y_true) + eps)) * 100.0)
    
    # Normalized RMSE by standard deviation of ground truth
    std_true = float(np.std(y_true))
    nrmse = float(rmse / (std_true + eps))
    
    # Integral metrics per trajectory sequence, averaged across trajectories and features
    iae = float(np.mean(np.sum(abs_error, axis=-2) * dt))
    ise = float(np.mean(np.sum(sq_error, axis=-2) * dt))

    return {
        "MSE": mse,
        "RMSE": rmse,
        "MAE": mae,
        "MAPE": mape,
        "NRMSE": nrmse,
        "IAE": iae,
        "ISE": ise
    }

    
#=== FUNCTION TO COMPUTE AND SAVE TRACKING METRICS ===#
def compute_and_save_tracking_metrics(
    y_np,        # Actual output [steps, batch_size]
    ref_np,      # Reference trajectory [steps, batch_size]
    dt,
    dirname,
    settle_tol=0.05, # Band for "tracking error"
    suffix=None
):
    """
    Pure tracking metrics for curve comparison including MAPE.
    Calculates how well the plant output follows a dynamic reference.
    """
    steps, batch_size = y_np.shape
    
    # Ensure ref_np is the same shape as y_np
    if np.isscalar(ref_np):
        ref_np = np.full_like(y_np, ref_np)
    elif ref_np.ndim == 1:
        ref_np = np.tile(ref_np[:, np.newaxis], (1, batch_size))

    error = y_np - ref_np
    abs_error = np.abs(error)

    # Small epsilon to prevent division by zero for near-zero references
    denom = np.abs(ref_np) + 1e-8

    # --- Integral & Percentage Metrics ---
    mae = abs_error.mean(axis=0)
    mape = (abs_error / denom).mean(axis=0) * 100.0  # MAPE in %
    mse = (error ** 2).mean(axis=0)
    rmse = np.sqrt(mse)
    iae = abs_error.sum(axis=0) * dt
    ise = (error ** 2).sum(axis=0) * dt

    # --- Dynamic Tracking Metrics ---
    max_error = np.max(abs_error, axis=0)
    
    # Time spent within tolerance band (%)
    within_band = abs_error <= (settle_tol * denom)
    time_in_band_pct = (np.sum(within_band, axis=0) / steps) * 100.0

    # --- Assemble DataFrame ---
    df = pd.DataFrame({
        "trajectory": np.arange(batch_size),
        "MAE": mae,
        "MAPE_%": mape,
        "MSE": mse,
        "RMSE": rmse,
        "IAE": iae,
        "ISE": ise,
        "Max_Error": max_error,
        "TimeInBand_%": time_in_band_pct
    })

    # Summary Statistics
    summary_df = df.describe().loc[['mean', 'std', 'min', 'max']].T.reset_index()
    summary_df.columns = ['metric', 'mean', 'std', 'min', 'max']

    # --- Save ---
    file_ext = f"_{suffix}" if suffix else ""
    save_df_to_csv(df, dirname, f"tracking_metrics_per_trajectory{file_ext}")
    save_df_to_csv(summary_df, dirname, f"tracking_metrics_summary{file_ext}")

    return df
















#=== FUNCTION TO SEED EVERYTHING FOR REPRODUCIBILITY ===#
def seed_everything(seed=42):
    """
    Seeds all relevant libraries to ensure reproducible results.

    Parameters:
    - seed (int): The numerical seed value used to initialize all random number generators (default: 42).

    Returns:
    - None: The function configures global runtime states and does not return a value.

    The function enforces absolute determinism across various execution contexts. It explicitly binds 
    the seed to Python's core `random` module, environment variables, NumPy's matrix operations, and 
    both CPU and GPU tensor variants in PyTorch. Finally, it overrides standard CUDA Deep Neural Network 
    (cuDNN) configurations to deactivate dynamic kernel auto-tuning algorithm selection, completely 
    eliminating stochastic variance across identical processing runs.
    """
    # Seed native Python behaviors and system environments
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # Seed third-party matrix execution engines
    np.random.seed(seed)
    
    # Seed deep learning core frameworks across standard processing hardware
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # Force synchronization across multi-GPU environments
    
    # Critical for CUDA reproducibility: override cuDNN runtime optimizations
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    print(f"Random seed set to: {seed}")


def compute_metrics(y_true, y_pred, eps=1e-8):
    """Computes MSE, RMSE, NRMSE (range-normalized), and MAPE.

    Parameters:
    -----------
    y_true : np.ndarray
        Ground truth sequence of shape (T,) or (T, dim)
    y_pred : np.ndarray
        Simulated/predicted sequence of shape (T,) or (T, dim)

    Returns:
    --------
    dict with keys: 'MSE', 'RMSE', 'NRMSE', 'MAPE'
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # 1. Mean Squared Error
    mse = np.mean((y_true - y_pred) ** 2)

    # 2. Root Mean Squared Error
    rmse = np.sqrt(mse)

    # 3. Normalized RMSE (Range-normalized: RMSE / (max - min))
    val_range = np.max(y_true) - np.min(y_true)
    nrmse = rmse / (val_range + eps)

    # 4. Mean Absolute Percentage Error (percentage scale)
    mape = np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + eps))) * 100.0

    return {
        "MSE": float(mse),
        "RMSE": float(rmse),
        "NRMSE": float(nrmse),
        "MAPE": float(mape),
    }

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

def extract_ssm_matrices_at_step(model, mamba_block, y_t_tensor, y_next_tensor):
    """
    Extracts the analytical SSM matrices (A, B, C, D) and discretized matrices (A_bar, B_bar)
    at the CURRENT (latest) step of the simulation sequence.
    
    Args:
        model: Your outer wrapper model class instance.
        mamba_block: The raw Mamba module instance (model.core).
        y_t_tensor: Tensor of shape [Batch, Seq_Len, input_dim]
        y_next_tensor: Tensor of shape [Batch, Seq_Len, input_dim]
    """
    # 1. Replicate how your outer wrapper converts raw plant signals to hidden_states.
    # If your wrapper has a custom forward pass, we mirror its embedding step here:
    
    # Example A: If your model concatenates or adds inputs and pushes them through a linear layer:
    if hasattr(model, 'input_projection'): # Change 'input_projection' to your wrapper's actual embedding layer name
        # If your model processes y_t and y_next together
        hidden_states = torch.cat([y_t_tensor, y_next_tensor], dim=-1)
        hidden_states = model.input_projection(hidden_states)
    else:
        # Fallback/Default: Combine them and check if we need to manually map them to d_model
        hidden_states = y_t_tensor + y_next_tensor 
        
        # If they are still raw plant dimensions (e.g., feature dim = 1 instead of 64)
        if hidden_states.shape[-1] != mamba_block.d_model:
            # Look for an embedding layer in your outer model dynamically
            embedding_layer = None
            for module in model.modules():
                if isinstance(module, nn.Linear) and module.out_features == mamba_block.d_model:
                    embedding_layer = module
                    break
            
            if embedding_layer is not None:
                # If your embedding layer expects concatenated inputs:
                if embedding_layer.in_features == (y_t_tensor.shape[-1] + y_next_tensor.shape[-1]):
                    hidden_states = torch.cat([y_t_tensor, y_next_tensor], dim=-1)
                hidden_states = embedding_layer(hidden_states)
            else:
                raise AttributeError(
                    f"Could not automatically find the embedding layer projecting raw features "
                    f"to d_model ({mamba_block.d_model}). Please pass hidden_states after your "
                    f"outer model's embedding layer."
                )

    batch, seqlen, dim = hidden_states.shape

    # 2. Project using F.linear (Now safely guaranteed to be Batch x SeqLen x 64)
    xz = F.linear(hidden_states, mamba_block.in_proj.weight, mamba_block.in_proj.bias)
    xz = rearrange(xz, "b l d -> b d l")

    x, z = xz.chunk(2, dim=1)

    # 3. Compute short convolution
    try:
        from causal_conv1d import causal_conv1d_fn
    except ImportError:
        causal_conv1d_fn = None

    if causal_conv1d_fn is None:
        x_conv = mamba_block.act(mamba_block.conv1d(x)[..., :seqlen])
    else:
        x_conv = causal_conv1d_fn(
            x=x,
            weight=rearrange(mamba_block.conv1d.weight, "d 1 w -> d w"),
            bias=mamba_block.conv1d.bias,
            activation=mamba_block.activation,
        )

    # 4. Pull dynamic projections for the entire sequence
    x_dbl = mamba_block.x_proj(rearrange(x_conv, "b d l -> (b l) d"))
    dt_proj, B_seq, C_seq = torch.split(
        x_dbl, [mamba_block.dt_rank, mamba_block.d_state, mamba_block.d_state], dim=-1
    )

    dt_seq = F.linear(dt_proj, mamba_block.dt_proj.weight)
    dt_seq = rearrange(dt_seq, "(b l) d -> b d l", l=seqlen)
    
    B_seq = rearrange(B_seq, "(b l) dstate -> b dstate l", l=seqlen)
    C_seq = rearrange(C_seq, "(b l) dstate -> b dstate l", l=seqlen)

    # 5. ISOLATE THE CURRENT SIMULATION STEP
    dt_current = dt_seq[..., -1]         
    B_current = B_seq[..., -1]           
    C_current = C_seq[..., -1]           

    # 6. Extract static variables
    A_static = -torch.exp(mamba_block.A_log.float())  
    D_static = mamba_block.D.float()                  

    # 7. Discretize
    dt_current = F.softplus(dt_current + mamba_block.dt_proj.bias.float())
    
    A_bar = torch.exp(torch.einsum("bd,dn->bdn", dt_current, A_static))
    B_bar = torch.einsum("bd,bn->bdn", dt_current, B_current)

    return {
        "A": A_static,
        "B": B_current,
        "C": C_current,
        "D": D_static,
        "dt": dt_current,
        "A_bar": A_bar,
        "B_bar": B_bar
    }