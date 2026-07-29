import os
import copy
import numpy as np
import torch
import pandas as pd
from src.sample.utils.saving_utils import save_df_to_csv, save_training_dataset
from src.sample.utils.plotting_utils import plot_signals, plot_stacked

import numpy as np
import pandas as pd
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

def compute_mimo_dtw_distance(seq_A, seq_B):
    """
    Computes multivariate DTW distance between two sequence arrays 
    of shape [Seq_Len, Channels].
    """
    # fastdtw handles multidimensional trajectories using Euclidean distance point-by-point
    distance, path = fastdtw(seq_A, seq_B, dist=euclidean)
    return distance

def compute_pairwise_dtw_matrix(sequences):
    """
    Computes symmetric pairwise DTW distance matrix across N sequences.
    
    Args:
        sequences: numpy array of shape [Num_Seqs, Seq_Len, Channels]
    Returns:
        dist_matrix: [Num_Seqs, Num_Seqs] symmetrical distance matrix
    """
    n_seqs = len(sequences)
    dist_matrix = np.zeros((n_seqs, n_seqs))
    
    print(f"📊 Computing pairwise DTW distance matrix for {n_seqs} sequences...")
    for i in range(n_seqs):
        for j in range(i + 1, n_seqs):
            d = compute_mimo_dtw_distance(sequences[i], sequences[j])
            dist_matrix[i, j] = d
            dist_matrix[j, i] = d
            
    return dist_matrix

def summarize_dtw_characterization(dist_matrix, signal_name="Signal"):
    """
    Extracts statistical summary metrics from a DTW distance matrix.
    """
    n_seqs = dist_matrix.shape[0]
    
    # Upper triangle indices (excluding diagonal self-distance 0)
    triu_indices = np.triu_indices(n_seqs, k=1)
    pairwise_distances = dist_matrix[triu_indices]
    
    # Sequence Medoid: The sequence with the lowest average distance to all others
    mean_distances_per_seq = dist_matrix.mean(axis=1)
    medoid_idx = int(np.argmin(mean_distances_per_seq))
    outlier_idx = int(np.argmax(mean_distances_per_seq))
    
    metrics = {
        "signal": signal_name,
        "mean_dtw_dist": np.mean(pairwise_distances),
        "std_dtw_dist": np.std(pairwise_distances),
        "min_dtw_dist": np.min(pairwise_distances),
        "max_dtw_dist": np.max(pairwise_distances),
        "medoid_seq_index": medoid_idx,
        "outlier_seq_index": outlier_idx,
    }
    
    print(f"\n--- DTW Summary: {signal_name} ---")
    print(f"  • Diversity (Mean DTW Distance) : {metrics['mean_dtw_dist']:.2f} ± {metrics['std_dtw_dist']:.2f}")
    print(f"  • Most Typical Sequence (Medoid): Index {medoid_idx}")
    print(f"  • Most Unique Sequence (Outlier): Index {outlier_idx}")
    
    return metrics



from sklearn.cluster import DBSCAN

def cluster_dtw_dbscan(dtw_matrix, eps=5.0, min_samples=3):
    """
    Discovers natural clusters and detects outlier sequences using DBSCAN.
    """
    db = DBSCAN(eps=eps, min_samples=min_samples, metric='precomputed')
    labels = db.fit_predict(dtw_matrix)
    
    # label == -1 means the trajectory is an outlier/anomaly
    num_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    num_outliers = list(labels).count(-1)
    
    print(f"Discovered {num_clusters} clusters with {num_outliers} outlier sequences.")
    return labels


def cluster_and_plot_dbscan(
    dtw_matrix, 
    sequences, 
    time_axis, 
    eps=15.0, 
    min_samples=2, 
    channel_idx=0, 
    dirname="dbscan_clusters"
):
    """
    Clusters DTW matrix using DBSCAN and renders each cluster using plot_signals.
    
    Args:
        dtw_matrix: [N, N] precomputed DTW distance matrix
        sequences: Numpy array [N_seqs, Seq_Len, N_channels]
        time_axis: 1D array of time steps (t)
        eps: Max DTW distance between two sequences to be considered neighbors
        min_samples: Min sequences required to form a dense region (cluster)
        channel_idx: Channel index to plot (e.g., 0 for u1)
        dirname: Directory to save cluster images via plot_signals
    """
    # 1. Run DBSCAN on the precomputed DTW distance matrix
    db = DBSCAN(eps=eps, min_samples=min_samples, metric='precomputed')
    cluster_labels = db.fit_predict(dtw_matrix)
    
    unique_clusters = np.unique(cluster_labels)
    n_clusters = len(unique_clusters) - (1 if -1 in unique_clusters else 0)
    n_outliers = np.sum(cluster_labels == -1)
    
    print(f"\n🔍 DBSCAN Discovered {n_clusters} clusters and {n_outliers} outlier sequences (eps={eps}, min_samples={min_samples}).")

    cluster_images = []

    # 2. Iterate through each cluster (including outliers at -1) and plot
    for cluster_id in unique_clusters:
        seq_indices = np.where(cluster_labels == cluster_id)[0]
        
        # Extract 1D trajectory arrays for the target channel
        cluster_signals = [sequences[idx, :, channel_idx] for idx in seq_indices]
        
        # Build cluster label descriptor
        if cluster_id == -1:
            cluster_name = f"Outliers / Noise ({len(seq_indices)} seqs)"
            filename = f"dbscan_outliers_channel_{channel_idx + 1}.png"
        else:
            cluster_name = f"Cluster {cluster_id} ({len(seq_indices)} seqs)"
            filename = f"dbscan_cluster_{cluster_id}_channel_{channel_idx + 1}.png"
            
        # Optional trace labels if group is small
        labels = [f"Seq {idx}" for idx in seq_indices] if len(seq_indices) <= 8 else None

        # 3. Render using your plot_signals utility
        img = plot_signals(
            t=time_axis,
            signals=cluster_signals,
            labels=labels,
            title=f"DBSCAN {cluster_name} — Channel u_{channel_idx + 1}",
            xlabel="Time (h)",
            ylabel=f"u_{channel_idx + 1}",
            figsize=(6, 4),
            show=True,
            filename=filename,
            dirname=dirname,
            asp=0.5
        )
        cluster_images.append(img)

    return cluster_labels, cluster_images

def generate_signal_single(hyperparam_config, channel_idx=1):
    """
    Generate smooth, band-limited control signals using an Active Shielding methodology
    to guarantee that each unique MIMO channel stays completely within its hard limits,
    utilizing independent, channel-specific lambda (bandwidth) and p (amplitude) settings.
    """
    sig_cfg = hyperparam_config["signal"]
    train_cfg = hyperparam_config["train"]
    plant_cfg = hyperparam_config["plant"]
    
    batch_size = train_cfg["batch_size"]
    seq_len = sig_cfg["seq_len"]
    device = train_cfg["device"]
    
    # 🎯 NEW: Dynamic Channel-Specific Lambda (Bandwidth) Extraction
    lambd = sig_cfg.get(f"u_{channel_idx}_lambd")
    if lambd is None: 
        lambd = sig_cfg["lambd"]  # Global fallback
    
    # 🎯 NEW: Dynamic Channel-Specific Configured Amplitude Extraction
    configured_p = sig_cfg.get(f"u_{channel_idx}_p")
    if configured_p is None: 
        configured_p = sig_cfg["p"]  # Global fallback
    
    # Step 1: Sample values from a uniform distribution [-1, 1]
    raw = torch.rand((batch_size, seq_len), device=device) * 2 - 1
    
    # Step 2: Fourier-transform to frequency domain
    fft_sig = torch.fft.rfft(raw, dim=1)
    freqs = torch.fft.rfftfreq(seq_len, d=sig_cfg["dt"])
    
    # Step 3: Drop frequencies above 1/lambda using the channel-specific lambda
    cutoff = 1.0 / lambd
    fft_sig[:, freqs > cutoff] = 0
    
    # Step 4: Inverse-Fourier-transform
    v_train = torch.fft.irfft(fft_sig, n=seq_len, dim=1)
    
    # Step 5: Normalize to [-1, 1]
    v_min = v_train.min(dim=1, keepdim=True)[0]
    v_max = v_train.max(dim=1, keepdim=True)[0]
    v_norm = 2 * (v_train - v_min) / (v_max - v_min + 1e-8) - 1
    
    # Dynamic Channel-Specific Center Value Extraction
    c_min = plant_cfg[f"u_{channel_idx}_D_center_min"]
    
        
    c_max = plant_cfg[f"u_{channel_idx}_D_center_max"]
    

    # Read channel-specific hard boundaries or fall back to system defaults
    u_hard_min = plant_cfg.get(f"u_{channel_idx}_hard_min")
    
    
    u_hard_max = plant_cfg.get(f"u_{channel_idx}_hard_max")
    
    #print("c_min", c_min)
    #print("c_max", c_max)
    # Generate random baseline centers across the channel-specific configured range
    u_center = torch.rand((batch_size, 1), device=device) * (c_max - c_min) + c_min
    
    # 🎯 ACTIVE SHIELDING: Calculate exact allowable deviation limits per trajectory row
    #print("u_hard_max:", u_hard_max)
    #print("u_center", u_center)
    dist_to_max = u_hard_max - u_center
    dist_to_min = u_center - u_hard_min
    max_safe_p = torch.minimum(dist_to_max, dist_to_min)
    
    # Adapt amplitude: use the channel-specific configured p, but scale down if it approaches boundaries
    adaptive_p = torch.minimum(torch.full_like(max_safe_p, configured_p), max_safe_p * 0.98)
    
    u_buffer = u_center + (v_norm * adaptive_p)
    
    # Guard clamp for precision floating point margins
    u_buffer = torch.clamp(u_buffer, u_hard_min, u_hard_max)
    
    return u_buffer, u_center


def generate_signals(hyperparam_config):
    """Generate independent MIMO control vectors using index-aware tracking."""
    train_cfg = hyperparam_config["train"]
    mamba_cfg = hyperparam_config["mamba"]
    plant_cfg = hyperparam_config["plant"]

    batch_size = train_cfg["batch_size"]
    output_dim = plant_cfg["output_dim"] 
    input_dim = plant_cfg["input_dim"]

    u_buffer = []
    D_center_list = []

    for i in range(input_dim):
        # Pass 1-based index to resolve channel configurations cleanly
        u_single, D_center = generate_signal_single(hyperparam_config, channel_idx=i+1)
        u_buffer.append(u_single.unsqueeze(-1))
        D_center_list.append(D_center)

    u_buffer = torch.cat(u_buffer, dim=-1)  
    D_center = torch.stack(D_center_list, dim=-1) 

    return u_buffer, D_center


def generate_signals_mix(hyperparam_config):
    """Generate independent MIMO control vectors mixing Canaday signals with constant signals."""
    train_cfg = hyperparam_config["train"]
    mamba_cfg = hyperparam_config["mamba"]
    sig_cfg = hyperparam_config["signal"]

    batch_size = train_cfg["batch_size"]
    output_dim = mamba_cfg["output_dim"] 
    seq_len = int(sig_cfg["seq_len"])
    
    # Probability of a batch item being a constant signal (e.g., 0.3 = 30%)
    # Default to 0.0 if not specified in config to remain backward compatible
    constant_prob = train_cfg["constant_signal_probability"]

    u_buffer = []
    D_center_list = []

    for i in range(output_dim):
        # 1. Generate the standard baseline Canaday signal for the channel
        u_single, D_center = generate_signal_single(hyperparam_config, channel_idx=i+1)
        
        # u_single shape: [batch_size, seq_len]
        # 2. Determine which batch items will be replaced with constants
        # We perform this independently per channel or globally per batch element
        for b in range(batch_size):
            if np.random.rand() < constant_prob:
                # Pick a random constant value within your physical input range
                # For the Trophophase plant, u1 bounds are [0.0, 1.0]
                u_min = hyperparam_config["plant"].get(f"u_{i+1}_min", 0.0)
                u_max = hyperparam_config["plant"].get(f"u_{i+1}_max", 1.0)
                
                # Sample a random continuous step value
                constant_val = u_min + (u_max - u_min) * np.random.rand()
                
                # Overwrite this specific batch sequence index with a flat line
                u_single[b, :] = constant_val
                
                # (Optional) If D_center tracking matters for the constant, zero it out or preserve it
                D_center[b, :] = 0.0 

        u_buffer.append(u_single.unsqueeze(-1))
        D_center_list.append(D_center)

    u_buffer = torch.cat(u_buffer, dim=-1)  # Shape: [batch_size, seq_len, output_dim]
    D_center = torch.stack(D_center_list, dim=-1) 

    return u_buffer, D_center

def generate_training_batch(plant, hyperparam_config):
    """
    Simulates the system and returns raw, continuous time-series arrays.
    No sliding window or derivative slicing is performed here.
    
    Returns:
        - raw_u: [batch_size, seq_len, output_dim]
        - raw_y: [batch_size, seq_len, input_dim]
        - raw_states: [batch_size, seq_len, state_dim]
        - D_center: [batch_size, output_dim]
    """
    sig_cfg = hyperparam_config["signal"]
    train_cfg = hyperparam_config["train"]

    seq_len = int(sig_cfg["seq_len"])
    dt = sig_cfg["dt"]
    batch_size = int(train_cfg["batch_size"])
    device = train_cfg["device"]

    # Initialize plant state
    state = plant.get_initial_state(batch_size)

    # Generate smooth input signals (u)
    u_buffer, D_center = generate_signals(hyperparam_config)

    raw_y_history = []
    raw_u_history = []
    raw_state_history = []

    # Simulate step-by-step
    for t_idx in range(seq_len):
        t = t_idx * dt
        u_signal = u_buffer[:, t_idx, :]  # [batch_size, output_dim]
        y_t = plant.get_y(state, t)       # [batch_size, input_dim]

        raw_y_history.append(y_t)
        raw_u_history.append(u_signal)
        raw_state_history.append(state.clone())

        state, _ = plant.step(state, u_signal, t, dt)
        state = state.detach()

    # Stack raw arrays along the time dimension (dim=1)
    raw_u = torch.stack(raw_u_history, dim=1).to(device)       # [batch_size, seq_len, output_dim]
    raw_y = torch.stack(raw_y_history, dim=1).to(device)       # [batch_size, seq_len, input_dim]
    raw_states = torch.stack(raw_state_history, dim=1).to(device) # [batch_size, seq_len, state_dim]

    return raw_u, raw_y, raw_states, D_center






# =====================================================================
# 3. Clean Dataset Compilation and Disk Exporter
# =====================================================================
import matplotlib.pyplot as plt

def plot_all_signals_overlay(signal_tensor, dt, dirname, signal_type, show_plot=True):
    """
    Plots all validated sequences overlaid on a single multi-channel plot.
    
    signal_tensor shape: [Num_Valid_Seqs, Seq_Len, Channels]
    signal_type: "Inputs_u" or "Outputs_y"
    """
    sig_np = signal_tensor.cpu().numpy()
    num_seqs, seq_len, num_channels = sig_np.shape
    time_axis = np.arange(seq_len) * dt

    prefix = "u" if "u" in signal_type.lower() else "y"
    #title_label = "Control Inputs ($u$)" if prefix == "u" else "System Outputs ($y$)"

    fig, axes = plt.subplots(num_channels, 1, figsize=(10, 3 * num_channels), sharex=True)
    if num_channels == 1:
        axes = [axes]  # Ensure axes list is iterable for 1D cases

    for i in range(num_channels):
        ax = axes[i]
        # Overlay each validated sequence with soft transparency
        for s_idx in range(num_seqs):
            ax.plot(time_axis, sig_np[s_idx, :, i], alpha=0.30, linewidth=1.1)

        # Highlight the ensemble average trajectory
        mean_sig = np.mean(sig_np[:, :, i], axis=0)
        ax.plot(time_axis, mean_sig, color="black", linestyle="--", linewidth=2.0, label="Ensemble Mean")

        ax.set_ylabel(rf"${prefix}_{{{i+1}}}$")
        #ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend(loc="upper right")

    axes[-1].set_xlabel(rf"$t \; / \; \mathrm{{h}}$")
    #fig.suptitle(f"All Validated {title_label} — Ensembles ($N={num_seqs}$ Sequences)", fontsize=12, fontweight="bold")
    plt.tight_layout()

    # Save to disk
    os.makedirs(dirname, exist_ok=True)
    save_path = os.path.join(dirname, f"all_{signal_type.lower()}_sequences_overlay.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    
    if show_plot:
        plt.show()
    else:
        plt.close(fig)

    print(f"🖼️ Overlay plot of all {num_seqs} {signal_type} saved to: {save_path}")

def generate_and_save_dataset(
    plant,
    hyperparam_config,
    dirname,
    show_plots=False,
    show_overlay_plot=False,
    save_logs=False
):
    """
    Generates and saves a raw continuous MIMO dataset. 
    Slicing, windowing, and target calculations are offloaded to training.
    
    Excludes any out-of-bounds curves from the final dataset.
    """
    sig_cfg = hyperparam_config["signal"]
    train_cfg = hyperparam_config["train"]
    plant_cfg = hyperparam_config["plant"]
    device = train_cfg["device"]
    dt = sig_cfg["dt"]
    
    input_dim = plant_cfg["input_dim"]   # e.g., (y1, y2)
    output_dim = plant_cfg["output_dim"] # e.g., (u1, u2)
    total_sequences = int(train_cfg["batch_size"])
    
    logs_dir = os.path.join(dirname, "dataset_logs")
    plots_dir = os.path.join(dirname, "dataset_plots")

    print(f"🚀 Running batch simulation for {total_sequences} sequences...")
    
    # 1. Fetch raw, un-sliced continuous histories from the simulation engine
    u_raw, y_raw, state_raw, batch_d_centers = generate_training_batch(plant, hyperparam_config)

    u_np = u_raw.cpu().numpy()          # Shape: [Total_Seqs, Seq_Len, output_dim]
    y_np = y_raw.cpu().numpy()          # Shape: [Total_Seqs, Seq_Len, input_dim]
    state_np = state_raw.cpu().numpy()  # Shape: [Total_Seqs, Seq_Len, state_dim]
    batch_d_centers_np = batch_d_centers.cpu().numpy()  

    violated_sequences_count = 0
    valid_u_list, valid_y_list, valid_states_list, valid_dfs = [], [], [], []
    per_sequence_correlations = []
    valid_idx_counter = 0

    # 2. Iterate through and validate each simulated sequence
    for s_idx in range(total_sequences):
        u = u_np[s_idx]             # [Seq_Len, output_dim]
        y_t = y_np[s_idx]           # [Seq_Len, input_dim]
        states = state_np[s_idx]     # [Seq_Len, state_dim]

        seq_has_violation = False

        # 📊 DYNAMIC OUTPUTS (y) BOUNDS CHECK
        for i in range(input_dim):
            single_seq_y = y_t[:, i]
            h_min = plant_cfg.get(f"y_{i+1}_hard_min")
            h_max = plant_cfg.get(f"y_{i+1}_hard_max")
            
            if h_min is not None and np.min(single_seq_y) < h_min:
                print(f"\033[93m⚠️ WARNING: [Seq {s_idx}] Output y_{i+1} dropped below limit! "
                      f"Bound: {h_min}, Min Found: {np.min(single_seq_y):.4f}\033[0m")
                seq_has_violation = True
                
            if h_max is not None and np.max(single_seq_y) > h_max:
                print(f"\033[93m⚠️ WARNING: [Seq {s_idx}] Output y_{i+1} exceeded limit! "
                      f"Bound: {h_max}, Max Found: {np.max(single_seq_y):.4f}\033[0m")
                seq_has_violation = True

        # 🕹️ DYNAMIC CONTROL INPUTS (u) BOUNDS CHECK
        for i in range(input_dim):
            single_seq_u = u[:, i]
            h_min = plant_cfg.get(f"f_u_{i+1}_hard_min") or plant_cfg.get(f"u_{i+1}_hard_min")
            h_max = plant_cfg.get(f"f_u_{i+1}_hard_max") or plant_cfg.get(f"u_{i+1}_hard_max")
            
            if h_min is not None and np.min(single_seq_u) < h_min:
                print(f"\033[93m⚠️ WARNING: [Seq {s_idx}] Input u_{i+1} dropped below limit! "
                      f"Bound: {h_min}, Min Found: {np.min(single_seq_u):.4f}\033[0m")
                seq_has_violation = True
                
            if h_max is not None and np.max(single_seq_u) > h_max:
                print(f"\033[93m⚠️ WARNING: [Seq {s_idx}] Input u_{i+1} exceeded limit! "
                      f"Bound: {h_max}, Max Found: {np.max(single_seq_u):.4f}\033[0m")
                seq_has_violation = True

        # 🛡️ DYNAMIC STATE VARIABLES HARD BOUNDS CHECK
        state_dim = states.shape[-1] 
        for i in range(state_dim):
            single_state_seq = states[:, i]
            x_min = plant_cfg.get(f"x_{i+1}_hard_min")
            x_max = plant_cfg.get(f"x_{i+1}_hard_max")
            
            if x_min is not None and np.min(single_state_seq) < x_min:
                print(f"\033[93m⚠️ WARNING: [Seq {s_idx}] State variable x_{i+1} dropped below limit! "
                      f"Bound: {x_min}, Min Found: {np.min(single_state_seq):.4f}\033[0m")
                seq_has_violation = True
                
            if x_max is not None and np.max(single_state_seq) > x_max:
                print(f"\033[93m⚠️ WARNING: [Seq {s_idx}] State variable x_{i+1} exceeded limit! "
                      f"Bound: {x_max}, Max Found: {np.max(single_state_seq):.4f}\033[0m")
                seq_has_violation = True

        # Rejection handling
        if seq_has_violation:
            violated_sequences_count += 1
            print(f"\033[91m🛑 Excluding [Seq {s_idx}] from final dataset structures.\033[0m")
            continue

        # 📈 Calculate Pearson Correlation for filtering
        seq_corr_metrics = {"sequence_index": valid_idx_counter}
        drop_due_to_correlation = False
        min_correlation_threshold = hyperparam_config["train"]["min_correlation_threshold"]
        
        for u_idx in range(input_dim):
            single_u_curve = u[:, u_idx]
            for y_idx in range(input_dim):
                single_y_curve = y_t[:, y_idx]
                
                if np.std(single_u_curve) == 0 or np.std(single_y_curve) == 0:
                    corr_val = 0.0
                else:
                    corr_val = np.corrcoef(single_u_curve, single_y_curve)[0, 1]
                
                if np.abs(corr_val) < min_correlation_threshold:
                    drop_due_to_correlation = True
                    print(f"\033[94mℹ️ INFO: [Seq {s_idx}] Rejected. Low correlation on u_{u_idx+1}──y_{y_idx+1} ({corr_val:+.4f})\033[0m")
                
                seq_corr_metrics[f"corr_u{u_idx+1}_y{y_idx+1}"] = corr_val

        if drop_due_to_correlation:
            violated_sequences_count += 1
            print(f"\033[91m🛑 Excluding [Seq {s_idx}] due to weak u-y behavior mapping.\033[0m")
            continue

        # Save valid references to our clean tracking lists
        valid_u_list.append(u_raw[s_idx])
        valid_y_list.append(y_raw[s_idx])
        valid_states_list.append(state_raw[s_idx])
        per_sequence_correlations.append(seq_corr_metrics)

        # Construct CSV log files dynamically
        time_axis = np.arange(len(u)) * dt
        columns, values = ["t"], [time_axis]

        for i in range(output_dim):
            columns.append(f"y_{i+1}")
            values.append(y_t[:, i])
        for i in range(input_dim):
            columns.append(f"u_{i+1}")
            values.append(u[:, i])
        for i in range(state_dim):
            columns.append(f"x_{i+1}")
            values.append(states[:, i])

        d_center = np.squeeze(batch_d_centers_np[s_idx])
        for i in range(input_dim):
            columns.append(f"D_center_u_{i+1}")
            d_val = float(d_center[i]) if output_dim > 1 else float(d_center)
            values.append(np.full(len(time_axis), d_val))

        seq_df = pd.DataFrame({col: val for col, val in zip(columns, values)})
        valid_dfs.append(seq_df)
        
        filename_base = f"sequence_{valid_idx_counter}.csv"
        valid_idx_counter += 1
        
        if save_logs:
            save_df_to_csv(seq_df, dirname=logs_dir, filename=filename_base)

        if show_plots:
            # Reuses your plotting routine using y_t as the base trajectory
            plot_configs = plant.get_plot_config()
            u_config = next((c for c in plot_configs if any(col.startswith("u") for col in c["cols"])), None)
            y_config = next((c for c in plot_configs if any(col.startswith("y") for col in c["cols"])), None)
                
            signals_to_plot = []
            labels_to_plot = []
            ylabels_to_plot = []

            for idx in range(u.shape[1]):
                signals_to_plot.append(u[:, idx])
                labels_to_plot.append([None])
                ylabels_to_plot.append(u_config["labels"][idx] if u_config else rf"Input $u_{{{idx+1}}}$")

            for idx in range(y_t.shape[1]):
                signals_to_plot.append(y_t[:, idx])
                labels_to_plot.append([None])
                ylabels_to_plot.append(y_config["labels"][idx] if y_config else rf"Output $y_{{{idx+1}}}$")

            dynamic_asp = [0.33] * len(signals_to_plot)
            plot_stacked(
                t=time_axis,
                signals=signals_to_plot,
                labels=labels_to_plot,
                xlabel=rf"$t \; / \; \mathrm{{h}}$",
                ylabel=ylabels_to_plot,
                asp=dynamic_asp,
                dirname=plots_dir,
                filename=f"{filename_base}_plot.png",
                show=True
            )
            # =================================================================
            # 3. SEPARATE STACKED PLOT FOR STATE VARIABLES (No legends needed)
            # =================================================================
            state_config = next((c for c in plot_configs if any(col.startswith("x") for col in c["cols"])), None)
            
            state_signals = []
            state_labels = []
            state_ylabels = []

            num_states = states.shape[1]  
            for idx in range(num_states):
                state_signals.append(states[:, idx])
                state_labels.append([None])  # Prevents redundant legend
                
                if state_config and idx < len(state_config["labels"]):
                    state_ylabels.append(state_config["labels"][idx])
                else:
                    state_ylabels.append(rf"State $x_{{{idx+1}}}$")

            state_asp = [0.33] * len(state_signals)
            
            plot_stacked(
                t=time_axis,
                signals=state_signals,
                labels=state_labels,
                xlabel=rf"$t \; / \; \mathrm{{h}}$",
                ylabel=state_ylabels,
                asp=state_asp,
                dirname=plots_dir,
                filename=f"{filename_base}_states_plot.png",
                show=True
            )

    # 🚨 FINAL BOUNDS VIOLATION SUMMARY
    violation_percentage = (violated_sequences_count / total_sequences) * 100
    valid_sequences_count = total_sequences - violated_sequences_count

    print("\n" + "="*60)
    print("📊 BOUNDS CHECK SUMMARY REPORT (y, u, & all states)")
    print(f"Total Raw Sequences Evaluated : {total_sequences}")
    if violated_sequences_count > 0:
        print(f"\033[91m\033[1m❌ Out-of-Bounds Curves Found : {violated_sequences_count} curves ({violation_percentage:.2f}%)\033[0m")
        print(f"\033[32m\033[1m✓ Clean Saved Dataset Curves   : {valid_sequences_count} curves accepted\033[0m")
    else:
        print("\033[92m\033[1m   All generated curves are within the defined hard boundaries! (100% accepted)\033[0m")
    print("="*60 + "\n")

    if valid_sequences_count == 0:
        print("\033[91m\033[1mCRITICAL ERROR: 100% of generated data curves violated bounds. No files exported.\033[0m")
        return

    # 3. Stack only validated raw tensors together 
    final_u_tensor = torch.stack(valid_u_list, dim=0).to(device)
    final_y_tensor = torch.stack(valid_y_list, dim=0).to(device)
    final_state_tensor = torch.stack(valid_states_list, dim=0).to(device)

    # 📈 NEW: Plot all validated u trajectories in one figure
    if show_overlay_plot and valid_sequences_count > 0:
        plot_all_signals_overlay(
            signal_tensor=final_u_tensor,
            dt=dt,
            dirname=plots_dir,
            signal_type="Inputs_u",
            show_plot=show_plots
        )

        plot_all_signals_overlay(
            signal_tensor=final_y_tensor,
            dt=dt,
            dirname=plots_dir,
            signal_type="Outputs_y",
            show_plot=show_plots
        )
    
    # Save global stats
    if per_sequence_correlations:
        per_seq_df = pd.DataFrame(per_sequence_correlations)
        save_df_to_csv(per_seq_df, dirname=dirname, filename="mimo_per_sequence_correlations.csv")
    
    data_to_save = {
        "u": final_u_tensor.cpu(),
        "y": final_y_tensor.cpu(),
        "states": final_state_tensor.cpu()
    }
    save_training_dataset(data_to_save, dirname=dirname)
    
    

    return data_to_save

def generate_and_save_dataset_marcia(
    plant,
    hyperparam_config,
    dirname,
    show_plots=False,
    save_logs=False
):
    """
    Generate and save a clean MIMO dataset tracking only current and future plant outputs.
    Removes historical delayed state inputs entirely. Logs a console warning specifying 
    which sequence transcends hard bounds defined inside hyperparam_config['plant'] for 
    outputs (y), control inputs (u), and state variables (x_1, x_2, x_3, x_4).
    
    EXCLUDES any out-of-bounds curves from the final training dataset and stats.
    """
    sig_cfg = hyperparam_config["signal"]
    train_cfg = hyperparam_config["train"]
    mamba_cfg = hyperparam_config["mamba"]
    plant_cfg = hyperparam_config["plant"]
    device = train_cfg["device"]
    dt = sig_cfg["dt"]
    
    input_dim = plant_cfg["input_dim"]   # e.g., (y1, y2)
    output_dim = plant_cfg["output_dim"] # e.g., (u1, u2)
    total_sequences = int(train_cfg["batch_size"])
    
    logs_dir = os.path.join(dirname, "dataset_logs")
    plots_dir = os.path.join(dirname, "dataset_plots")

    print(f"🚀 Running batch simulation for {total_sequences} sequences...")
    
    # Capturing the raw continuous state trajectory matrix from your generation engine
    x_tensor_raw, y_target_raw, batch_d_centers, state_tensor_raw = generate_training_batch(plant, hyperparam_config)

    x_np = x_tensor_raw.cpu().numpy()  
    y_np = y_target_raw.cpu().numpy()  
    batch_d_centers_np = batch_d_centers.cpu().numpy()  
    state_np = state_tensor_raw.cpu().numpy() # Shape expected: [Total_Seqs, Seq_Len, state_dim]

    violated_sequences_count = 0
    valid_x_list, valid_y_list, valid_dfs = [], [], []
    
    # 📈 NEW: List to accumulate correlation results for valid sequences
    per_sequence_correlations = []
    valid_idx_counter = 0

    for s_idx in range(total_sequences):
        y_t = x_np[s_idx, :, :input_dim]          
        y_next = x_np[s_idx, :, input_dim:]       
        u = y_np[s_idx]                           
        states = state_np[s_idx]                  # Slices state history for this specific sequence

        seq_has_violation = False

        # 📊 1. DYNAMIC OUTPUTS (y) BOUNDS CHECK
        for i in range(input_dim):
            single_seq_y = y_t[:, i]
            h_min = plant_cfg.get(f"y_{i+1}_hard_min")
            h_max = plant_cfg.get(f"y_{i+1}_hard_max")
            
            if h_min is not None and np.min(single_seq_y) < h_min:
                print(f"\033[93m⚠️ WARNING: [Seq {s_idx}] Output y_{i+1} dropped below limit! "
                      f"Bound: {h_min}, Min Found: {np.min(single_seq_y):.4f}\033[0m")
                seq_has_violation = True
                
            if h_max is not None and np.max(single_seq_y) > h_max:
                print(f"\033[93m⚠️ WARNING: [Seq {s_idx}] Output y_{i+1} exceeded limit! "
                      f"Bound: {h_max}, Max Found: {np.max(single_seq_y):.4f}\033[0m")
                seq_has_violation = True

        # 🕹️ 2. DYNAMIC CONTROL INPUTS (u) BOUNDS CHECK
        for i in range(output_dim):
            
            single_seq_u = u[:, i]
            h_min = plant_cfg.get(f"f_u_{i+1}_hard_min") or plant_cfg.get(f"u_{i+1}_hard_min")
            h_max = plant_cfg.get(f"f_u_{i+1}_hard_max") or plant_cfg.get(f"u_{i+1}_hard_max")
            
            if h_min is not None and np.min(single_seq_u) < h_min:
                print(f"\033[93m⚠️ WARNING: [Seq {s_idx}] Input u_{i+1} dropped below limit! "
                      f"Bound: {h_min}, Min Found: {np.min(single_seq_u):.4f}\033[0m")
                seq_has_violation = True
                
            if h_max is not None and np.max(single_seq_u) > h_max:
                print(f"\033[93m⚠️ WARNING: [Seq {s_idx}] Input u_{i+1} exceeded limit! "
                      f"Bound: {h_max}, Max Found: {np.max(single_seq_u):.4f}\033[0m")
                seq_has_violation = True

        # 🛡️ 3. DYNAMIC STATE VARIABLES HARD BOUNDS CHECK
        state_dim = states.shape[-1] 

        for i in range(state_dim):
            single_state_seq = states[:, i]
            x_min = plant_cfg.get(f"x_{i+1}_hard_min")
            x_max = plant_cfg.get(f"x_{i+1}_hard_max")
            
            if x_min is not None and np.min(single_state_seq) < x_min:
                print(f"\033[93m⚠️ WARNING: [Seq {s_idx}] State variable x_{i+1} dropped below limit! "
                      f"Bound: {x_min}, Min Found: {np.min(single_state_seq):.4f}\033[0m")
                seq_has_violation = True
                
            if x_max is not None and np.max(single_state_seq) > x_max:
                print(f"\033[93m⚠️ WARNING: [Seq {s_idx}] State variable x_{i+1} exceeded limit! "
                      f"Bound: {x_max}, Max Found: {np.max(single_state_seq):.4f}\033[0m")
                seq_has_violation = True

        #Rejection handling
        if seq_has_violation:
            violated_sequences_count += 1
            print(f"\033[91m🛑 Excluding [Seq {s_idx}] from final dataset structures.\033[0m")
            continue

        # 📈 NEW: Calculate Pearson Correlation for this individual valid sequence trace
        seq_corr_metrics = {"sequence_index": valid_idx_counter}
        drop_due_to_correlation = False
        min_correlation_threshold = hyperparam_config["train"]["min_correlation_threshold"]
        
        for u_idx in range(output_dim):
            single_u_curve = u[:, u_idx]
            for y_idx in range(input_dim):
                single_y_curve = y_t[:, y_idx]
                
                # Check for zero variance to avoid warnings/NaN values
                if np.std(single_u_curve) == 0 or np.std(single_y_curve) == 0:
                    corr_val = 0.0
                else:
                    corr_val = np.corrcoef(single_u_curve, single_y_curve)[0, 1]
                
                # 🎯 Filtering Rule: Check if the correlation magnitude drops below 0.7
                # (Using np.abs() keeps strong inverse behaviors like -0.85)
                if np.abs(corr_val) < min_correlation_threshold:
                    drop_due_to_correlation = True
                    print(f"\033[94mℹ️ INFO: [Seq {s_idx}] Rejected. Low correlation on u_{u_idx+1}──y_{y_idx+1} ({corr_val:+.4f})\033[0m")
                
                seq_corr_metrics[f"corr_u{u_idx+1}_y{y_idx+1}"] = corr_val

        # 🎯 Rejection handler specifically for weak correlation tracking
        if drop_due_to_correlation:
            violated_sequences_count += 1
            print(f"\033[91m🛑 Excluding [Seq {s_idx}] due to weak u-y behavior mapping.\033[0m")
            continue

        # Append execution histories ONLY if both bounds AND correlation conditions are satisfied
        valid_x_list.append(x_tensor_raw[s_idx])
        valid_y_list.append(y_target_raw[s_idx])
        per_sequence_correlations.append(seq_corr_metrics)

        # Construct logs dynamically
        time_axis = np.arange(len(u)) * dt
        columns, values = ["t"], [time_axis]

        for i in range(input_dim):
            columns.append(f"y_{i+1}_t")
            values.append(y_t[:, i])
        for i in range(input_dim):
            columns.append(f"y_{i+1}_next")
            values.append(y_next[:, i])
        for i in range(output_dim):
            columns.append(f"u_{i+1}")
            values.append(u[:, i])

        for i in range(state_dim):
            columns.append(f"x_{i+1}")
            values.append(states[:, i])

        d_center = np.squeeze(batch_d_centers_np[s_idx])
        for i in range(output_dim):
            columns.append(f"D_center_u_{i+1}")
            d_val = float(d_center[i]) if output_dim > 1 else float(d_center)
            values.append(np.full(len(time_axis), d_val))

        seq_df = pd.DataFrame({col: val for col, val in zip(columns, values)})
        valid_dfs.append(seq_df)
        
        filename_base = f"sequence_{valid_idx_counter}.csv"
        valid_idx_counter += 1
        
        if save_logs:
            save_df_to_csv(seq_df, dirname=logs_dir, filename=filename_base)

        if show_plots:
            # 📈 Build a concise correlation string for the title
            corr_strings = []
            for k, v in seq_corr_metrics.items():
                if k != "sequence_index":
                    short_label = k.replace("corr_", "").replace("_", "-")
                    corr_strings.append(f"{short_label}:{v:+.2f}")
            title_corr_suffix = "\n" + " | ".join(corr_strings)

            # Retrieve dynamic metadata from the active plant class
            plot_configs = plant.get_plot_config()
            
            # Identify the specific sub-configs by checking if any column starts with 'u' or 'y'
            # Identify the specific sub-configs by checking if any column starts with 'u' or 'y'
            u_config = next((c for c in plot_configs if any(col.startswith("u") for col in c["cols"])), None)
            y_config = next((c for c in plot_configs if any(col.startswith("y") for col in c["cols"])), None)
            x_config = next((c for c in plot_configs if any(col.startswith("x") for col in c["cols"])), None)
                
            signals_to_plot = []
            labels_to_plot = []
            ylabels_to_plot = []

            # 1. DYNAMIC INPUT SUBPLOTS (No legends needed)
            num_inputs = u.shape[1]
            for idx in range(num_inputs):
                signals_to_plot.append(u[:, idx])
                labels_to_plot.append([None])  # Prevents redundant legend
                
                if u_config and idx < len(u_config["labels"]):
                    ylabels_to_plot.append(u_config["labels"][idx])
                else:
                    ylabels_to_plot.append(rf"Input $u_{{{idx+1}}}$")

            # 2. DYNAMIC OUTPUT SUBPLOTS (Legends kept for overlapping signals)
            num_outputs = y_t.shape[1]
            for idx in range(num_outputs):
                signals_to_plot.append([y_t[:, idx], y_next[:, idx]])
                labels_to_plot.append(["Nominal output", "Future output"]) # Kept here
                
                if y_config and idx < len(y_config["labels"]):
                    ylabels_to_plot.append(y_config["labels"][idx])
                else:
                    ylabels_to_plot.append(rf"Output $y_{{{idx+1}}}$")

            # --- Generate Original Input/Output Plot ---
            dynamic_asp = [0.33] * len(signals_to_plot)
            plot_stacked(
                t=time_axis,
                signals=signals_to_plot,
                labels=labels_to_plot,
                xlabel=rf"$t \; / \; \mathrm{{h}}$",
                ylabel=ylabels_to_plot,
                asp=dynamic_asp,
                dirname=plots_dir,
                filename=f"{filename_base}_plot.png",
                show=True
            )

            # =================================================================
            # 3. SEPARATE STACKED PLOT FOR STATE VARIABLES (No legends needed)
            # =================================================================
            state_signals = []
            state_labels = []
            state_ylabels = []

            num_states = states.shape[1]  
            for idx in range(num_states):
                state_signals.append(states[:, idx])
                state_labels.append([None])  # Prevents redundant legend
                
                if x_config and idx < len(x_config["labels"]):
                    state_ylabels.append(x_config["labels"][idx])
                else:
                    state_ylabels.append(rf"State $x_{{{idx+1}}}$")

            state_asp = [0.33] * len(state_signals)
            
            plot_stacked(
                t=time_axis,
                signals=state_signals,
                labels=state_labels,
                xlabel=rf"$t \; / \; \mathrm{{h}}$",
                ylabel=state_ylabels,
                asp=state_asp,
                dirname=plots_dir,
                filename=f"{filename_base}_states_plot.png",
                show=True
            )

    # 🚨 FINAL BOUNDS VIOLATION & FILTERING SUMMARY
    violation_percentage = (violated_sequences_count / total_sequences) * 100
    valid_sequences_count = total_sequences - violated_sequences_count

    print("\n" + "="*60)
    print("📊 BOUNDS CHECK SUMMARY REPORT (y, u, & all states)")
    print(f"Total Raw Sequences Evaluated : {total_sequences}")
    if violated_sequences_count > 0:
        print(f"\033[91m\033[1m❌ Out-of-Bounds Curves Found : {violated_sequences_count} curves ({violation_percentage:.2f}%)\033[0m")
        print(f"\033[32m\033[1m✓ Clean Saved Dataset Curves   : {valid_sequences_count} curves accepted\033[0m")
    else:
        print("\033[92m\033[1m   All generated curves are within the defined hard boundaries! (100% accepted)\033[0m")
    print("="*60 + "\n")

    if valid_sequences_count == 0:
        print("\033[91m\033[1mCRITICAL ERROR: 100% of generated data curves violated bounds. No files exported.\033[0m")
        return

    # Stack only valid tensors back together 
    final_x_tensor = torch.stack(valid_x_list, dim=0).to(device)
    final_y_target = torch.stack(valid_y_list, dim=0).to(device)
    
    x_np_filtered = final_x_tensor.cpu().numpy()
    y_np_filtered = final_y_target.cpu().numpy()

    # 📈 NEW: Convert the per-sequence list of summaries into a DataFrame and export to disk
    if per_sequence_correlations:
        per_seq_df = pd.DataFrame(per_sequence_correlations)
        save_df_to_csv(per_seq_df, dirname=dirname, filename="mimo_per_sequence_correlations.csv")
        print(f"🎉 Per-sequence correlation matrix saved to: {os.path.join(dirname, 'mimo_per_sequence_correlations.csv')}")

    # 📊 Compile Clean Filtered Dataset Statistics
    print("📊 Compiling CLEAN (filtered) MIMO macro-dataset statistics...")
    stats_data = {"Metric": ["Mean", "Std_Dev", "Min", "25%", "50%", "75%", "Max"]}

    for i in range(input_dim):
        slice_y_t = x_np_filtered[:, :, i]
        stats_data[f"y_{i+1}_t"] = [
            np.mean(slice_y_t), np.std(slice_y_t), np.min(slice_y_t),
            np.percentile(slice_y_t, 25), np.percentile(slice_y_t, 50),
            np.percentile(slice_y_t, 75), np.max(slice_y_t)
        ]
        
    for i in range(input_dim):
        slice_y_next = x_np_filtered[:, :, input_dim + i]
        stats_data[f"y_{i+1}_next"] = [
            np.mean(slice_y_next), np.std(slice_y_next), np.min(slice_y_next),
            np.percentile(slice_y_next, 25), np.percentile(slice_y_next, 50),
            np.percentile(slice_y_next, 75), np.max(slice_y_next)
        ]

    for i in range(output_dim):
        slice_u = y_np_filtered[:, :, i]
        stats_data[f"u_{i+1}"] = [
            np.mean(slice_u), np.std(slice_u), np.min(slice_u),
            np.percentile(slice_u, 25), np.percentile(slice_u, 50),
            np.percentile(slice_u, 75), np.max(slice_u)
        ]

    stats_df = pd.DataFrame(stats_data)
    save_df_to_csv(stats_df, dirname=dirname, filename="mimo_dataset_global_statistics.csv")

    # Package verified training dataset structures
    data_to_save = {
        "x": final_x_tensor.cpu(),
        "y": final_y_target.cpu()
    }
    save_training_dataset(data_to_save, dirname=dirname)
    
    print(f"🎉 Clean tracking MIMO dataset generated successfully ({valid_sequences_count} valid traces preserved).")

def generate_training_batch_marcia(plant, hyperparam_config):
    """
    Generate a clean MIMO training batch without any look-back delay tensors.
    Slices perfectly to match only y_t and y_next. Also tracks and returns 
    the raw plant continuous state history [x1, x2, x3, x4] for validation clipping.
    """
    sig_cfg = hyperparam_config["signal"]
    train_cfg = hyperparam_config["train"]

    seq_len = int(sig_cfg["seq_len"])
    dt = sig_cfg["dt"]
    batch_size = int(train_cfg["batch_size"])
    delta_steps = int(train_cfg["delay_steps"]) # Step lookahead window length
    device = train_cfg["device"]

    # Initialize plant state
    state = plant.get_initial_state(batch_size)

    # Extend seq_len long enough to pull clean future states (y_next)
    extended_config = copy.deepcopy(hyperparam_config)
    extended_config["signal"]["seq_len"] = seq_len + delta_steps
    u_buffer, D_center = generate_signals(extended_config)

    raw_y_history = []
    raw_u_history = []
    raw_state_history = [] # 🛠️ NEW: Initialize state tracking history list

    total_simulation_steps = seq_len + delta_steps
    for t_idx in range(total_simulation_steps):
        t = t_idx * dt
        u_signal = u_buffer[:, t_idx, :]  # Shape: [batch_size, output_dim]
        y_t = plant.get_y(state, t)       # Shape: [batch_size, input_dim]

        raw_y_history.append(y_t)
        raw_u_history.append(u_signal)
        raw_state_history.append(state.clone()) # 🛠️ NEW: Store the current state tensor [batch_size, state_dim]

        state, _ = plant.step(state, u_signal, t, dt)
        state = state.detach()

    # Slice out exact pairs without historical delay padding
    all_y_t = raw_y_history[:seq_len]  
    all_y_next = raw_y_history[delta_steps : seq_len + delta_steps]  
    all_u = raw_u_history[:seq_len]  
    
    # 🛠️ NEW: Slice states matching your current time-step tracking window (0 to seq_len)
    all_states = raw_state_history[:seq_len]

    # Construct the training matrix tensors
    # x_tensor shape: [batch_size, seq_len, input_dim * 2] (holding [y_t, y_next])
    x_tensor = torch.cat([
        torch.stack(all_y_t, dim=1),  
        torch.stack(all_y_next, dim=1)
    ], dim=-1).to(device)

    # y_target shape: [batch_size, seq_len, output_dim]
    y_target = torch.stack(all_u, dim=1).to(device)

    # 🛠️ NEW: Stack states to output a clean [batch_size, seq_len, state_dim] tensor
    state_tensor = torch.stack(all_states, dim=1).to(device)

    # Return exactly 4 values to resolve the ValueError unpack crash
    return x_tensor, y_target, D_center, state_tensor