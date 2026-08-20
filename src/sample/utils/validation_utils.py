from typing import Dict, Any, List, Tuple, Union
import os
import torch
import numpy as np
import pandas as pd
from src.sample.utils.plotting_utils import plot_stacked
from src.sample.utils.saving_and_loading_utils import *
from src.sample.utils.general_utils import *


def simulate_tracking_stateful(
    model,
    plant,
    val_data,  # Tuple (Y_val, U_val, X_val) or Dict {"y": ..., "u": ..., "x": ...}
    hyperparam_config,
    x_scaler,
    y_scaler,
    dirname,
    mode="closed_loop",  # Options: "open_loop" or "closed_loop"
    plot_individual_plots=True,
):
    """Simulates trajectory tracking using validation data sequences in either

    Open-Loop or Closed-Loop mode, compares outputs, inputs, and state
    trajectories, and generates stacked comparison plots.

    Parameters:
    -----------
    model : nn.Module
        Trained inverse controller model.
    plant : class or object instance
        Physical plant model simulation environment.
    val_data : tuple or dict
        Validation set containing reference trajectories Y_val, controls U_val,
        and state trajectories X_val.
    hyperparam_config : dict
        Hyperparameter configuration dictionary.
    x_scaler, y_scaler : StandardScaler
        Fitted scalers for inputs (v_k features) and outputs (control signals
        u).
    dirname : str
        Directory path to save reports and figures.
    mode : str
        'closed_loop' (iterative prediction with real-time plant feedback) or
        'open_loop' (direct batch sequence prediction).
    plot_individual_plots : bool
        Whether to generate stacked comparison plots for every validation
        sequence.
    """
    # --- 1. UNPACK VALIDATION DATA ---
    X_val = None
    if isinstance(val_data, dict):
        Y_val = val_data["y"]
        U_val = val_data.get("u", None)
        X_val = val_data.get("x", None)
    elif isinstance(val_data, (tuple, list)):
        Y_val = val_data[0]
        U_val = val_data[1] if len(val_data) > 1 else None
        X_val = val_data[2] if len(val_data) > 2 else None
    else:
        raise ValueError(
            "val_data must be a tuple/list (Y_val, U_val, X_val) or dict with 'y', 'u', 'x' keys."
        )

    # Ensure Y_val tensor format: [batch_size, steps, input_dim]
    if not isinstance(Y_val, torch.Tensor):
        Y_val = torch.tensor(Y_val, dtype=torch.float32)

    batch_size, steps, input_dim = Y_val.shape

    # Format U_val and X_val if provided
    if U_val is not None and not isinstance(U_val, torch.Tensor):
        U_val = torch.tensor(U_val, dtype=torch.float32)

    if X_val is not None and not isinstance(X_val, torch.Tensor):
        X_val = torch.tensor(X_val, dtype=torch.float32)

    # Extract configuration
    train_cfg = hyperparam_config["train"]
    plant_cfg = hyperparam_config["plant"]
    training_data_cfg = hyperparam_config["training_data_cfg"]

    dt = training_data_cfg["dt"]
    device = train_cfg["device"]
    output_dim = plant_cfg["output_dim"]
    n_y = train_cfg["n_y"]
    n_u = train_cfg["n_u"]

    model.eval()

    # --- 2. HANDLE PLANT INSTANTIATION & INITIAL STATE ---
    if isinstance(plant, type):
        plant_instance = plant(hyperparam_config)
    else:
        plant_instance = plant

    # Initialize initial state x_0 using X_val[:, 0, :] if available
    if X_val is not None:
        init_state = X_val[:, 0, :].to(device=device, dtype=torch.float32)
    else:
        init_state = plant_instance.get_initial_state(batch_size)

    sample_state = plant_instance.get_initial_state(1)
    state_dim = sample_state.shape[-1]

    print(f"\n🚀 Running Validation Tracking ({mode.upper()} MODE)")
    print(
        f"📊 Evaluating {batch_size} validation trajectories over {steps} time steps..."
    )
    if X_val is not None:
        print(f"🔍 Ground truth states detected (state_dim = {state_dim})")

    # =========================================================================
    # MODE A: OPEN-LOOP VALIDATION
    # =========================================================================
    if mode.lower() == "open_loop":
        if hasattr(model, "core") and hasattr(model.core, "return_bc"):
            model.core.return_bc = False

        # Build windowed input sequences from reference data
        v_seqs_raw = []
        for b in range(batch_size):
            y_ref = Y_val[b].cpu().numpy()
            u_gt = (
                U_val[b].cpu().numpy()
                if U_val is not None
                else np.zeros((steps, output_dim))
            )

            v_traj = []
            for i in range(steps):
                next_idx = min(i + 1, steps - 1)
                r_target = y_ref[next_idx]

                y_window = y_ref[max(0, i - n_y) : i + 1]
                if len(y_window) < (n_y + 1):
                    pad_len = (n_y + 1) - len(y_window)
                    y_window = np.pad(
                        y_window, ((pad_len, 0), (0, 0)), mode="edge"
                    )

                y_hist_rev = y_window[::-1].flatten()

                if n_u > 0:
                    u_window = u_gt[max(0, i - n_u) : i]
                    if len(u_window) < n_u:
                        pad_len = n_u - len(u_window)
                        u_window = np.pad(
                            u_window, ((pad_len, 0), (0, 0)), mode="edge"
                        )
                    u_hist_rev = u_window[::-1].flatten()
                else:
                    u_hist_rev = np.array([])

                v_k = np.concatenate([r_target, y_hist_rev, u_hist_rev])
                v_traj.append(v_k)

            v_seqs_raw.append(np.array(v_traj))

        v_seqs_raw = np.array(v_seqs_raw)  # [batch_size, steps, v_dim]

        # Standardize features
        b_idx, s_idx, d_idx = v_seqs_raw.shape
        v_flat = v_seqs_raw.reshape(-1, d_idx)
        v_scaled = x_scaler.transform(v_flat).reshape(b_idx, s_idx, d_idx)
        v_tensor = torch.tensor(v_scaled, dtype=torch.float32, device=device)

        # Batch sequence prediction
        with torch.no_grad():
            if hasattr(model, "reset_memory"):
                model.reset_memory(batch_size=batch_size, device=device)

            u_pred_norm = model(v_tensor)
            u_pred_np = u_pred_norm.cpu().numpy()

        u_pred_flat = u_pred_np.reshape(-1, output_dim)
        u_unscaled_flat = y_scaler.inverse_transform(u_pred_flat)
        u_unscaled = u_unscaled_flat.reshape(batch_size, steps, output_dim)
        u_unscaled = np.clip(
            u_unscaled,
            plant_cfg["u_1_hard_min"],
            plant_cfg["u_1_hard_max"],
        )

        # Simulate state and output responses in open-loop
        all_y = torch.zeros((steps, batch_size, input_dim), device=device)
        all_states = torch.zeros(
            (steps, batch_size, state_dim), device=device
        )
        all_u = torch.tensor(
            u_unscaled, dtype=torch.float32, device=device
        ).permute(1, 0, 2)

        state = init_state.clone()
        with torch.no_grad():
            for i in range(steps):
                t = i * dt
                y_curr = plant_instance.get_y(state, t)
                all_y[i] = y_curr
                all_states[i] = state

                u_step = all_u[i]
                if output_dim == 1:
                    state, _ = plant_instance.step(
                        state=state, u=u_step[:, 0:1], t=t, dt=dt
                    )
                else:
                    state, _ = plant_instance.step(
                        state=state, u=u_step, t=t, dt=dt
                    )

    # =========================================================================
    # MODE B: CLOSED-LOOP VALIDATION
    # =========================================================================
    elif mode.lower() == "closed_loop":
        if hasattr(model, "core") and hasattr(model.core, "return_bc"):
            model.core.return_bc = True

        all_y = torch.zeros((steps, batch_size, input_dim), device=device)
        all_u = torch.zeros((steps, batch_size, output_dim), device=device)
        all_states = torch.zeros(
            (steps, batch_size, state_dim), device=device
        )

        state = init_state.clone()

        if hasattr(model, "reset_memory"):
            model.reset_memory(batch_size=batch_size, device=device)

        warmup_steps = 10
        initial_y = plant_instance.get_y(state, 0.0).cpu().numpy()
        y_histories = [[initial_y[b].copy()] for b in range(batch_size)]
        u_histories = [[np.zeros(output_dim)] for b in range(batch_size)]

        with torch.no_grad():
            for i in range(steps):
                t = i * dt
                y_current = plant_instance.get_y(state, t)
                y_curr_np = y_current.cpu().numpy()

                for b in range(batch_size):
                    y_histories[b].append(y_curr_np[b])
                    if len(y_histories[b]) > (n_y + 1):
                        y_histories[b].pop(0)

                next_idx = min(i + 1, steps - 1)
                tgt_r_np = Y_val[:, next_idx, :].cpu().numpy()

                if i < warmup_steps:
                    u_unscaled = 0.5 * np.ones((batch_size, output_dim))
                    u = torch.tensor(
                        u_unscaled, dtype=torch.float32, device=device
                    )
                else:
                    v_k_batch_raw = []
                    for b in range(batch_size):
                        y_window = np.array(y_histories[b])
                        y_hist_reversed = y_window[::-1].flatten()

                        u_window = (
                            np.array(u_histories[b])
                            if n_u > 0
                            else np.array([])
                        )
                        u_hist_reversed = (
                            u_window[::-1].flatten()
                            if n_u > 0
                            else np.array([])
                        )

                        v_k_single = np.concatenate(
                            [tgt_r_np[b], y_hist_reversed, u_hist_reversed]
                        )
                        v_k_batch_raw.append(v_k_single)

                    v_k_batch_raw = np.array(v_k_batch_raw)
                    v_k_scaled = x_scaler.transform(v_k_batch_raw)
                    v_k_tensor = torch.tensor(
                        v_k_scaled, dtype=torch.float32, device=device
                    )

                    u_norm_tensor = model.step(v_k_tensor)
                    u_norm_np = u_norm_tensor.cpu().numpy()
                    u_unscaled = y_scaler.inverse_transform(u_norm_np)

                    u_unscaled = np.clip(
                        u_unscaled,
                        plant_cfg["u_1_hard_min"],
                        plant_cfg["u_1_hard_max"],
                    )
                    u = torch.tensor(
                        u_unscaled, dtype=torch.float32, device=device
                    )

                for b in range(batch_size):
                    u_histories[b].append(u_unscaled[b])
                    if len(u_histories[b]) > n_u:
                        u_histories[b].pop(0)

                if output_dim == 1:
                    state, _ = plant_instance.step(
                        state=state, u=u[:, 0:1], t=t, dt=dt
                    )
                else:
                    try:
                        state, _ = plant_instance.step(
                            state=state, u=u, t=t, dt=dt
                        )
                    except TypeError:
                        kwargs = {
                            f"u{j+1}": u[:, j : j + 1] for j in range(output_dim)
                        }
                        state, _ = plant_instance.step(
                            state=state, t=t, dt=dt, **kwargs
                        )

                all_y[i] = y_current
                all_u[i] = u
                all_states[i] = state
    else:
        raise ValueError("mode parameter must be 'open_loop' or 'closed_loop'.")

    # =========================================================================
    # --- 3. METRICS, PLOTTING, & EXPORT ---
    # =========================================================================
    time_axis = np.arange(steps) * dt
    trajectory_reports = []
    trajectory_images = []

    y_np = all_y.cpu().numpy()  # [steps, batch_size, input_dim]
    u_np = all_u.cpu().numpy()  # [steps, batch_size, output_dim]
    s_np = all_states.cpu().numpy()  # [steps, batch_size, state_dim]

    # Reorder to [batch_size, steps, dim]
    s_sim_batch = np.transpose(s_np, (1, 0, 2))
    u_sim_batch = np.transpose(u_np, (1, 0, 2))
    y_sim_batch = np.transpose(y_np, (1, 0, 2))

    r_gt_batch = Y_val.cpu().numpy()  # [batch_size, steps, input_dim]
    u_gt_batch = U_val.cpu().numpy() if U_val is not None else None
    x_gt_batch = X_val.cpu().numpy() if X_val is not None else None

    # --- Generate DataFrames & Plots for each sequence ---
    for b in range(batch_size):
        state_dirname = os.path.join(dirname, f"{mode}_val_sequence_{b}")
        os.makedirs(state_dirname, exist_ok=True)

        y_traj_sim = y_sim_batch[b]  # [steps, input_dim]
        u_traj_sim = u_sim_batch[b]  # [steps, output_dim]
        s_traj_sim = s_sim_batch[b]  # [steps, state_dim]

        total_stacked_blocks = input_dim + output_dim
        df_data = {
            "time": np.tile(time_axis, total_stacked_blocks),
            "signal_type": np.repeat(
                [f"y_{i+1}" for i in range(input_dim)]
                + [f"u_{i+1}" for i in range(output_dim)],
                steps,
            ),
            "value": np.concatenate(
                [y_traj_sim[:, i] for i in range(input_dim)]
                + [u_traj_sim[:, i] for i in range(output_dim)]
            ),
        }

        # Append simulated states
        for i in range(state_dim):
            df_data[f"state_sim_{i+1}"] = np.tile(
                s_traj_sim[:, i], total_stacked_blocks
            )

        # Append ground-truth states
        if x_gt_batch is not None:
            s_traj_gt = x_gt_batch[b]
            for i in range(state_dim):
                df_data[f"state_gt_{i+1}"] = np.tile(
                    s_traj_gt[:, i], total_stacked_blocks
                )

        df_traj = pd.DataFrame(df_data)
        save_df_to_csv(
            df_traj, dirname=state_dirname, filename="state_report"
        )
        trajectory_reports.append(df_traj)

        # --- Stacked Plot Generation via plot_stacked ---
        if plot_individual_plots:
            signals = []
            labels = []
            ylabels = []

            # 1. Fetch plot configuration from plant (or model) if available
            plot_config = None
            if hasattr(plant, "get_plot_config"):
                plot_config = plant.get_plot_config()
            elif hasattr(model, "get_plot_config"):
                plot_config = model.get_plot_config()

            def get_config_ylabel(var_prefix, idx, fallback):
                if not plot_config:
                    return fallback
                for cfg in plot_config:
                    cols = cfg.get("cols", [])
                    if (
                        var_prefix in cols
                        or f"{var_prefix}_{idx+1}" in cols
                        or any(c.startswith(var_prefix) for c in cols)
                    ):
                        yl = cfg.get("ylabel")
                        if isinstance(yl, (list, tuple)):
                            return yl[idx] if idx < len(yl) else " / ".join(yl)
                        elif isinstance(yl, str):
                            return yl
                return fallback

            xlabel = "Time [s]"
            if plot_config:
                for cfg in plot_config:
                    if "xlabel" in cfg:
                        xl = cfg["xlabel"]
                        xlabel = xl[0] if isinstance(xl, (list, tuple)) else xl
                        break

            # 2. Output Signals
            for i in range(input_dim):
                signals.append([y_traj_sim[:, i], r_gt_batch[b, :, i]])
                labels.append(["Simulated", "Reference"])
                ylabels.append(get_config_ylabel("y", i, f"$y_{{{i+1}}}$"))

            # 3. Control Signals
            for j in range(output_dim):
                if u_gt_batch is not None:
                    signals.append([u_traj_sim[:, j], u_gt_batch[b, :, j]])
                    labels.append(["Simulated", "Ground Truth"])
                else:
                    signals.append(u_traj_sim[:, j])
                    labels.append(["Simulated"])
                ylabels.append(get_config_ylabel("u", j, f"$u_{{{j+1}}}$"))

            # 4. State Signals
            for k in range(state_dim):
                if x_gt_batch is not None:
                    signals.append([s_traj_sim[:, k], x_gt_batch[b, :, k]])
                    labels.append(["Simulated", "Ground Truth"])
                else:
                    signals.append(s_traj_sim[:, k])
                    labels.append(["Simulated"])
                ylabels.append(get_config_ylabel("x", k, f"$x_{{{k+1}}}$"))

            plot_filename = f"tracking_stacked_plot_seq_{b}"
            img = plot_stacked(
                t=time_axis,
                signals=signals,
                labels=labels,
                xlabel=xlabel,
                ylabel=ylabels,
                filename=plot_filename,
                dirname=state_dirname,
                asp=0.33,
                hspace=0.08,
            )
            trajectory_images.append(img)

    # =========================================================================
    # --- 4. COMPUTE PER-SEQUENCE METRICS AND DATASET AVERAGES ---
    # =========================================================================
    per_sequence_results = []

    for b in range(batch_size):
        seq_metrics = {
            "sequence_id": b,
            "outputs": {},
            "controls": {},
            "states": {},
        }

        # <<< PASTE / REPLACE HERE >>>
        # 1. Output tracking metrics (y_sim vs y_ref)
        for i in range(input_dim):
            y_sim = y_sim_batch[b, :, i][:, None]  # Reshape from (steps,) to (steps, 1)
            y_ref = r_gt_batch[b, :, i][:, None]
            seq_metrics["outputs"][f"y_{i+1}"] = compute_and_save_tracking_metrics(
                y_sim, y_ref, dt, dirname=None, suffix=f"y_{i+1}_seq_{b}"
            )

        # 2. Control signal metrics (u_sim vs u_gt)
        if u_gt_batch is not None:
            for j in range(output_dim):
                u_sim = u_sim_batch[b, :, j][:, None]  # Reshape to (steps, 1)
                u_gt = u_gt_batch[b, :, j][:, None]
                seq_metrics["controls"][f"u_{j+1}"] = compute_and_save_tracking_metrics(
                    u_sim, u_gt, dt, dirname=None, suffix=f"u_{j+1}_seq_{b}"
                )

        # 3. State tracking metrics (x_sim vs x_gt)
        if x_gt_batch is not None:
            for k in range(state_dim):
                x_sim = s_sim_batch[b, :, k][:, None]  # Reshape to (steps, 1)
                x_gt = x_gt_batch[b, :, k][:, None]
                seq_metrics["states"][f"state_{k+1}"] = compute_and_save_tracking_metrics(
                    x_sim, x_gt, dt, dirname=None, suffix=f"state_{k+1}_seq_{b}"
                )
        # <<< END PASTE >>>

        per_sequence_results.append(seq_metrics)

    # Calculate dataset-wide average metrics across all sequences
    dataset_averages = {"outputs": {}, "controls": {}, "states": {}}

    for category in ["outputs", "controls", "states"]:
        # Find all signal keys present for this category
        signal_keys = set()
        for seq in per_sequence_results:
            signal_keys.update(seq[category].keys())

        for sig_key in signal_keys:
            # Collect metric values for this signal across all batches
            all_metric_keys = per_sequence_results[0][category][sig_key].keys()
            avg_metrics = {}
            for m_key in all_metric_keys:
                vals = [
                    seq[category][sig_key][m_key]
                    for seq in per_sequence_results
                    if sig_key in seq[category] and m_key in seq[category][sig_key]
                ]
                avg_metrics[m_key] = float(np.mean(vals)) if len(vals) > 0 else np.nan

            dataset_averages[category][sig_key] = avg_metrics

    # =========================================================================
    # --- 5. BUILD & SAVE DATAFRAMES ---
    # =========================================================================
    # 1. Detailed per-sequence metrics DataFrame
    per_seq_rows = []
    for seq in per_sequence_results:
        seq_id = seq["sequence_id"]
        for category in ["outputs", "controls", "states"]:
            for signal_name, metrics in seq[category].items():
                per_seq_rows.append(
                    {"sequence_id": seq_id, "category": category, "signal": signal_name, **metrics}
                )

    df_per_sequence = pd.DataFrame(per_seq_rows)

    # 2. Overall dataset averages DataFrame
    avg_rows = []
    for category, category_dict in dataset_averages.items():
        for signal_name, metrics in category_dict.items():
            avg_rows.append(
                {"category": category, "signal": signal_name, **metrics}
            )

    df_averages = pd.DataFrame(avg_rows)

    # 3. Save CSV summary reports to the main mode directory (`dirname`)
    save_df_to_csv(
        df=df_per_sequence,
        dirname=dirname,
        filename=f"tracking_metrics_per_sequence_{mode}",
    )

    save_df_to_csv(
        df=df_averages,
        dirname=dirname,
        filename=f"tracking_metrics_dataset_averages_{mode}",
    )

    return {
        "mode": mode,
        "trajectory_dataframes": trajectory_reports,
        "trajectory_images": trajectory_images,
        "per_sequence_metrics": per_sequence_results,
        "dataset_averages": dataset_averages,
        "simulated_outputs": y_sim_batch,
        "simulated_controls": u_sim_batch,
        "simulated_states": s_sim_batch,
        "gt_states": x_gt_batch,
    }


def simulate_tracking_stateful_external_ref_trajectory(
    model,
    plant,
    r_trajectories,  # List of reference trajectories, one for each output dimension. Shape: [steps] for each trajectory.
    hyperparam_config,
    x_scaler,
    y_scaler,
    dirname,
    plot_individual_plots=False
):
    """
    Simulates a controlled MIMO plant over a specified time horizon using stateful,
    step-by-step inference (carrying forward Mamba hidden states) while tracking a
    separate reference trajectory for each output dimension.
    
    Aligned with the robust plant instantiation and plotting routines of the validation rollout.
    """
    # Extract configuration sub-dictionaries
    train_cfg = hyperparam_config["train"]
    training_data_cfg = hyperparam_config["training_data_cfg"]
    sim_cfg = hyperparam_config["simulate"]
    plant_cfg = hyperparam_config["plant"]

    model.core.return_bc = True  # Enable returning B and C
    
    # Unpack specific parameters
    steps = sim_cfg["seq_len"]
    dt = training_data_cfg["dt"]
    batch_size = sim_cfg["batch_size"]
    device = train_cfg["device"]
    input_dim = plant_cfg["input_dim"]    # Number of plant outputs (y1, y2, ...)
    output_dim = plant_cfg["output_dim"]  # Number of control inputs (u1, u2, ...)

    # 🌟 1. HANDLE PLANT INSTANTIATION (Class vs. Instance)
    if isinstance(plant, type):
        plant_instance = plant(hyperparam_config)
    else:
        plant_instance = plant

    # Validate r_trajectories
    if len(r_trajectories) != input_dim:
        raise ValueError(f"Expected {input_dim} reference trajectories, got {len(r_trajectories)}")

    # Convert r_trajectories to a tensor of shape [steps, input_dim]
    r_trajectory = torch.stack(r_trajectories, dim=1)  # Shape: [steps, input_dim]
    r_np = r_trajectory.cpu().numpy()  # Shape: [steps, input_dim]

    # Initialize GPU tensor buffers
    all_y = torch.zeros((steps, batch_size, input_dim), device=device)  
    all_u = torch.zeros((steps, batch_size, output_dim), device=device)  
    
    # Dynamically query plant state dimensions to avoid hardcoding
    sample_state = plant_instance.get_initial_state(1)
    state_dim = sample_state.shape[-1]
    all_states = torch.zeros((steps, batch_size, state_dim), device=device)  

    state = plant_instance.get_initial_state(batch_size)

    # Prepare model for evaluation mode
    model.eval()
    ssm_history = {
        "step": [], "time": [],
        "A_bar": [], "B_bar": [], "C": [], "dt": []
    }
    print(f"📈 Testing Stateful MIMO Trajectory Tracking: {batch_size} trajectories across {steps} steps...")

    # CORRECT MEMORY INITIALIZATION
    model.reset_memory(batch_size=batch_size, device=device)

    # INITIALIZE SLIDING WINDOW RUNNING BUFFERS FOR THE BATCH
    n_y = train_cfg["n_y"]
    n_u = train_cfg["n_u"]

    # 🌟 Calculate the physical lookback threshold
    # Since we need (n_y + 1) past outputs and n_u past controls:
    warmup_steps = 10 # max(n_y + 1, n_u)

    # Seed history buffers with only the very first step instead of dummy-repeating them
    initial_y = plant_instance.get_y(state, 0.0).cpu().numpy()  # [batch_size, input_dim]
    y_histories = [[initial_y[b].copy()] for b in range(batch_size)]
    u_histories = [[np.zeros(output_dim)] for b in range(batch_size)]

    # Execute forward tracking simulation
    with torch.no_grad():
        for i in range(steps):
            t = i * dt
            y_current = plant_instance.get_y(state, t)  # Shape: [batch_size, input_dim]
            y_curr_np = y_current.cpu().numpy()

            # 1. Update running history with the newly observed plant output
            for b in range(batch_size):
                y_histories[b].append(y_curr_np[b])
                if len(y_histories[b]) > (n_y + 1):
                    y_histories[b].pop(0)

            # Look-ahead: Target reference state for the NEXT time-step (i+1)
            next_idx = min(i + 1, steps - 1)
            target_r = r_trajectory[next_idx].expand(batch_size, input_dim)
            tgt_r_np = target_r.cpu().numpy()

            # 🌟 Determine if we have enough physical history to start model control
            if i < warmup_steps:
                # --- WARMUP PHASE ---
                # Model does not act yet. Use safe default control actions (zeros)
                u_unscaled = 0.5* np.ones((batch_size, output_dim))
                u = torch.tensor(u_unscaled, dtype=torch.float32, device=device)
                
            else:
                # --- ACTIVE CONTROL PHASE ---
                # We construct the input vector v_k using only genuine accumulated histories
                v_k_batch_raw = []
                for b in range(batch_size):
                    y_window = np.array(y_histories[b])
                    y_hist_reversed = y_window[::-1].flatten()

                    u_window = np.array(u_histories[b]) if n_u > 0 else np.array([])
                    u_hist_reversed = u_window[::-1].flatten() if n_u > 0 else np.array([])

                    # Combine: [Target_Future, Past_Outputs, Past_Controls]
                    v_k_single = np.concatenate([tgt_r_np[b], y_hist_reversed, u_hist_reversed])
                    v_k_batch_raw.append(v_k_single)

                # Convert, scale, and infer
                v_k_batch_raw = np.array(v_k_batch_raw)
                v_k_scaled = x_scaler.transform(v_k_batch_raw)
                v_k_tensor = torch.tensor(v_k_scaled, dtype=torch.float32, device=device)

                u_norm_tensor = model.step(v_k_tensor)
                
                u_norm_np = u_norm_tensor.cpu().numpy() 
                u_unscaled = y_scaler.inverse_transform(u_norm_np)

                # Force physical actuator limits
                u_unscaled = np.clip(u_unscaled, plant_cfg["u_1_hard_min"], plant_cfg["u_1_hard_max"])  
                u = torch.tensor(u_unscaled, dtype=torch.float32, device=device)  

            # 2. Update control history with the chosen action
            for b in range(batch_size):
                u_histories[b].append(u_unscaled[b])
                if len(u_histories[b]) > n_u:
                    u_histories[b].pop(0)

            # Step the physical plant forward
            if output_dim == 1:
                state, _ = plant_instance.step(state=state, u=u[:, 0:1], t=t, dt=dt)
            else:
                try:
                    state, _ = plant_instance.step(state=state, u=u, t=t, dt=dt)
                except TypeError:
                    kwargs = {f"u{j+1}": u[:, j:j+1] for j in range(output_dim)}
                    state, _ = plant_instance.step(state=state, t=t, dt=dt, **kwargs)

            # Logging metrics
            all_y[i] = y_current  
            all_u[i] = u  
            all_states[i] = state 

    # --- PLOTTING & EXPORT CONFIGURATION ---
    time_axis = np.arange(steps) * dt
    trajectory_reports = []
    total_stacked_blocks = input_dim + output_dim
    
    # Retrieve the metadata configuration blocks from the plant safely
    plot_metadata = plant_instance.get_plot_config() if hasattr(plant_instance, "get_plot_config") else []

    # Safe lookup logic mirrored from validation sequence
    state_meta = next((c for c in plot_metadata if any(col.startswith("x") for col in c["cols"])), {})
    output_meta = next((c for c in plot_metadata if any(col.startswith("y") for col in c["cols"])), {})
    control_meta = next((c for c in plot_metadata if any(col.startswith("u") for col in c["cols"])), {})
    
    save_to_json(
        data=ssm_history,
        dirname=dirname,          
        filename="ssm_matrices_history"
    )
    
    # Parse and save individual trajectory records
    for b in range(batch_size):
        state_dirname = os.path.join(dirname, f"initial_state_{b}")
        os.makedirs(state_dirname, exist_ok=True)

        y_traj = all_y[:, b, :].cpu().numpy()            # Shape: [steps, input_dim]
        u_traj = all_u[:, b, :].cpu().numpy()            # Shape: [steps, output_dim]
        states_traj = all_states[:, b, :].cpu().numpy()  # Shape: [steps, state_dim]

        # Save DataFrame for this trajectory
        df_data = {
            "time": np.tile(time_axis, total_stacked_blocks),
            "signal_type": np.repeat(
                [f"y_{i+1}" for i in range(input_dim)] + [f"u_{i+1}" for i in range(output_dim)],
                steps
            ),
            "value": np.concatenate([
                y_traj[:, i] for i in range(input_dim)
            ] + [
                u_traj[:, i] for i in range(output_dim)
            ]),
        }
        
        for i in range(states_traj.shape[1]):
            df_data[f"state_{i+1}"] = np.tile(states_traj[:, i], total_stacked_blocks)

        df_traj = pd.DataFrame(df_data)
        save_df_to_csv(df_traj, dirname=state_dirname, filename="state_report")
        trajectory_reports.append(df_traj)

        if plot_individual_plots:
            # 1. Individual Plot: Control Signals
            for i in range(output_dim):
                labels_list = control_meta.get("labels", [])
                label = labels_list[i] if i < len(labels_list) else f"Control Input (u_{i+1})"
                title = control_meta.get("title", "Control Profile")
                
                plot_signals(
                    t=time_axis,
                    signals=[u_traj[:, i]],
                    labels=[label],
                    title=f"Trajectory {b}: {title}",
                    xlabel=rf"$t \; / \; \mathrm{{s}}$",
                    ylabel=control_meta.get("ylabel", "Action Value"),
                    dirname=state_dirname,
                    filename=f"plot_control_signal_u_{i+1}"
                )

            # 2. Individual Plot: Output tracking performance
            for i in range(input_dim):
                labels_list = output_meta.get("labels", [])
                ind_y_label = labels_list[0] if len(labels_list) > 0 else f"Output (y_{i+1})"
                ind_r_label = labels_list[1] if len(labels_list) > 1 else f"Target (r_{i+1})"
                title = output_meta.get("title", "Tracking Performance")
                
                plot_signals(
                    t=time_axis,
                    signals=[y_traj[:, i], r_np[:, i]],  
                    labels=[ind_y_label, ind_r_label],   
                    title=f"Trajectory {b}: {title}",
                    xlabel=rf"$t \; / \; \mathrm{{s}}$",
                    ylabel=output_meta.get("ylabel", "Signal Value"),
                    dirname=state_dirname,
                    filename=f"plot_output_tracking_y_{i+1}"
                )
                
            # 3. Individual Plot: Internal plant states
            for i in range(states_traj.shape[1]):
                labels_list = state_meta.get("labels", [])
                label = labels_list[i] if i < len(labels_list) else f"State x_{i+1}"
                title = state_meta.get("title", "Internal Plant States")
                
                plot_signals(
                    t=time_axis,
                    signals=[states_traj[:, i]],
                    labels=[label],
                    title=f"Trajectory {b}: {title} - {label}",
                    xlabel=rf"$t \; / \; \mathrm{{s}}$",
                    ylabel=state_meta.get("ylabel", "State Magnitude"),
                    dirname=state_dirname,
                    filename=f"plot_plant_state_x_{i+1}"
                )

    # =========================================================================
    # UNIFIED BATCH OVERLAY STACKED PLOT (Outputs + Controls + States)
    # =========================================================================
    y_np = all_y.cpu().numpy()       # Shape: [steps, batch_size, input_dim]
    u_np = all_u.cpu().numpy()       # Shape: [steps, batch_size, output_dim]
    s_np = all_states.cpu().numpy()  # Shape: [steps, batch_size, state_dim]

    # Helper function to query label/ylabel safely from plot_metadata
    def get_meta_info(prefix, index, default_label, default_ylabel):
        for block in plot_metadata:
            cols = block.get("cols", [])
            # Match block by column naming standard (e.g. 'y', 'y_1', 'x_2')
            if any(col == prefix or col.startswith(f"{prefix}_") or col.startswith(f"{prefix}") for col in cols):
                labels = block.get("labels", [])
                ylabel = block.get("ylabel", default_ylabel)
                
                # Fetch output label
                label_val = labels[index] if index < len(labels) else default_label
                
                # Handle cases where 'ylabel' is either a list or a string
                if isinstance(ylabel, list):
                    ylabel_val = ylabel[index] if index < len(ylabel) else default_ylabel
                else:
                    ylabel_val = ylabel if index == 0 else default_ylabel
                
                return label_val, ylabel_val
        
        return default_label, default_ylabel

    # Helper function to extract x-axis label safely
    time_meta = next((c for c in plot_metadata if "t" in c.get("cols", [])), {})
    time_xlabel_list = time_meta.get("xlabel", [r"$t \; / \; \mathrm{s}$"])
    xlabel_str = time_xlabel_list[0] if isinstance(time_xlabel_list, list) and time_xlabel_list else time_meta.get("xlabel", r"$t \; / \; \mathrm{s}$")

    signals_to_plot = []
    labels_to_plot = []
    ylabels_to_plot = []

    # -------------------------------------------------------------------------
    # 1. Stack System Outputs (y)
    # -------------------------------------------------------------------------
    for i in range(input_dim):
        ref_label, y_label = get_meta_info("y", i, f"Output y_{i+1}", f"$y_{i+1}$")
        
        row_signals = [y_np[:, j, i] for j in range(batch_size)] + [r_np[:, i]]
        row_labels = [f"{batch_size} Simulated Curves" if j == 0 else "" for j in range(batch_size)] + [f"Reference ({ref_label})"]
        
        signals_to_plot.append(row_signals)
        labels_to_plot.append(row_labels)
        ylabels_to_plot.append(y_label)

    # -------------------------------------------------------------------------
    # 2. Stack Control Actions (u)
    # -------------------------------------------------------------------------
    for i in range(output_dim):
        act_label, u_label = get_meta_info("u", i, f"Control u_{i+1}", f"$u_{i+1}$")
        
        row_signals = [u_np[:, j, i] for j in range(batch_size)]
        row_labels = [f"{batch_size} Simulated Curves" if j == 0 else "" for j in range(batch_size)]
        
        signals_to_plot.append(row_signals)
        labels_to_plot.append(row_labels)
        ylabels_to_plot.append(u_label)

    # -------------------------------------------------------------------------
    # 3. Stack Internal Plant States (x)
    # -------------------------------------------------------------------------
    for i in range(s_np.shape[2]):
        st_label, x_label = get_meta_info("x", i, f"State x_{i+1}", f"$x_{i+1}$")
        
        row_signals = [s_np[:, j, i] for j in range(batch_size)]
        row_labels = [f"{batch_size} Simulated Curves" if j == 0 else "" for j in range(batch_size)]
        
        signals_to_plot.append(row_signals)
        labels_to_plot.append(row_labels)
        ylabels_to_plot.append(x_label)

    # -------------------------------------------------------------------------
    # 4. Render Unified Stacked Figure
    # -------------------------------------------------------------------------
    plot_stacked(
        t=time_axis,
        signals=signals_to_plot,
        labels=labels_to_plot,
        xlabel=xlabel_str,
        ylabel=ylabels_to_plot,
        asp=[0.25] * len(signals_to_plot),  # Scaled aspect ratio for larger stack
        dirname=dirname,
        filename="batch_summary_system_trajectory_stacked.png",
        show=True
    )

    # --- METRICS COMPILATION ---
    tracking_metrics = {}
    for i in range(input_dim):
        y_traj = y_np[:, :, i]  # Shape: [steps, batch_size]
        r_traj = r_np[:, i]     # Shape: [steps]
        metrics = compute_and_save_tracking_metrics(y_traj, r_traj, dt, dirname, suffix=f"y_{i+1}")
        tracking_metrics[f"y_{i+1}"] = metrics

    return {
        "trajectory_dataframes": trajectory_reports,
        "metrics": tracking_metrics,
        "simulated_outputs": y_np,
        "simulated_controls": all_u.cpu().numpy()
    }




def simulate_tracking_esn(
    model,
    plant,
    r_trajectories,  
    hyperparam_config,
    x_scaler,
    y_scaler,
    dirname,
    plot_individual_plots=False
):
    """
    Simulates a controlled MIMO plant using torchdiffeq adaptive-step integration 
    over a specified time horizon while tracking separate reference trajectories using an ESN.
    """
    training_data_cfg = hyperparam_config["training_data_cfg"]
    sim_cfg = hyperparam_config["simulate"]
    esn_cfg = hyperparam_config.get("esn", hyperparam_config.get("plant", {}))
    plant_cfg = hyperparam_config["plant"]

    steps = sim_cfg["seq_len"]
    dt = training_data_cfg["dt"]
    batch_size = sim_cfg["batch_size"]
    
    # ESN runs on CPU via NumPy, but torchdiffeq can still use your configured device
    device = hyperparam_config["train"]["device"]
    
    input_dim = plant_cfg["input_dim"]    
    output_dim = plant_cfg["output_dim"]  

    if len(r_trajectories) != input_dim:
        raise ValueError(f"Expected {input_dim} reference trajectories, got {len(r_trajectories)}")

    r_trajectory = torch.stack(r_trajectories, dim=1).to(device)  
    r_np = r_trajectory.cpu().numpy()  

    all_y = torch.zeros((steps, batch_size, input_dim), device=device)  
    all_u = torch.zeros((steps, batch_size, output_dim), device=device)  
    
    sample_state = plant.get_initial_state(1)
    state_dim = sample_state.shape[-1]
    all_states = torch.zeros((steps, batch_size, state_dim), device=device)  

    # Force initial state to float32 to protect torchdiffeq execution graph
    state = plant.get_initial_state(batch_size).to(device=device, dtype=torch.float32)

    print(f"📈 Testing ESN MIMO Trajectory Tracking: {batch_size} trajectories across {steps} steps...")

    # ReservoirPy processes historical feedback paths step-by-step per batch index
    history_pairs = [[] for _ in range(batch_size)]

    # Instantiate our localized adaptive step wrapper
   

    # --- 🌀 STEP-BY-STEP CLOSED-LOOP ROLLOUT LOOP ---
    for i in range(steps):
        t_start = i * dt
        t_end = t_start + dt
        
        # 1. Obtain current tracking performance observable
        with torch.no_grad():
            y_current = plant.get_y(state, t_start).to(dtype=torch.float32)  

        target_r = r_trajectory[i].expand(batch_size, input_dim)  
        y_curr_np = y_current.cpu().numpy()  
        tgt_r_np = target_r.cpu().numpy()  

        # 2. Sequential Normalization Processing
        u_unscaled_batch = []
        for b_idx in range(batch_size):
            input_pair = np.hstack([y_curr_np[b_idx], tgt_r_np[b_idx]])  
            input_normalized = x_scaler.transform([input_pair])[0] if x_scaler else input_pair
            
            # ESN expects the historical trajectory up to this time step
            history_pairs[b_idx].append(input_normalized)
            curr_history = np.array(history_pairs[b_idx])  

            # 3. Model inference through ESN forward interface
            # Pass the complete historical trace for this index; take the final output step
            u_seq_norm = model.forward(curr_history)
            u_norm_step = u_seq_norm[-1, :]  # Shape: [output_dim]

            
            u_unscaled = y_scaler.inverse_transform([u_norm_step])[0]
                
            u_unscaled_batch.append(u_unscaled)

        # Convert back to array for hard-clipping and tensor conversion
        u_unscaled_arr = np.array(u_unscaled_batch)

        # 4. Enforce actuator hard clipping profiles 
        u_unscaled_arr = np.clip(u_unscaled_arr, plant_cfg["u_1_hard_min"], plant_cfg["u_1_hard_max"])  
        u = torch.tensor(u_unscaled_arr, dtype=torch.float32, device=device)  

        # 5. Logging current conditions BEFORE step integration update
        all_y[i] = y_current  
        all_u[i] = u  
        all_states[i] = state  

        # 6. INTEGRATION STEP 
        
        
        with torch.no_grad():
            state, _ = plant.step(state, u, t_start, dt)

        
    # --- PLOTTING & EXPORT BLOCKS (Fully Untouched) ---
    time_axis = np.arange(steps) * dt
    trajectory_reports = []
    total_stacked_blocks = input_dim + output_dim
    
    plot_metadata = plant.get_plot_config()
    
    for b in range(batch_size):
        state_dirname = os.path.join(dirname, f"initial_state_{b}")
        os.makedirs(state_dirname, exist_ok=True)

        y_traj = all_y[:, b, :].cpu().numpy()  
        u_traj = all_u[:, b, :].cpu().numpy()  
        states_traj = all_states[:, b, :].cpu().numpy()  

        df_data = {
            "time": np.tile(time_axis, total_stacked_blocks),
            "signal_type": np.repeat(
                [f"y_{i+1}" for i in range(input_dim)] + [f"u_{i+1}" for i in range(output_dim)],
                steps
            ),
            "value": np.concatenate([y_traj[:, i] for i in range(input_dim)] + [u_traj[:, i] for i in range(output_dim)]),
        }
        
        for i in range(states_traj.shape[1]):
            df_data[f"state_{i+1}"] = np.tile(states_traj[:, i], total_stacked_blocks)

        df_traj = pd.DataFrame(df_data)
        save_df_to_csv(df_traj, dirname=state_dirname, filename="state_report")
        trajectory_reports.append(df_traj)

        if plot_individual_plots:
            u_meta = next((item for item in plot_metadata if "u" in item["cols"]), {})
            for i in range(output_dim):
                label = u_meta.get("labels", [f"u_{i+1}"])[0] if i == 0 else f"Control Input (u_{i+1})"
                title = u_meta.get("title", "Control Profile") if i == 0 else f"Control Input Profile (u_{i+1})"
                plot_signals(t=time_axis, signals=[u_traj[:, i]], labels=[label], title=f"Trajectory {b}: {title}", xlabel="Time (h)", ylabel=u_meta.get("ylabel", "Action Value"), dirname=state_dirname, filename=f"plot_control_signal_u_{i+1}")

            y_meta = next((item for item in plot_metadata if "y" in item["cols"]), {})
            meta_labels_ind = y_meta.get("labels", [])
            for i in range(input_dim):
                title = y_meta.get("title", "Tracking Performance")
                if len(meta_labels_ind) > (i * 2 + 1):
                    ind_y_label = meta_labels_ind[i * 2]
                    ind_r_label = meta_labels_ind[i * 2 + 1]
                elif len(meta_labels_ind) > i:
                    ind_y_label = meta_labels_ind[i]
                    ind_r_label = f"Target (r_{i+1})"
                else:
                    ind_y_label = f"Output (y_{i+1})"
                    ind_r_label = f"Target (r_{i+1})"
                
                plot_signals(t=time_axis, signals=[y_traj[:, i], r_np[:, i]], labels=[ind_y_label, ind_r_label], title=f"Trajectory {b}: {title} (y_{i+1})", xlabel="Time (h)", ylabel=y_meta.get("ylabel", "Signal Value"), dirname=state_dirname, filename=f"plot_output_tracking_y_{i+1}")

            for i in range(states_traj.shape[1]):
                x_meta = next((item for item in plot_metadata if "x" in item["cols"]), {})
                label = x_meta.get("labels", [f"State x_{i+1}"])[0]
                title = x_meta.get("title", f"Internal Plant State (x_{i+1})")
                plot_signals(t=time_axis, signals=[states_traj[:, i]], labels=[label], title=f"Trajectory {b}: {title}", xlabel="Time (h)", ylabel=x_meta.get("ylabel", "State Magnitude"), dirname=state_dirname, filename=f"plot_plant_state_x_{i+1}")

    y_np = all_y.cpu().numpy()       
    u_np = all_u.cpu().numpy()       
    s_np = all_states.cpu().numpy()  

    y_meta = plot_metadata[2] if len(plot_metadata) > 2 else {}
    meta_labels = y_meta.get("labels", [])
    for i in range(input_dim):
        summary_signals = [y_np[:, j, i] for j in range(batch_size)] + [r_np[:, i]]
        base_y_label = meta_labels[0] if len(meta_labels) > 0 else f"y_{i+1}"
        base_r_label = meta_labels[1] if len(meta_labels) > 1 else f"r_{i+1}"
        plot_signals(t=time_axis, signals=summary_signals, labels=[f"Traj {j} ({base_y_label})" for j in range(batch_size)] + [f"Target ({base_r_label})"], title=f"Batch Convergence ({base_y_label}) - {batch_size} Trajectories Overview", xlabel="Time (h)", ylabel=y_meta.get("ylabel", "System Output"), dirname=dirname, filename=f"batch_summary_y_{i+1}")

    u_meta = plot_metadata[3] if len(plot_metadata) > 3 else {}
    for i in range(output_dim):
        label_base = u_meta.get("labels", [f"u_{i+1}"])[0] if i == 0 else f"u_{i+1}"
        title_base = u_meta.get("title", "Control Profile") if i == 0 else f"Control Input Profile (u_{i+1})"
        summary_inputs = [u_np[:, j, i] for j in range(batch_size)]
        plot_signals(t=time_axis, signals=summary_inputs, labels=[f"Traj {j} ({label_base})" for j in range(batch_size)], title=f"Batch Profile: {title_base} - Overlaid Actions", xlabel="Time (h)", ylabel=u_meta.get("ylabel", "Action Value"), dirname=dirname, filename=f"batch_summary_u_{i+1}")

    for i in range(s_np.shape[2]):
        x_meta = plot_metadata[i] if i < len(plot_metadata) else {}
        label_base = x_meta.get("labels", [f"x_{i+1}"])[0]
        title_base = x_meta.get("title", f"State x_{i+1}")
        summary_states = [s_np[:, j, i] for j in range(batch_size)]
        plot_signals(t=time_axis, signals=summary_states, labels=[f"Traj {j} ({label_base})" for j in range(batch_size)], title=f"Batch Trajectories: {title_base} Ensembles", xlabel="Time (h)", ylabel=x_meta.get("ylabel", "State Magnitude"), dirname=dirname, filename=f"batch_summary_x_{i+1}")

    tracking_metrics = {}
    for i in range(input_dim):
        y_traj = y_np[:, :, i]  
        r_traj = r_np[:, i]  
        metrics = compute_and_save_tracking_metrics(y_traj, r_traj, dt, dirname, suffix=f"y_{i+1}")
        tracking_metrics[f"y_{i+1}"] = metrics

    return {
        "trajectory_dataframes": trajectory_reports,
        "metrics": tracking_metrics,
        "simulated_outputs": y_np,
        "simulated_controls": all_u.cpu().numpy()
    }


def simulate_tracking_stateful_multi_model(
    models_dict: Dict[str, Dict[str, Any]],  # {"Model_Name": {"model": m, "x_scaler": x_s, "y_scaler": y_s}}
    plant: Any,
    r_trajectories: List[torch.Tensor],       # List of reference trajectories [steps] for each output dim
    hyperparam_config: Dict[str, Any],
    dirname: str,
    plot_individual_plots: bool = False
) -> Dict[str, Any]:
    """
    Simulates and compares multiple controlled MIMO plant inverse models over a 
    specified time horizon using stateful, step-by-step inference on identical initial conditions.
    """
    # Extract global configurations
    train_cfg = hyperparam_config["train"]
    training_data_cfg = hyperparam_config["training_data_cfg"]
    sim_cfg = hyperparam_config["simulate"]
    plant_cfg = hyperparam_config["plant"]

    steps = sim_cfg["seq_len"]
    dt = training_data_cfg["dt"]
    batch_size = sim_cfg["batch_size"]
    device = train_cfg["device"]
    input_dim = plant_cfg["input_dim"]    # Plant outputs (y1, y2, ...)
    output_dim = plant_cfg["output_dim"]  # Plant inputs / control actions (u1, u2, ...)

    n_y = train_cfg["n_y"]
    n_u = train_cfg["n_u"]
    warmup_steps = 10

    # Instantiate Plant
    plant_instance = plant(hyperparam_config) if isinstance(plant, type) else plant

    # Validate and prepare reference trajectory
    if len(r_trajectories) != input_dim:
        raise ValueError(f"Expected {input_dim} reference trajectories, got {len(r_trajectories)}")
    
    r_trajectory = torch.stack(r_trajectories, dim=1).to(device)  # [steps, input_dim]
    r_np = r_trajectory.cpu().numpy()

    # Capture canonical baseline initial state across ALL model runs for absolute fairness
    baseline_initial_state = plant_instance.get_initial_state(batch_size)
    sample_state = plant_instance.get_initial_state(1)
    state_dim = sample_state.shape[-1]

    # Data structures to store results for every model
    model_results = {}

    # =========================================================================
    # 1. RUN SIMULATION FOR EACH MODEL SEPARATELY
    # =========================================================================
    for model_name, model_meta in models_dict.items():
        print(f"\n🚀 Simulating Model: [{model_name}] ({batch_size} trajectories x {steps} steps)...")
        
        model = model_meta["model"]
        x_scaler = model_meta["x_scaler"]
        y_scaler = model_meta["y_scaler"]

        model.eval()
        
        # Agnostic state/memory reset call
        if hasattr(model, "reset_memory"):
            model.reset_memory(batch_size=batch_size, device=device)
        elif hasattr(model, "reset_hidden_states"):
            model.reset_hidden_states(batch_size=batch_size, device=device)

        if hasattr(model, "core") and hasattr(model.core, "return_bc"):
            model.core.return_bc = True

        # Copy identical initial plant state
        state = baseline_initial_state.clone() if torch.is_tensor(baseline_initial_state) else baseline_initial_state.copy()

        # Allocate storage buffers for this model
        all_y = torch.zeros((steps, batch_size, input_dim), device=device)
        all_u = torch.zeros((steps, batch_size, output_dim), device=device)
        all_states = torch.zeros((steps, batch_size, state_dim), device=device)

        # Seed signal history buffers
        initial_y = plant_instance.get_y(state, 0.0).cpu().numpy()
        y_histories = [[initial_y[b].copy()] for b in range(batch_size)]
        u_histories = [[np.zeros(output_dim)] for b in range(batch_size)]

        with torch.no_grad():
            for i in range(steps):
                t = i * dt
                y_current = plant_instance.get_y(state, t)
                y_curr_np = y_current.cpu().numpy()

                # Update output window history
                for b in range(batch_size):
                    y_histories[b].append(y_curr_np[b])
                    if len(y_histories[b]) > (n_y + 1):
                        y_histories[b].pop(0)

                # Look-ahead target reference
                next_idx = min(i + 1, steps - 1)
                target_r = r_trajectory[next_idx].expand(batch_size, input_dim)
                tgt_r_np = target_r.cpu().numpy()

                if i < warmup_steps:
                    # Warmup Phase
                    u_unscaled = 0.5 * np.ones((batch_size, output_dim))
                    u = torch.tensor(u_unscaled, dtype=torch.float32, device=device)
                else:
                    # Model Control Phase
                    v_k_batch_raw = []
                    for b in range(batch_size):
                        y_window = np.array(y_histories[b])
                        y_hist_reversed = y_window[::-1].flatten()

                        u_window = np.array(u_histories[b]) if n_u > 0 else np.array([])
                        u_hist_reversed = u_window[::-1].flatten() if n_u > 0 else np.array([])

                        v_k_single = np.concatenate([tgt_r_np[b], y_hist_reversed, u_hist_reversed])
                        v_k_batch_raw.append(v_k_single)

                    v_k_batch_raw = np.array(v_k_batch_raw)
                    v_k_scaled = x_scaler.transform(v_k_batch_raw)
                    v_k_tensor = torch.tensor(v_k_scaled, dtype=torch.float32, device=device)

                    # Model Step (Works for Mamba, LSTMs, Transformers, MLPs)
                    if hasattr(model, "step"):
                        u_norm_tensor = model.step(v_k_tensor)
                    else:
                        u_norm_tensor = model(v_k_tensor)

                    u_norm_np = u_norm_tensor.cpu().numpy()
                    u_unscaled = y_scaler.inverse_transform(u_norm_np)

                    # Clip to physical limits
                    u_unscaled = np.clip(u_unscaled, plant_cfg["u_1_hard_min"], plant_cfg["u_1_hard_max"])
                    u = torch.tensor(u_unscaled, dtype=torch.float32, device=device)

                # Update control action history window
                for b in range(batch_size):
                    u_histories[b].append(u_unscaled[b])
                    if len(u_histories[b]) > n_u:
                        u_histories[b].pop(0)

                # Step physical plant
                if output_dim == 1:
                    state, _ = plant_instance.step(state=state, u=u[:, 0:1], t=t, dt=dt)
                else:
                    try:
                        state, _ = plant_instance.step(state=state, u=u, t=t, dt=dt)
                    except TypeError:
                        kwargs = {f"u{j+1}": u[:, j:j+1] for j in range(output_dim)}
                        state, _ = plant_instance.step(state=state, t=t, dt=dt, **kwargs)

                # Store sequence history
                all_y[i] = y_current
                all_u[i] = u
                all_states[i] = state

        # Save model trajectory results
        model_results[model_name] = {
            "y": all_y.cpu().numpy(),       # [steps, batch_size, input_dim]
            "u": all_u.cpu().numpy(),       # [steps, batch_size, output_dim]
            "states": all_states.cpu().numpy()  # [steps, batch_size, state_dim]
        }

    # =========================================================================
    # 2. OVERLAYED PLOTTING ACROSS MODELS
    # =========================================================================
    time_axis = np.arange(steps) * dt
    plot_metadata = plant_instance.get_plot_config() if hasattr(plant_instance, "get_plot_config") else []

    output_meta = next((c for c in plot_metadata if any(col.startswith("y") for col in c["cols"])), {})
    control_meta = next((c for c in plot_metadata if any(col.startswith("u") for col in c["cols"])), {})
    state_meta = next((c for c in plot_metadata if any(col.startswith("x") for col in c["cols"])), {})

    # --- 2A. Stacked Output Tracking Comparison Plot ---
    signals_to_plot, labels_to_plot, ylabels_to_plot = [], [], []

    for i in range(input_dim):
        row_signals, row_labels = [], []
        
        # Add Reference Curve
        row_signals.append(r_np[:, i])
        row_labels.append("Reference Target")

        # Add mean response trajectory for each model
        for model_name, res in model_results.items():
            mean_y_traj = res["y"][:, :, i].mean(axis=1)  # Average over batch
            row_signals.append(mean_y_traj)
            row_labels.append(f"{model_name}")

        signals_to_plot.append(row_signals)
        labels_to_plot.append(row_labels)
        
        y_labels_list = output_meta.get("ylabel", [])
        row_ylabel = y_labels_list[i] if isinstance(y_labels_list, list) and i < len(y_labels_list) else f"Output {i+1}"
        ylabels_to_plot.append(row_ylabel)

    plot_stacked(
        t=time_axis,
        signals=signals_to_plot,
        labels=labels_to_plot,
        xlabel=rf"$t \; / \; \mathrm{{s}}$",
        ylabel=ylabels_to_plot,
        asp=[0.33] * len(signals_to_plot),
        dirname=dirname,
        filename="models_comparison_outputs_stacked.png",
        show=True
    )

    # --- 2B. Stacked Control Actions Comparison Plot ---
    signals_to_plot, labels_to_plot, ylabels_to_plot = [], [], []

    for i in range(output_dim):
        row_signals, row_labels = [], []

        for model_name, res in model_results.items():
            mean_u_traj = res["u"][:, :, i].mean(axis=1)
            row_signals.append(mean_u_traj)
            row_labels.append(f"{model_name}")

        signals_to_plot.append(row_signals)
        labels_to_plot.append(row_labels)

        y_labels_list = control_meta.get("ylabel", [])
        row_ylabel = y_labels_list[i] if isinstance(y_labels_list, list) and i < len(y_labels_list) else f"Action {i+1}"
        ylabels_to_plot.append(row_ylabel)

    plot_stacked(
        t=time_axis,
        signals=signals_to_plot,
        labels=labels_to_plot,
        xlabel=rf"$t \; / \; \mathrm{{s}}$",
        ylabel=ylabels_to_plot,
        asp=[0.33] * len(signals_to_plot),
        dirname=dirname,
        filename="models_comparison_controls_stacked.png",
        show=True
    )

    # =========================================================================
    # 3. METRICS COMPILATION FOR ALL MODELS
    # =========================================================================
    comparison_metrics = {}
    for model_name, res in model_results.items():
        comparison_metrics[model_name] = {}
        for i in range(input_dim):
            y_traj = res["y"][:, :, i]
            r_traj = r_np[:, i]
            metrics = compute_and_save_tracking_metrics(
                y_traj, r_traj, dt, dirname=os.path.join(dirname, model_name), suffix=f"y_{i+1}"
            )
            comparison_metrics[model_name][f"y_{i+1}"] = metrics

    return {
        "model_results": model_results,
        "metrics": comparison_metrics
    }

def simulate_tracking_stateful_external_ref_trajectory(
    model,
    plant,
    r_trajectories,  # List of reference trajectories, one for each output dimension. Shape: [steps] for each trajectory.
    hyperparam_config,
    x_scaler,
    y_scaler,
    dirname,
    plot_individual_plots=False
):
    """
    Simulates a controlled MIMO plant over a specified time horizon using stateful,
    step-by-step inference (carrying forward Mamba hidden states) while tracking a
    separate reference trajectory for each output dimension.
    
    Aligned with the robust plant instantiation and plotting routines of the validation rollout.
    """
    # Extract configuration sub-dictionaries
    train_cfg = hyperparam_config["train"]
    training_data_cfg = hyperparam_config["training_data_cfg"]
    sim_cfg = hyperparam_config["simulate"]
    plant_cfg = hyperparam_config["plant"]

    model.core.return_bc = True  # Enable returning B and C
    
    # Unpack specific parameters
    steps = sim_cfg["seq_len"]
    dt = training_data_cfg["dt"]
    batch_size = sim_cfg["batch_size"]
    device = train_cfg["device"]
    input_dim = plant_cfg["input_dim"]    # Number of plant outputs (y1, y2, ...)
    output_dim = plant_cfg["output_dim"]  # Number of control inputs (u1, u2, ...)

    # 🌟 1. HANDLE PLANT INSTANTIATION (Class vs. Instance)
    if isinstance(plant, type):
        plant_instance = plant(hyperparam_config)
    else:
        plant_instance = plant

    # Validate r_trajectories
    if len(r_trajectories) != input_dim:
        raise ValueError(f"Expected {input_dim} reference trajectories, got {len(r_trajectories)}")

    # Convert r_trajectories to a tensor of shape [steps, input_dim]
    r_trajectory = torch.stack(r_trajectories, dim=1)  # Shape: [steps, input_dim]
    r_np = r_trajectory.cpu().numpy()  # Shape: [steps, input_dim]

    # Initialize GPU tensor buffers
    all_y = torch.zeros((steps, batch_size, input_dim), device=device)  
    all_u = torch.zeros((steps, batch_size, output_dim), device=device)  
    
    # Dynamically query plant state dimensions to avoid hardcoding
    sample_state = plant_instance.get_initial_state(1)
    state_dim = sample_state.shape[-1]
    all_states = torch.zeros((steps, batch_size, state_dim), device=device)  

    state = plant_instance.get_initial_state(batch_size)

    # Prepare model for evaluation mode
    model.eval()
    ssm_history = {
        "step": [], "time": [],
        "A_bar": [], "B_bar": [], "C": [], "dt": []
    }
    print(f"📈 Testing Stateful MIMO Trajectory Tracking: {batch_size} trajectories across {steps} steps...")

    # CORRECT MEMORY INITIALIZATION
    model.reset_memory(batch_size=batch_size, device=device)

    # INITIALIZE SLIDING WINDOW RUNNING BUFFERS FOR THE BATCH
    n_y = train_cfg["n_y"]
    n_u = train_cfg["n_u"]

    # 🌟 Calculate the physical lookback threshold
    # Since we need (n_y + 1) past outputs and n_u past controls:
    warmup_steps = 10 # max(n_y + 1, n_u)

    # Seed history buffers with only the very first step instead of dummy-repeating them
    initial_y = plant_instance.get_y(state, 0.0).cpu().numpy()  # [batch_size, input_dim]
    y_histories = [[initial_y[b].copy()] for b in range(batch_size)]
    u_histories = [[np.zeros(output_dim)] for b in range(batch_size)]

    # Execute forward tracking simulation
    with torch.no_grad():
        for i in range(steps):
            t = i * dt
            y_current = plant_instance.get_y(state, t)  # Shape: [batch_size, input_dim]
            y_curr_np = y_current.cpu().numpy()

            # 1. Update running history with the newly observed plant output
            for b in range(batch_size):
                y_histories[b].append(y_curr_np[b])
                if len(y_histories[b]) > (n_y + 1):
                    y_histories[b].pop(0)

            # Look-ahead: Target reference state for the NEXT time-step (i+1)
            next_idx = min(i + 1, steps - 1)
            target_r = r_trajectory[next_idx].expand(batch_size, input_dim)
            tgt_r_np = target_r.cpu().numpy()

            # 🌟 Determine if we have enough physical history to start model control
            if i < warmup_steps:
                # --- WARMUP PHASE ---
                # Model does not act yet. Use safe default control actions (zeros)
                u_unscaled = 0.5* np.ones((batch_size, output_dim))
                u = torch.tensor(u_unscaled, dtype=torch.float32, device=device)
                
            else:
                # --- ACTIVE CONTROL PHASE ---
                # We construct the input vector v_k using only genuine accumulated histories
                v_k_batch_raw = []
                for b in range(batch_size):
                    y_window = np.array(y_histories[b])
                    y_hist_reversed = y_window[::-1].flatten()

                    u_window = np.array(u_histories[b]) if n_u > 0 else np.array([])
                    u_hist_reversed = u_window[::-1].flatten() if n_u > 0 else np.array([])

                    # Combine: [Target_Future, Past_Outputs, Past_Controls]
                    v_k_single = np.concatenate([tgt_r_np[b], y_hist_reversed, u_hist_reversed])
                    v_k_batch_raw.append(v_k_single)

                # Convert, scale, and infer
                v_k_batch_raw = np.array(v_k_batch_raw)
                v_k_scaled = x_scaler.transform(v_k_batch_raw)
                v_k_tensor = torch.tensor(v_k_scaled, dtype=torch.float32, device=device)

                u_norm_tensor = model.step(v_k_tensor)
                
                u_norm_np = u_norm_tensor.cpu().numpy() 
                u_unscaled = y_scaler.inverse_transform(u_norm_np)

                # Force physical actuator limits
                u_unscaled = np.clip(u_unscaled, plant_cfg["u_1_hard_min"], plant_cfg["u_1_hard_max"])  
                u = torch.tensor(u_unscaled, dtype=torch.float32, device=device)  

            # 2. Update control history with the chosen action
            for b in range(batch_size):
                u_histories[b].append(u_unscaled[b])
                if len(u_histories[b]) > n_u:
                    u_histories[b].pop(0)

            # Step the physical plant forward
            if output_dim == 1:
                state, _ = plant_instance.step(state=state, u=u[:, 0:1], t=t, dt=dt)
            else:
                try:
                    state, _ = plant_instance.step(state=state, u=u, t=t, dt=dt)
                except TypeError:
                    kwargs = {f"u{j+1}": u[:, j:j+1] for j in range(output_dim)}
                    state, _ = plant_instance.step(state=state, t=t, dt=dt, **kwargs)

            # Logging metrics
            all_y[i] = y_current  
            all_u[i] = u  
            all_states[i] = state 

    # --- PLOTTING & EXPORT CONFIGURATION ---
    time_axis = np.arange(steps) * dt
    trajectory_reports = []
    total_stacked_blocks = input_dim + output_dim
    
    # Retrieve the metadata configuration blocks from the plant safely
    plot_metadata = plant_instance.get_plot_config() if hasattr(plant_instance, "get_plot_config") else []

    # Safe lookup logic mirrored from validation sequence
    state_meta = next((c for c in plot_metadata if any(col.startswith("x") for col in c["cols"])), {})
    output_meta = next((c for c in plot_metadata if any(col.startswith("y") for col in c["cols"])), {})
    control_meta = next((c for c in plot_metadata if any(col.startswith("u") for col in c["cols"])), {})
    
    save_to_json(
        data=ssm_history,
        dirname=dirname,          
        filename="ssm_matrices_history"
    )
    
    # Parse and save individual trajectory records
    for b in range(batch_size):
        state_dirname = os.path.join(dirname, f"initial_state_{b}")
        os.makedirs(state_dirname, exist_ok=True)

        y_traj = all_y[:, b, :].cpu().numpy()            # Shape: [steps, input_dim]
        u_traj = all_u[:, b, :].cpu().numpy()            # Shape: [steps, output_dim]
        states_traj = all_states[:, b, :].cpu().numpy()  # Shape: [steps, state_dim]

        # Save DataFrame for this trajectory
        df_data = {
            "time": np.tile(time_axis, total_stacked_blocks),
            "signal_type": np.repeat(
                [f"y_{i+1}" for i in range(input_dim)] + [f"u_{i+1}" for i in range(output_dim)],
                steps
            ),
            "value": np.concatenate([
                y_traj[:, i] for i in range(input_dim)
            ] + [
                u_traj[:, i] for i in range(output_dim)
            ]),
        }
        
        for i in range(states_traj.shape[1]):
            df_data[f"state_{i+1}"] = np.tile(states_traj[:, i], total_stacked_blocks)

        df_traj = pd.DataFrame(df_data)
        save_df_to_csv(df_traj, dirname=state_dirname, filename="state_report")
        trajectory_reports.append(df_traj)

        if plot_individual_plots:
            # 1. Individual Plot: Control Signals
            for i in range(output_dim):
                labels_list = control_meta.get("labels", [])
                label = labels_list[i] if i < len(labels_list) else f"Control Input (u_{i+1})"
                title = control_meta.get("title", "Control Profile")
                
                plot_signals(
                    t=time_axis,
                    signals=[u_traj[:, i]],
                    labels=[label],
                    title=f"Trajectory {b}: {title}",
                    xlabel=rf"$t \; / \; \mathrm{{s}}$",
                    ylabel=control_meta.get("ylabel", "Action Value"),
                    dirname=state_dirname,
                    filename=f"plot_control_signal_u_{i+1}"
                )

            # 2. Individual Plot: Output tracking performance
            for i in range(input_dim):
                labels_list = output_meta.get("labels", [])
                ind_y_label = labels_list[0] if len(labels_list) > 0 else f"Output (y_{i+1})"
                ind_r_label = labels_list[1] if len(labels_list) > 1 else f"Target (r_{i+1})"
                title = output_meta.get("title", "Tracking Performance")
                
                plot_signals(
                    t=time_axis,
                    signals=[y_traj[:, i], r_np[:, i]],  
                    labels=[ind_y_label, ind_r_label],   
                    title=f"Trajectory {b}: {title}",
                    xlabel=rf"$t \; / \; \mathrm{{s}}$",
                    ylabel=output_meta.get("ylabel", "Signal Value"),
                    dirname=state_dirname,
                    filename=f"plot_output_tracking_y_{i+1}"
                )
                
            # 3. Individual Plot: Internal plant states
            for i in range(states_traj.shape[1]):
                labels_list = state_meta.get("labels", [])
                label = labels_list[i] if i < len(labels_list) else f"State x_{i+1}"
                title = state_meta.get("title", "Internal Plant States")
                
                plot_signals(
                    t=time_axis,
                    signals=[states_traj[:, i]],
                    labels=[label],
                    title=f"Trajectory {b}: {title} - {label}",
                    xlabel=rf"$t \; / \; \mathrm{{s}}$",
                    ylabel=state_meta.get("ylabel", "State Magnitude"),
                    dirname=state_dirname,
                    filename=f"plot_plant_state_x_{i+1}"
                )

    # =========================================================================
    # UNIFIED BATCH OVERLAY STACKED PLOT (Outputs + Controls + States)
    # =========================================================================
    y_np = all_y.cpu().numpy()       # Shape: [steps, batch_size, input_dim]
    u_np = all_u.cpu().numpy()       # Shape: [steps, batch_size, output_dim]
    s_np = all_states.cpu().numpy()  # Shape: [steps, batch_size, state_dim]

    # Helper function to query label/ylabel safely from plot_metadata
    def get_meta_info(prefix, index, default_label, default_ylabel):
        for block in plot_metadata:
            cols = block.get("cols", [])
            # Match block by column naming standard (e.g. 'y', 'y_1', 'x_2')
            if any(col == prefix or col.startswith(f"{prefix}_") or col.startswith(f"{prefix}") for col in cols):
                labels = block.get("labels", [])
                ylabel = block.get("ylabel", default_ylabel)
                
                # Fetch output label
                label_val = labels[index] if index < len(labels) else default_label
                
                # Handle cases where 'ylabel' is either a list or a string
                if isinstance(ylabel, list):
                    ylabel_val = ylabel[index] if index < len(ylabel) else default_ylabel
                else:
                    ylabel_val = ylabel if index == 0 else default_ylabel
                
                return label_val, ylabel_val
        
        return default_label, default_ylabel

    # Helper function to extract x-axis label safely
    time_meta = next((c for c in plot_metadata if "t" in c.get("cols", [])), {})
    time_xlabel_list = time_meta.get("xlabel", [r"$t \; / \; \mathrm{s}$"])
    xlabel_str = time_xlabel_list[0] if isinstance(time_xlabel_list, list) and time_xlabel_list else time_meta.get("xlabel", r"$t \; / \; \mathrm{s}$")

    signals_to_plot = []
    labels_to_plot = []
    ylabels_to_plot = []

    # -------------------------------------------------------------------------
    # 1. Stack System Outputs (y)
    # -------------------------------------------------------------------------
    for i in range(input_dim):
        ref_label, y_label = get_meta_info("y", i, f"Output y_{i+1}", f"$y_{i+1}$")
        
        row_signals = [y_np[:, j, i] for j in range(batch_size)] + [r_np[:, i]]
        row_labels = [f"{batch_size} Simulated Curves" if j == 0 else "" for j in range(batch_size)] + [f"Reference ({ref_label})"]
        
        signals_to_plot.append(row_signals)
        labels_to_plot.append(row_labels)
        ylabels_to_plot.append(y_label)

    # -------------------------------------------------------------------------
    # 2. Stack Control Actions (u)
    # -------------------------------------------------------------------------
    for i in range(output_dim):
        act_label, u_label = get_meta_info("u", i, f"Control u_{i+1}", f"$u_{i+1}$")
        
        row_signals = [u_np[:, j, i] for j in range(batch_size)]
        row_labels = [f"{batch_size} Simulated Curves" if j == 0 else "" for j in range(batch_size)]
        
        signals_to_plot.append(row_signals)
        labels_to_plot.append(row_labels)
        ylabels_to_plot.append(u_label)

    # -------------------------------------------------------------------------
    # 3. Stack Internal Plant States (x)
    # -------------------------------------------------------------------------
    for i in range(s_np.shape[2]):
        st_label, x_label = get_meta_info("x", i, f"State x_{i+1}", f"$x_{i+1}$")
        
        row_signals = [s_np[:, j, i] for j in range(batch_size)]
        row_labels = [f"{batch_size} Simulated Curves" if j == 0 else "" for j in range(batch_size)]
        
        signals_to_plot.append(row_signals)
        labels_to_plot.append(row_labels)
        ylabels_to_plot.append(x_label)

    # -------------------------------------------------------------------------
    # 4. Render Unified Stacked Figure
    # -------------------------------------------------------------------------
    plot_stacked(
        t=time_axis,
        signals=signals_to_plot,
        labels=labels_to_plot,
        xlabel=xlabel_str,
        ylabel=ylabels_to_plot,
        asp=[0.25] * len(signals_to_plot),  # Scaled aspect ratio for larger stack
        dirname=dirname,
        filename="batch_summary_system_trajectory_stacked.png",
        show=True
    )

    # --- METRICS COMPILATION ---
    tracking_metrics = {}
    for i in range(input_dim):
        y_traj = y_np[:, :, i]  # Shape: [steps, batch_size]
        r_traj = r_np[:, i]     # Shape: [steps]
        metrics = compute_and_save_tracking_metrics(y_traj, r_traj, dt, dirname, suffix=f"y_{i+1}")
        tracking_metrics[f"y_{i+1}"] = metrics

    return {
        "trajectory_dataframes": trajectory_reports,
        "metrics": tracking_metrics,
        "simulated_outputs": y_np,
        "simulated_controls": all_u.cpu().numpy()
    }



def generate_smooth_profile_trajectory(time_axis, config):
    """
    Generates a reference trajectory that starts flat, rises via a sigmoid,
    stays constant at a peak plateau, sinks via a sigmoid, and stays flat at the floor.
    
    Args:
        time_axis (torch.Tensor): 1D tensor representing the simulation time timeline.
        config (dict): Parameters containing:
            - 'y_floor': Baseline value (e.g., initial concentration)
            - 'y_peak': Upper plateau value
            - 't_rise_center': Time at the midpoint of the rising ramp
            - 'k_rise': Slope/steepness coefficient for the rise phase
            - 't_sink_center': Time at the midpoint of the sinking ramp
            - 'k_sink': Slope/steepness coefficient for the sinking phase
    Returns:
        r_t (torch.Tensor): Smooth trajectory evaluated at each timestep.
    """
    y_floor = config['y_floor']
    y_peak = config['y_peak']
    delta_y = y_peak - y_floor
    
    # 📈 Rising Sigmoid Component
    # Centers the transition around t_rise_center
    sigma_rise = 1.0 / (1.0 + torch.exp(-config['k_rise'] * (time_axis - config['t_rise_center'])))
    
    # 📉 Sinking Sigmoid Component
    # Centers the transition around t_sink_center
    sigma_sink = 1.0 / (1.0 + torch.exp(-config['k_sink'] * (time_axis - config['t_sink_center'])))
    
    # Combined smooth blending expression
    r_t = y_floor + (delta_y * sigma_rise) - (delta_y * sigma_sink)
    
    return r_t


def generate_exponential_decay_trajectory(steps, dt, y_start, y_target, tau, device="cpu"):
    """
    Generates a smooth reference trajectory that starts at y_start and 
    exponentially transitions down to y_target governed by time constant tau.
    """
    # Create the time axis
    t_axis = torch.arange(steps, device=device, dtype=torch.float32) * dt
    
    # Compute the exponential curve
    r = y_target + (y_start - y_target) * torch.exp(-t_axis / tau)
    
    # Return with shape [steps, 1] to keep consistency with your other generators
    return r.unsqueeze(-1)

#=== FUNCTION TO GENERATE REFERENCE TRAJECTORY ===#
def generate_reference_trajectory(steps, dt, device, mode="constant", constant_val=0.3, gain=1.0, period=20.0):
    """
    Generates reference target trajectories for physical control system tracking simulations.

    Parameters:
    - steps (int): Total sequence length of the reference trajectory.
    - dt (float): Sampling time increment between sequential steps.
    - device (str/torch.device): Target execution hardware device mapping for the output tensor.
    - mode (str): Style of reference generated; choices are "constant" or "dynamic" (default: "constant").
    - constant_val (float): Fixed tracking setpoint used if mode is "constant" (default: 0.3).
    - gain (float): Tuning scaling modifier to sharpen tracking switches if mode is "dynamic" (default: 1.0).
    - period (float): Cyclical hour sequence timeframe for oscillations if mode is "dynamic" (default: 20.0).

    Returns:
    - r_trajectory (torch.Tensor): Reference target values mapped to the execution device, shaped [steps, 1].

    The function acts as an isolated generator for processing setpoints. Depending on the requested 
    mode, it outputs either a uniform static tensor filled with a base setpoint value or computes 
    a dynamic, bounded time-varying profile utilizing transcendental mathematical operators to yield 
    smooth rectangular transitions.
    """
    if mode == "constant":
        # Create a tensor of shape [steps, 1] filled with a static target value
        r_trajectory = torch.full((steps, 1), constant_val, device=device, dtype=torch.float32)
    
    elif mode == "dynamic":
        # Generate time steps array
        time_axis = np.arange(steps) * dt
        
        # Calculate a smooth time-varying waveform base
        sine_base = np.sin(2 * np.pi * time_axis / period)
        
        # Use tanh to sharpen the curve into a "soft" rectangle with an upward linear drift

        noise = 0 #np.random.uniform(-0.005, 0.005, size=time_axis.shape)

        #r_trajectory_np = 0.25 + 0.04 * np.tanh(gain * sine_base) - 0.00 * time_axis   # Add small random noise for realism #chemostat

        r_trajectory_np = 0.02 + 0.005 * np.tanh(gain * sine_base) - 0.00 * time_axis +noise  # Add small random noise for realism
        
        # Convert the structural numpy baseline into a target PyTorch tensor array
        r_trajectory = torch.tensor(r_trajectory_np, device=device, dtype=torch.float32).unsqueeze(1)
    
    else:
        raise ValueError(f"Unknown reference mode selection: '{mode}'. Choose 'constant' or 'dynamic'.")

    return r_trajectory