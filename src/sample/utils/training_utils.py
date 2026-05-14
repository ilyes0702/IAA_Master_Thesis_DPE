# Import standard libraries
import numpy as np
import pandas as pd

# import machine learning modules
from src.sample.utils.loss_utils import *
from src.sample.decorators.general_decorators import *
from src.sample.utils.saving_utils import *
from src.sample.config import *
from src.sample.utils.plotting_utils import plot_signals
import torch
plt.style.use("src/sample/style.mplstyle")

@track_resources
def GPUtrain_controller_from_disk(model, dataset_path, hyperparam_config, dirname="name_directory", num_sequences_to_use=None):
    # --- EXTRACT HYPERPARAMETERS ---
    train_cfg = hyperparam_config["train"]
    epochs = train_cfg["epochs"]
    device = train_cfg["device"]
    lr = train_cfg["lr"]
    dt = hyperparam_config["signal"]["dt"]

    # --- INITIALIZE ---
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    #criterion = relative_huber_loss
    criterion = hybrid_control_loss
    sequence_loss_history = []
    
    print(f"📂 Loading dataset from {dataset_path}...")
    dataset = torch.load(dataset_path, weights_only=True) 
    batches_x = dataset["x"]
    batches_y = dataset["y"]

    # --- DATA SLICING LOGIC ---
    # Concatenate all batches into one large tensor: [Total_Seqs, Seq_Len, Features]
    full_x = torch.cat(dataset["x"], dim=0)
    full_y = torch.cat(dataset["y"], dim=0)
    # If num_sequences_to_use is provided, slice the data
    if num_sequences_to_use is not None:
        full_x = full_x[:num_sequences_to_use]
        full_y = full_y[:num_sequences_to_use]
        print(f"✂️ Sliced dataset to the first {num_sequences_to_use} sequences.")
    else:
        print(f"✅ Using full dataset ({full_x.shape[0]} sequences).")

    model.to(device)

    for epoch in range(epochs):
        for b_idx, (x_batch, y_batch) in enumerate(zip(batches_x, batches_y)):
            # Move batch to GPU
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            batch_size = x_batch.shape[0]

            model.train()

            for s_idx in range(full_x.shape[0]):
                optimizer.zero_grad()
                
                u_train_single = full_y[s_idx:s_idx+1, :, :].to(device)
                x_input_single = full_x[s_idx:s_idx+1, :, :].to(device)

                # 1. Prediction (Forward Pass)
                u_pred_single = model(x_batch[s_idx:s_idx+1, :, :])
                
                # 2. Optimization (Backward Pass)
                loss = criterion(u_pred_single, u_train_single)
                loss.backward()
                optimizer.step()
                
                sequence_loss_history.append(loss.item())

                # --- NEW: SAVE PREDICTIONS CSV ---
                

                # --- DATA PREPARATION for CSV and PLOT ---
                # detach() removes from graph, cpu() moves to RAM, numpy() converts, flatten() makes it 1D
                u_p_np = u_pred_single[0].detach().cpu().numpy().flatten()
                u_t_np = u_train_single[0].detach().cpu().numpy().flatten()
                t_axis = np.arange(len(u_p_np)) * dt

                # --- SAVE PREDICTIONS CSV ---
                comparison_df = pd.DataFrame({
                    "t": t_axis,
                    "u_train": u_t_np,
                    "u_pred": u_p_np,
                    "abs_error": np.abs(u_t_np - u_p_np)
                })

                # We use your custom save_df_to_csv function
                # This will land in: results/DATE/TIME/dirname/predictions/reports/
                csv_filename = f"epoch_{epoch}_batch_{b_idx}_seq_{s_idx}_preds"
                save_df_to_csv(comparison_df, 
                               dirname=f"{dirname}/predictions", 
                               filename=csv_filename)
                

                # plot_signals(
                #         t_axis, 
                #         [u_t_np, u_p_np], # Pass the flattened numpy arrays
                #         labels=["Ground Truth (u)", "Mamba Prediction (u_hat)"],
                #         xlabel="Time", ylabel="Control Signal",
                #         title=f"Mamba Prediction Accuracy Epoch {epoch} Seq {s_idx}",
                #         dirname=dirname+"/sequences",
                #         filename=f"prediction_accuracy_epoch_{epoch}_batch_{b_idx}_seq_{s_idx}"
                #     )

                # plot_signals(
                #         t_axis, 
                #         [u_t_np, u_p_np], # Pass the flattened numpy arrays
                #         labels=["Ground Truth (u)", "Mamba Prediction (u_hat)"],
                #         xlabel="Time", ylabel="Control Signal",
                #         title=f"Mamba Prediction Accuracy Epoch {epoch} Seq {s_idx}",
                #         dirname=dirname+"/sequences",
                #         filename=f"prediction_accuracy"
                #     )

        print(f"🚀 Epoch {epoch+1}/{epochs} Finished | Final Seq Loss: {loss.item():.6f}")

    # Final model and loss plots
    save_model(model, dirname=dirname, hyperparam_config=hyperparam_config, filename="trained_controller_disk")
    
    # Save master loss history
    loss_df = pd.DataFrame({"sequence_index": range(len(sequence_loss_history)), "loss": sequence_loss_history})
    save_df_to_csv(loss_df, dirname=dirname, filename="total_sequence_loss")

    plot_signals(
        loss_df["sequence_index"].values, [loss_df["loss"].values],
        labels=["Hybrid Control Loss"], 
        xlabel="Total Sequences Trained", 
        ylabel="Loss",
        title="Learning Curve (Per Sequence) - From Disk",
        dirname=dirname, filename="sequence_loss_from_disk_plot"
    )

    return sequence_loss_history



