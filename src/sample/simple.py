import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os
import numpy as np
import torch
from tslearn.clustering import TimeSeriesKMeans
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
from src.sample.utils.plotting_utils import plot_signals

plt.style.use("src/sample/style.mplstyle")

dataset_path = "results/2026-07-03/2026-07-03_17-01-29/TrophophasePlant_training_data/dataset/2026-07-03_17-01-29_training_data.pt"

# 1. Load the dictionary (Safe loading)
data_dict = torch.load(dataset_path, weights_only=True)

# 2. Extract and push ONLY the tensor to GPU if needed for calculations
y_tensor = data_dict['y'][:10].to("cuda")

print("\n=== Tensor Details ===")
print(f"Data type: {y_tensor.dtype}")
print(f"Tensor Shape: {y_tensor.shape}")
print(f"Min value: {y_tensor.min()} | Max value: {y_tensor.max()}")

# Print the first 3 rows/sequences safely
#print("\nFirst 3 entries:")
#print(y_tensor[:3].cpu()) # Bring back to CPU just to print cleanly

# 3. Scale your time series!
scaler = TimeSeriesScalerMeanVariance()
# Slicing the first 20 sequences and pulling to numpy
X_scaled = scaler.fit_transform(y_tensor.cpu().numpy())  

# 4. Initialize TimeSeriesKMeans with Soft-DTW
km = TimeSeriesKMeans(
    n_clusters=3, 
    metric="softdtw", 
    metric_params={"gamma": 0.1}, # Added gamma definition for softdtw smoothing
    max_iter=10, 
    n_jobs=-1, 
    random_state=42
)

# # 5. Fit the model and get cluster assignments
# cluster_labels = km.fit_predict(X_scaled)

# print("\nCluster assignments for each sequence:", cluster_labels)

# =====================================================================
# PREPARING INPUTS FOR YOUR CUSTOM PLOT_SIGNALS FUNCTION
# =====================================================================

# 1. Reconstruct the true time timeline axis (t)
dt_original = 0.01
downsample_factor = 1
effective_dt = dt_original * downsample_factor
num_timesteps = X_scaled.shape[1]

# This creates the 1D horizontal x-axis array your function expects
time_axis = np.arange(num_timesteps) * effective_dt

# 2. Extract signals into a list of 1D numpy arrays
# Your function loops through 'signals', so we split X_scaled into a list of sequences
signals_list = [X_scaled[i].ravel() for i in range(len(X_scaled))]

# # 3. Generate labels string tracking cluster IDs for the legend
# labels_list = [f"Sequence {i} (Cluster {cluster_labels[i]})" for i in range(len(X_scaled))]

# 4. Define your export path parameters
output_dir = os.path.dirname(dataset_path).replace("dataset", "dataset_plots")
os.makedirs(output_dir, exist_ok=True)
save_file_path = os.path.join(output_dir, "soft_dtw_clustering_results.png")

# =====================================================================
# EXECUTE YOUR PLOT_SIGNALS FUNCTION
# =====================================================================
print("\nCompiling visual canvas using plot_signals()...")

img_asset = plot_signals(
    t=time_axis,
    signals=signals_list,
    #labels=labels_list,
    title="MIMO Trajectories Grouped by Soft-DTW Shape Profile",
    xlabel="Time [s]",
    ylabel="Scaled Signal Value (Normalized)",
    figsize=(8, 8),  # Square canvas plays well with your fixed aspect ratio logic
    save_path=save_file_path,
    show=True,
    filename="soft_dtw_clustering_raster",
    dirname=output_dir
)