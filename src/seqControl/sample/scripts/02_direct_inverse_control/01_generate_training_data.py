"""Script to generate and save training data for plant models.

This script selects a plant model, generates a dataset using
generate_and_save_dataset and writes it to a folder named
<PlantClassName>_training_data.

Usage: run this script from the project root (or ensure PYTHONPATH
includes the project) so imports resolve correctly.
"""
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from tensorboard import data
from tslearn.metrics import cdist_dtw
from tslearn.barycenters import dtw_barycenter_averaging
from tslearn.preprocessing import TimeSeriesScalerMeanVariance

from seqControl.sample.utils.saving_and_loading_utils import *
from seqControl.sample.utils.data_generation_utils import *
from seqControl.sample.classes.plants.ChemostatPlant import *
from seqControl.sample.classes.plants.IdiophasePlant import *
from seqControl.sample.classes.plants.MassSpringDamperPlant import *
from seqControl.sample.classes.plants.BajpaiReussPlant import *
from seqControl.sample.classes.plants.TrophophasePlant import *
from seqControl.sample.classes.plants.PenicillinPlantBirol2002 import *
from seqControl.sample.classes.plants.CocultivationPlant import *
from seqControl.sample.classes.plants.IndForProteinProductionPlant import *

from seqControl.hyperparam_config import *
import matplotlib.pyplot as plt
from seqControl.sample.utils.plotting_utils import *


# Apply project style for plots (if plotting is enabled)
plt.style.use("src/sample/style.mplstyle")


def main() -> None:
	"""Create a plant instance and generate training data.

	Change which plant is instantiated by uncommenting the other
	option below. The dataset folder is named after the plant class.
	"""

	# CHOOSE PLANT AND HYPERPARAMETER CONFIGURATION
	plant = TrophophasePlant(hyperparam_config=hyperparam_config_TrophophasePlant)
	hyperparam_config = plant.hyperparam_config

	dirname1 = plant.__class__.__name__ + "_training_data" +"_run_1"
	dirname2 = plant.__class__.__name__ + "_training_data" +"_run_2"
	save_to_json(hyperparam_config, dirname1, "training_data_hyperparam_config")

	# cfg1 = {
	# 		"u_1_D_center_min": 0.6,
	# 		"u_1_D_center_max": 0.9,
	
	# 		"u_1_hard_min": 0.0,
	# 		"u_1_hard_max": 1,
	
	# 		"x_1_hard_min": 0,
	# 		"x_1_hard_max": None,
	
	# 		"y_1_hard_min": 0,
	# 		"y_1_hard_max": 0.12,
	
	# 		"input_dim": 1,  # y
	# 		"output_dim": 1,  # u
	
	# 		"u_1_p" : 0.5,
	# 		"u_1_lambd" : 4,
	
	# 		"dt" : 0.01,
	
	# 		"batch_size": 2000,
	# 		"seq_len":    2001,
	
	# 		"min_correlation_threshold": -1.1
	
	# 	}
	
	# cfg2 = {
	# 		"u_1_D_center_min": 0.0,
	# 		"u_1_D_center_max": 1.0,
	
	# 		"u_1_hard_min": 0.0,
	# 		"u_1_hard_max": 1,
	
	# 		"x_1_hard_min": 0,
	# 		"x_1_hard_max": None,
	
	# 		"y_1_hard_min": 0,
	# 		"y_1_hard_max": 0.12,
	
	# 		"input_dim": 1,  # y
	# 		"output_dim": 1,  # u
	
	# 		"u_1_p" : 0.5,
	# 		"u_1_lambd" : 4,
	
	# 		"dt" : 0.01,
	
	# 		"batch_size": 10000,
	# 		"seq_len":    2001,
	
	# 		"min_correlation_threshold": -1.1
	
	# 	}
	
    # Generate and save MIMO dataset with delays
	data = generate_and_save_dataset(
        plant,
        hyperparam_config["training_data_cfg"],
        dirname=dirname1,
        show_plots=False,  
		show_overlay_plot=True,
        save_logs=True
		)
	# stats = {
	# 	"shape": tuple(data.shape),
	# 	"dtype": str(data.dtype),
	# 	"mean": data.mean(dim=0),
	# 	"std": data.std(dim=0),
	# 	"min": data.amin(dim=0),
	# 	"max": data.amax(dim=0),
	# 	"quantiles": torch.quantile(data, torch.tensor([0.25, 0.5, 0.75]), dim=0),
	# }

	print(data.keys)

	# data2 = generate_and_save_dataset(
	# 		plant,
	# 		cfg2,
	# 		dirname=dirname2,
	# 		show_plots=False,  
	# 		show_overlay_plot=True,
	# 		save_logs=True
	# 		)

	


	# from tslearn.clustering import TimeSeriesKMeans
	# from tslearn.preprocessing import TimeSeriesScalerMeanVariance
	# X = data["y"]
	# print(X.shape)
	# X_np = X.cpu().numpy()
	# X_scaled = TimeSeriesScalerMeanVariance().fit_transform(X_np)

	# # Precompute pairwise DTW distances (required for DBSCAN + DTW)
	# dist_matrix = cdist_dtw(X_scaled, n_jobs=-1)

	# #--- Optional: k-distance plot to help pick eps ---
	# min_samples = 3
	# neighbors = NearestNeighbors(n_neighbors=min_samples, metric="precomputed")
	# neighbors.fit(dist_matrix)
	# k_dists = np.sort(neighbors.kneighbors(dist_matrix)[0][:, -1])
	# plt.plot(k_dists); plt.ylabel(f"{min_samples}-NN DTW distance"); plt.show()
	# #Look for the "elbow" in this curve -> use that value as eps.

	# model = DBSCAN(eps=5.0, min_samples=3, metric="precomputed", n_jobs=-1)
	# cluster_labels = model.fit_predict(dist_matrix)

	# print("Discovered Cluster Labels:", cluster_labels)
	# n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
	# n_noise = list(cluster_labels).count(-1)
	# print(f"Clusters found: {n_clusters}, noise points: {n_noise}")

	# # DBSCAN has no cluster_centers_, so build DTW barycenters manually
	# unique_labels = sorted(set(cluster_labels) - {-1})
	# centroids = np.array([
	# 	dtw_barycenter_averaging(X_scaled[cluster_labels == lbl])
	# 	for lbl in unique_labels
	# ])
	# print("Cluster Centroids Shape:", centroids.shape)
	# seq_len = X.shape[1]
	# time_axis = np.linspace(0, 10, seq_len)

	# pil_img = plot_clustered_signals_pil(
	# 	t=time_axis,
	# 	X=X,
	# 	cluster_labels=cluster_labels,
	# 	centroids=None,        
	# 	title="MIMO Trajectory Clusters (tslearn DTW)",
	# 	xlabel=rf"t / h",
	# 	ylabel=rf"u_1",
	# 	figsize=(7, 4),
	# 	show=True,
	# 	filename="clustered_signals.png",
	# 	dirname="cluster_plots",
	# 	asp=0.4
	# )
if __name__ == "__main__":
	main()