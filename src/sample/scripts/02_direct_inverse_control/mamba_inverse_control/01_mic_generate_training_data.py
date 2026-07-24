"""Script to generate and save training data for plant models.

This script selects a plant model, generates a dataset using
generate_and_save_dataset and writes it to a folder named
<PlantClassName>_training_data.

Usage: run this script from the project root (or ensure PYTHONPATH
includes the project) so imports resolve correctly.
"""


from src.sample.utils.data_generation_utils import *
from src.sample.classes.plants.ChemostatPlant import *
from src.sample.classes.plants.IdiophasePlant import *
from src.sample.classes.plants.MassSpringDamperPlant import *
from src.sample.classes.plants.BajpaiReussPlant import *
from src.sample.classes.plants.TrophophasePlant import *
from src.sample.classes.plants.PenicillinPlantBirol2002 import *
from src.sample.classes.plants.CocultivationPlant import *
from src.sample.classes.plants.IndForProteinProductionPlant import *

from src.hyperparam_config import *
import matplotlib.pyplot as plt
from src.sample.utils.plotting_utils import *


# Apply project style for plots (if plotting is enabled)
plt.style.use("src/sample/style.mplstyle")


def main() -> None:
	"""Create a plant instance and generate training data.

	Change which plant is instantiated by uncommenting the other
	option below. The dataset folder is named after the plant class.
	"""

	#hyperparam_config = hyperparam_config_BajpaiReussPlant
	hyperparam_config = hyperparam_config_ChemostatPlant

	plant = ChemostatPlant(hyperparam_config=hyperparam_config)

	#plant = IdiophasePlant(hyperparam_config)
	dirname = plant.__class__.__name__ + "_training_data"

    # Generate and save MIMO dataset with delays
	data = generate_and_save_dataset(
        plant,
        hyperparam_config,
        dirname=dirname,
        show_plots=False,  
        save_logs=True
		)

	from tslearn.clustering import TimeSeriesKMeans
	from tslearn.preprocessing import TimeSeriesScalerMeanVariance
	X = data["u"]
	print(X.shape)
	X_np = X.cpu().numpy()
	X_scaled = TimeSeriesScalerMeanVariance().fit_transform(X_np)
	model = TimeSeriesKMeans(n_clusters= 3,
							metric="dtw",
							max_iter=10,
							random_state=42,
							n_jobs = -1)
	cluster_labels = model.fit_predict(X_scaled)

	print("Discovered Cluster Labels:", cluster_labels)
	# Shape of cluster centroids (barycenters): [n_clusters, 1001, 1]
	print("Cluster Centroids Shape:", model.cluster_centers_.shape)
	# model.fit() 

	seq_len = X.shape[1]  # 1001
	time_axis = np.linspace(0, 10, seq_len)  # or your dt-based array

	pil_img = plot_clustered_signals_pil(
		t=time_axis,
		X=X,                             # [35, 1001, 1] tensor or numpy array
		cluster_labels=cluster_labels,   # Array of length 35
		centroids=model.cluster_centers_,# [3, 1001, 1]
		title="MIMO Trajectory Clusters (tslearn DTW)",
		xlabel="Time (h)",
		ylabel="u_1",
		figsize=(7, 4),
		show=True,
		filename="clustered_signals.png",
		dirname="cluster_plots",
		asp=0.4
	)

if __name__ == "__main__":
	main()