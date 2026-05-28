"""Script to generate and save training data for plant models.

This script selects a plant model, generates a dataset using
generate_and_save_dataset and writes it to a folder named
<PlantClassName>_training_data.

Usage: run this script from the project root (or ensure PYTHONPATH
includes the project) so imports resolve correctly.
"""

from src.sample.utils.data_generation_utils import *
from src.sample.classes.ChemostatPlant import ChemostatPlant
from src.sample.classes.TrophophasePlant import TrophophasePlant
from src.hyperparam_config import *
import matplotlib.pyplot as plt

# Apply project style for plots (if plotting is enabled)
plt.style.use("src/sample/style.mplstyle")


def main() -> None:
	"""Create a plant instance and generate training data.

	Change which plant is instantiated by uncommenting the other
	option below. The dataset folder is named after the plant class.
	"""

	hyperparam_config = hyperparam_config_ChemostatPlant

	# Choose plant model and corresponding hyperparameters
	#plant = ChemostatPlant(hyperparam_config=hyperparam_config)
	plant = ChemostatPlant(hyperparam_config=hyperparam_config)

	# Directory name will be e.g. 'ChemostatPlant_training_data'
	dirname = plant.__class__.__name__ + "_training_data"

	# Generate and save a single batch of training data without showing plots
	generate_and_save_dataset(plant, 
						   hyperparam_config, 
						   dirname=dirname, 
						   show_plots=False)


if __name__ == "__main__":
	main()