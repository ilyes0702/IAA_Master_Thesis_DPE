"""Script to generate and save training data for plant models.

This script selects a plant model, generates a dataset using
generate_and_save_dataset and writes it to a folder named
<PlantClassName>_training_data.

Usage: run this script from the project root (or ensure PYTHONPATH
includes the project) so imports resolve correctly.
"""

from src.sample.utils.data_generation_utils import generate_and_save_dataset
from src.sample.utils.data_generation_utils import *
from src.sample.classes.plants.ChemostatPlant import ChemostatPlant
from src.sample.classes.plants.IdiophasePlant import IdiophasePlant
from src.sample.classes.plants.MassSpringDamperPlant import MassSpringDamperPlant
from src.sample.classes.plants.BajpaiReussPlant import BajpaiReussPlant

from src.sample.classes.plants.YeastFermentation import FedBatchYeastPlant
from src.sample.classes.plants.TrophophasePlant import TrophophasePlant
from src.hyperparam_config import *
import matplotlib.pyplot as plt
from src.sample.classes.plants.PenicillinPlantBirol2002 import PenicillinPlantBirol2002

# Apply project style for plots (if plotting is enabled)
plt.style.use("src/sample/style.mplstyle")


def main() -> None:
	"""Create a plant instance and generate training data.

	Change which plant is instantiated by uncommenting the other
	option below. The dataset folder is named after the plant class.
	"""

	#hyperparam_config = hyperparam_config_BajpaiReussPlant
	hyperparam_config = hyperparam_config_TrophophasePlant

	plant = TrophophasePlant(hyperparam_config=hyperparam_config)

	#plant = IdiophasePlant(hyperparam_config)
	dirname = plant.__class__.__name__ + "_training_data"

    # Generate and save MIMO dataset with delays
	generate_and_save_dataset(
        plant,
        hyperparam_config,
        dirname=dirname,
        show_plots=True,  
        save_logs=True
		)

	print()

if __name__ == "__main__":
	main()