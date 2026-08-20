"""Script to generate and save training data for plant models.

This script selects a plant model, generates a dataset using
generate_and_save_dataset and writes it to a folder named
<PlantClassName>_training_data.

Usage: run this script from the project root (or ensure PYTHONPATH
includes the project) so imports resolve correctly.
"""

from seqControl.sample.utils.data_generation_utils import generate_and_save_dataset
from seqControl.sample.utils.data_generation_utils import *
from seqControl.sample.classes.plants.ChemostatPlant import *
from seqControl.sample.classes.plants.IdiophasePlant import IdiophasePlant
from seqControl.sample.classes.plants.MassSpringDamperPlant import MassSpringDamperPlant
from seqControl.sample.classes.plants.BajpaiReussPlant import BajpaiReussPlant

from seqControl.sample.classes.plants.TrophophasePlant import *
from seqControl.hyperparam_config import *
import matplotlib.pyplot as plt
from seqControl.sample.classes.plants.PenicillinPlantBirol2002 import PenicillinPlantBirol2002
from seqControl.sample.classes.plants.SimpleLinearPlant import SimpleLinearPlant

# Apply project style for plots (if plotting is enabled)
plt.style.use("src/sample/style.mplstyle")


def main() -> None:
	"""Create a plant instance and generate training data.

	Change which plant is instantiated by uncommenting the other
	option below. The dataset folder is named after the plant class.
	"""

	#hyperparam_config = hyperparam_config_BajpaiReussPlant
	hyperparam_config = hyperparam_config_TrophophasePlant
	print(hyperparam_config)
	plant = TrophophasePlant(hyperparam_config=hyperparam_config)

	#plant = IdiophasePlant(hyperparam_config)
	dirname = plant.__class__.__name__ + "_training_data"

    # Generate and save MIMO dataset with delays
	generate_and_save_dataset_marcia(
        plant,
        hyperparam_config,
        dirname=dirname,
        show_plots=True,  
        save_logs=True
		)


if __name__ == "__main__":
	main()