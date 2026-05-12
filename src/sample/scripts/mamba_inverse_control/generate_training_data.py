from src.sample.utils.data_generation_utils import generate_and_save_dataset
from src.sample.classes.GPUChemostatPlant import GPUChemostatPlant
from src.hyperparam_config import hyperparam_config
import matplotlib.pyplot as plt
plt.style.use("src/sample/style.mplstyle")
plant = GPUChemostatPlant(hyperparam_config=hyperparam_config)

dirname = plant.__class__.__name__ + "_training_ata"


generate_and_save_dataset(plant, hyperparam_config, num_batches=1, dirname= dirname)
