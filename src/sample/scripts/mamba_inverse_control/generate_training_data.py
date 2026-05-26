from src.sample.utils.data_generation_utils_new import generate_and_save_dataset
from src.sample.classes.ChemostatPlant import ChemostatPlant
from src.sample.classes.SecondOrderLinearPlant import SecondOrderLinearPlant
from src.hyperparam_config import hyperparam_config, hyperparam_config_SecondOrderLinearPlant
import matplotlib.pyplot as plt
plt.style.use("src/sample/style.mplstyle")


plant = ChemostatPlant(hyperparam_config=hyperparam_config)
#plant = SecondOrderLinearPlant(hyperparam_config=hyperparam_config_SecondOrderLinearPlant)

dirname = plant.__class__.__name__ + "_training_data"

generate_and_save_dataset(plant, hyperparam_config, num_batches=1, dirname= dirname, show_plots=False)