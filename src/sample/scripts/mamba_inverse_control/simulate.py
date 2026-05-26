import matplotlib.pyplot as plt
from src.sample.config import get_run_description, log_message
plt.style.use("src/sample/style.mplstyle")
import torch

# Choose your plant model here - swap between different process models
from src.sample.utils.general_utils import *
from src.hyperparam_config import hyperparam_config
from src.sample.classes.ChemostatPlant import ChemostatPlant
from src.sample.classes.MambaInverseController import MambaInverseController_exp

# Device Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

if __name__ == "__main__":
    # 0. Log Run Description
    #run_description = get_run_description()
    #log_message(f"RUN DESCRIPTION: {run_description}")
    
    # 1. Initialize plant 
    plant = ChemostatPlant(hyperparam_config=hyperparam_config) 

    # 2. Load trained controller
    controller_path = "models/2026-05-13/2026-05-13_12-18-55/GPUChemostatPlant_training_None/2026-05-13_12-18-55_trained_controller_disk.pt"

    controller_path_4000 = "models/2026-05-18/2026-05-18_11-35-09/ChemostatPlant_training_4000/2026-05-18_11-35-09_trained_controller_disk.pt"

    controller_path = "models/2026-05-19/2026-05-19_12-01-09/ChemostatPlant_training_6000/2026-05-19_12-01-09_trained_controller_disk.pt"

    controller_path = "models/2026-05-19/2026-05-19_13-02-01/ChemostatPlant_training_None/2026-05-19_13-02-01_trained_controller_disk.pt"

    controller_path = "models/2026-05-19/2026-05-19_15-27-50/ChemostatPlant_training_None/2026-05-19_15-27-50_trained_controller_disk.pt" # trained with bounded relative loss and 20-step delay, ok results

    controller_path = "models/2026-05-25/2026-05-25_10-03-58/ChemostatPlant_training_None/fold_5/2026-05-25_10-03-58_best_fold_model.pt" #
    
    loaded_controller = load_model(MambaInverseController_exp, controller_path)

    normalization_stats_path = "results/2026-05-25/2026-05-25_10-03-58/ChemostatPlant_training_None/reports/2026-05-25_10-03-58_normalization_stats.json"
    

    r_dynamic = generate_reference_trajectory(
        steps=hyperparam_config["simulate"]["seq_len"],
        dt=hyperparam_config["signal"]["dt"],
        device=hyperparam_config["train"]["device"],
        mode="dynamic",
        gain=1.5,      # Sharper transition edges
        period=40.0    # 24-hour cycle dynamics
    )

    r_static = generate_reference_trajectory(
        steps=hyperparam_config["simulate"]["seq_len"],
        dt=hyperparam_config["signal"]["dt"],
        device=hyperparam_config["train"]["device"],
        mode="constant",
        constant_val=0.2
    )

    
    print(f"Model class: {loaded_controller.__class__.__name__}")  # Should print "MambaInverseController_exp"
    print("Output projection bias:", loaded_controller.output_proj.bias.item())  # Likely non-zero!
    # loaded_controller.output_proj.bias.data.fill_(0)  # Zero the bias term
    # print("Output projection bias:", loaded_controller.output_proj.bias.item())

    simulate_tracking_exp(
        model=loaded_controller,
        plant=plant,
        r_trajectory=r_dynamic,
        hyperparam_config=hyperparam_config,
        normalization_stats_path=normalization_stats_path,
        dirname=plant.__class__.__name__
    )

    