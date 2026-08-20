import matplotlib.pyplot as plt
plt.style.use("src/sample/style.mplstyle")
import torch

# Choose your plant model here - swap between different process models
from src.sample.classes.PenicilinFermentationProcessTropophase import FermentationProcess, GPUFermentationProcess
from seqControl.sample.classes.plants.SimpleLinearPlant import SimpleLinearPlant

from seqControl.sample.classes.controllers.MambaInverseController import MambaInverseController
from seqControl.sample.utils.general_utils import simulate_control, train_controller
from seqControl.sample.utils.general_utils import load_controller

# --- 1. Device Configuration --- #
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

if __name__ == "__main__":
    #plant = FermentationProcess()
    plant = GPUFermentationProcess() 
    dt = 0.01
    dirname = plant.__class__.__name__

    controlller_path = "models/2026-04-28/2026-04-28_06-52-09/Canaday_Training_Results/2026-04-28_06-52-09_trained_controller.pt"
    loaded_controller, config = load_controller(MambaInverseController, controlller_path, device)

    # 2. Simulation
    simulate_control(
        model=loaded_controller, 
        plant=plant, 
        reference_signal=plant.ref_value, 
        duration=50, 
        dt=dt, 
        device=device,
        dirname=dirname
    )