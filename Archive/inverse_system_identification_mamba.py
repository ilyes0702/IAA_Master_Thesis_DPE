import matplotlib.pyplot as plt
plt.style.use("src/sample/style.mplstyle")
import torch

# Choose your plant model here - swap between different process models
from src.sample.classes.PenicilinFermentationProcessTropophase import FermentationProcess
from sample.classes.SimpleLinearPlant import SimpleLinearPlant

from src.sample.classes.MambaInverseController import MambaInverseController
from src.sample.utils.general_utils import simulate_control, train_controller

# --- 1. Device Configuration --- #
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

if __name__ == "__main__":
    # --- SELECT YOUR PLANT MODEL HERE ---
    # Swap these lines to use different process models
    #plant = FermentationProcess()
    plant = SimpleLinearPlant()
    dt = 0.01
    dirname = plant.__class__.__name__
    # --- SELECT YOUR CONTROLLER MODEL HERE ---
    # Swap these lines to use different controller architectures
    controller = MambaInverseController().to(device)
    # controller = LSTMInverseController().to(device)

    # 1. Training (Now returns FULL data)
    train_controller(
        model=controller, 
        plant=plant, 
        epochs=8000, 
        seq_len=10000, 
        dt=dt, 
        device=device,
        dirname=dirname
    )
    
    # 2. Simulation
    simulate_control(
        model=controller, 
        plant=plant, 
        reference_signal=plant.ref_value, 
        duration=50, 
        dt=dt, 
        device=device,
        dirname=dirname
    )