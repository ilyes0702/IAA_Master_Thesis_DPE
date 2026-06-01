"""
Train the Mamba inverse controller for the Chemostat plant.

This script loads a prepared training dataset (features X and targets Y),
initializes the plant and controller using the shared hyperparameter
configuration, and runs the training routine to produce a trained controller
and any associated artifacts (saved models, logs, plots).

Usage: run this file as a script. Adjust dataset_path below if needed.
"""

import torch

from src.sample.config import *

from src.sample.classes.ChemostatPlant import ChemostatPlant
from src.sample.classes.MassSpringDamperPlant import MassSpringDamperPlant
from src.sample.classes.TrophophasePlant import TrophophasePlant
from src.sample.classes.IdiophasePlant import IdiophasePlant
from src.sample.classes.MambaInverseController import *

from src.hyperparam_config import *

from src.sample.utils.training_utils import *  # train_controller, etc.

from src.sample.utils.saving_utils import *

# Device configuration: use GPU if available, otherwise fallback to CPU.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


if __name__ == "__main__":
    # Instantiate plant using shared hyperparameters.
    hyperparam_config = hyperparam_config_ChemostatPlant
    
    plant = ChemostatPlant(hyperparam_config=hyperparam_config)
    #plant = TrophophasePlant(hyperparam_config=hyperparam_config_TrophophasePlant)

    dataset_path = (
       "results/2026-06-01/2026-06-01_12-58-27/ChemostatPlant_training_data/dataset/2026-06-01_12-58-27_training_data.pt"
    )


    # Load the dataset from disk. The file is expected to be a dict-like object containing 'x' and 'y' keys. If loading fails, inspect the path first.
    dataset = torch.load(dataset_path, weights_only=True)

    # Extract features (X) and targets (Y).
    # X: (total_sequences, seq_len, 2*num_outputs) -> [y_t, y_next]
    # Y: (total_sequences, seq_len, 1*num_inputs) -> [u_control]
    X = dataset["x"]
    Y = dataset["y"]

    # Quick sanity prints to confirm shapes and sample values.
    print("✅ Dataset successfully loaded!")
    print(f"Features (X) shape: {X.shape}")
    print(f"Targets  (Y) shape: {Y.shape}")


    # Initialize the inverse controller and move it to the selected device.
    controller = MambaInverseController(hyperparam_config=hyperparam_config).to(device)

    # Directory name to store training artifacts (models, plots, logs).
    dirname = f"{plant.__class__.__name__}_training"

    # Run the training routine. show_plots can be toggled for interactive use.
    train_controller(
        controller,
        X,
        Y,
        hyperparam_config,
        dirname=dirname,
        show_plots=False,
    )
        
        