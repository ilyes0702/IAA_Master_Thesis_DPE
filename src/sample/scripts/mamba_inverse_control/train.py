"""
Train the Mamba inverse controller for the Chemostat plant.

This script loads a prepared training dataset (features X and targets Y),
initializes the plant and controller using the shared hyperparameter
configuration, and runs the training routine to produce a trained controller
and any associated artifacts (saved models, logs, plots).

Usage: run this file as a script. Adjust dataset_path below if needed.
"""

import torch
from src.sample.classes.ChemostatPlant import ChemostatPlant
from src.hyperparam_config import hyperparam_config
from src.sample.classes.MambaInverseController import MambaInverseController
from src.sample.utils.training_utils import *  # train_controller, etc.
from src.sample.config import *
from src.sample.utils.saving_utils import *

# Device configuration: use GPU if available, otherwise fallback to CPU.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


if __name__ == "__main__":
    # Instantiate plant using shared hyperparameters (for naming and metadata).
    plant = ChemostatPlant(hyperparam_config=hyperparam_config)

    # Path to the prepared training dataset. Change if you generated data at
    # a different timestamp or location.
    dataset_path = (
        "results/2026-05-27/2026-05-27_14-20-39/ChemostatPlant_training_data/dataset/2026-05-27_14-20-39_training_data.pt"
    )

    # Load the dataset from disk. The file is expected to be a dict-like object
    # containing 'x' and 'y' keys. If loading fails, inspect the path first.
    dataset = torch.load(dataset_path, weights_only=True)

    # Extract features (X) and targets (Y).
    # X: (total_sequences, seq_len, 2) -> [y_t, y_next]
    # Y: (total_sequences, seq_len, 1) -> [u_control]
    X = dataset["x"]
    Y = dataset["y"]

    # Quick sanity prints to confirm shapes and sample values.
    print("✅ Dataset successfully loaded!")
    print(f"Features (X) shape: {X.shape}")
    print(f"Targets  (Y) shape: {Y.shape}")

    print("\nFirst sequence item sample:")
    print(f"First step inputs [y(t), y(t+Δ)]: {X[0, 0]}")
    print(f"First step target control [u(t)]: {Y[0, 0]}")

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
        
        