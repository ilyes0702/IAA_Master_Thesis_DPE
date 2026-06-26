"""
Train the Mamba inverse controller for the Chemostat plant.

This script loads a prepared training dataset (features X and targets Y),
initializes the plant and controller using the shared hyperparameter
configuration, and runs the training routine to produce a trained controller
and any associated artifacts (saved models, logs, plots).

Usage: run this file as a script. Adjust dataset_path below if needed.
"""

import torch

from src.sample.classes.plants.PenicillinPlantBirol2002 import PenicillinPlantBirol2002
from src.sample.config import *

from src.sample.classes.plants.ChemostatPlant import ChemostatPlant
from src.sample.classes.plants.MassSpringDamperPlant import MassSpringDamperPlant
from src.sample.classes.plants.TrophophasePlant import TrophophasePlant
from src.sample.classes.plants.IdiophasePlant import IdiophasePlant

from src.sample.classes.plants.YeastFermentation import FedBatchYeastPlant

from src.sample.classes.controllers.MambaInverseController import *
from src.sample.classes.controllers.ESNInverseController import *



from src.hyperparam_config import *

from src.sample.utils.training_utils import * 
from src.sample.utils.saving_utils import *

# Device configuration: use GPU if available, otherwise fallback to CPU.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


if __name__ == "__main__":
    # Instantiate plant using shared hyperparameters.
    hyperparam_config = hyperparam_config_TrophophasePlant
    
    plant = TrophophasePlant(hyperparam_config=hyperparam_config)
    
    dataset_path = (
       "results/2026-06-26/2026-06-26_14-43-08/TrophophasePlant_training_data/dataset/2026-06-26_14-43-08_training_data.pt"
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
    #controller = MambaInverseController(hyperparam_config=hyperparam_config).to(device)

    controller = MambaInverseController_stateful(hyperparam_config=hyperparam_config)

    # Directory name to store training artifacts (models, plots, logs).
    dirname = f"{plant.__class__.__name__}_training"

    save_to_json(hyperparam_config, dirname, "traingin_data_path")
    save_to_json(hyperparam_config, dirname,"hyperparam_config")
    # Run the training routine. show_plots can be toggled for interactive use.
    train_controller(
        controller,
        X,
        Y,
        hyperparam_config,
        plant,
        dirname=dirname
    )

        