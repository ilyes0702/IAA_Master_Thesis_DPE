import torch
hyperparam_config = {
    "plant": {
        "tau": 2.0,
        "gain": 0.8,
        "y_max": 1.0,  # Scaling constant
        "u_max": 1.0   # Scaling constant
    },
    "signal": {
        "lambd": 5.0,
        "p": 0.4,
        "seq_len": 500,
        "dt": 0.01
    },
    "train": {
        "epochs": 100,
        "batch_size": 512,
        "lr": 1e-3,
        "device": "cuda" if torch.cuda.is_available() else "cpu"
    },
    "mamba": {
        "d_model": 64,
        "n_layers": 4,
        "d_state": 16
    }
}