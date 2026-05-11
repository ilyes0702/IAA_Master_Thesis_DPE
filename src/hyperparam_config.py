import torch
hyperparam_config = {
    "plant": {
        "tau": 2.0,
        "gain": 0.8,
        "y_max": 1.0,  # Scaling constant
        "u_max": 1.0   # Scaling constant
    },
    "signal": {
        "lambd": 200,
        "p": 0.2,
        "seq_len": 1000,
        "dt": 0.5
    },
    "train": {
        "epochs": 30,
        "batch_size": 256,
        "lr": 1e-3,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "delay_steps": 20
    },
    "mamba": {
        "d_model": 64,
        "n_layers": 4,
        "d_state": 16
    },
    "simulate": {
        "batch_size": 10,
        "seq_len": 200,
    }
}