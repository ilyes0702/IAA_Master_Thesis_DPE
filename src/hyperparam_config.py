import torch
hyperparam_config = {
    "plant": {
        "tau": 2.0,
        "gain": 0.8,
    },
    "signal": {
        "lambd": 20,
        "p": 0.2,
        "seq_len": 200,
        "dt": 0.2
    },
    "train": {
        "epochs": 1,
        "batch_size": 5000,
        "lr": 1e-3,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "delay_steps": 10
    },
    "mamba": {
        "d_model": 64,
        "n_layers": 4,
        "d_state": 16
    },
    "simulate": {
        "batch_size": 10,
        "seq_len": 300,
    }
}