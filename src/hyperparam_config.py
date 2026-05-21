import torch
hyperparam_config = {
    "signal": {
        "lambd": 15,
        "p": 0.15,
        "seq_len": 100,
        "dt": 0.2
    },
    "train": {
        "number_of_folds": 5,
        "epochs": 30,
        "batch_size": 2000,
        "lr": 1e-3,
        "device": "cuda", # if torch.cuda.is_available() else "cpu",
        "delay_steps": 10,
        "loss_function": "nn.MSELoss()", # "mape_loss", "sobolev_loss", "log_cosh_loss", "cosine_shape_loss"
        "lr_decay_rate":1,
        "critical_loss_value": 1e-2,
        "patience_steps": 1000
    },
    "mamba": {
        "d_model": 64,
        "d_state": 64
    },
    "simulate": {
        "batch_size": 10,
        "seq_len": 500,
    }
}

hyperparam_config_SecondOrderLinearPlant = {
    "signal": {
        "lambd": 100,
        "p": 0.1,
        "seq_len": 100,
        "dt": 0.01
    },
    "train": {
        "epochs": 1,
        "batch_size": 5000,
        "lr": 1e-3,
        "device": "cuda", # if torch.cuda.is_available() else "cpu",
        "delay_steps": 1,
        "loss_function": "nn.MSELoss()", # "mape_loss", "sobolev_loss", "log_cosh_loss", "cosine_shape_loss"
        "lr_decay_rate":1,
        "critical_loss_value": 1e-3,
        "patience_steps": 1000
    },
    "mamba": {
        "d_model": 2,
        "n_layers": 1,
        "d_state": 64
    },
    "simulate": {
        "batch_size": 10,
        "seq_len": 500,
    }
}