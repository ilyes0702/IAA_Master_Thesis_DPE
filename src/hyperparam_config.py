import torch
hyperparam_config = {
    "signal": {
        "lambd": 20,
        "p": 0.2,
        "seq_len": 100,
        "dt": 0.5
    },
    "train": {
        "epochs": 1,
        "batch_size": 5000,
        "lr": 1e-3,
        "device": "cuda", # if torch.cuda.is_available() else "cpu",
        "delay_steps": 20,
        "loss_function": "nn.MSELoss()", # "mape_loss", "sobolev_loss", "log_cosh_loss", "cosine_shape_loss"
        "lr_decay_rate":1,
        "critical_loss_value": 1e-5,
        "patience_steps": 1000
    },
    "mamba": {
        "d_model": 64,
        "n_layers": 4,
        "d_state": 16
    },
    "simulate": {
        "batch_size": 10,
        "seq_len": 500,
    }
}