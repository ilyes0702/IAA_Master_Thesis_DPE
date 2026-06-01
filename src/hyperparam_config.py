
hyperparam_config_ChemostatPlant = {
    "plant" :{
        "mu-max": 0.5,
        "Ks": 0.2,
        "Y": 0.6,
        "sR": 1.0,
        "D_center_min": 0.25,
        "D_center_max": 0.3
    },
    "signal": {
        "lambd": 15,
        "p": 0.15,
        "seq_len": 1001,
        "dt": 0.1
    },
    "train": {
        "k_folds": 5,
        "epochs": 30,
        "batch_size": 10000,
        "lr": 1e-3,
        "device": "cuda", # if torch.cuda.is_available() else "cpu",
        "delay_steps": 1,
        "loss_function": "MSELoss()", 
        "lr_decay_rate":1,
    },
    "mamba": {
        "d_model": 64,
        "d_state": 16
    },
    "simulate": {
        "batch_size": 10,
        "seq_len": 500,
    }
}

hyperparam_config_MassSpringDamperPlant = {
    "plant" :{
        "m": 0.1,
        "d": -0.5,
        "k": 1.0,
        "D_center_min": 0.5,
        "D_center_max": 0.5
    },
    "signal": {
        "lambd": 15,
        "p": 0.15,
        "seq_len": 1001,
        "dt": 0.1
    },
    "train": {
        "k_folds": 5,
        "epochs": 50,
        "batch_size":10000,
        "lr": 1e-3,
        "device": "cuda", # if torch.cuda.is_available() else "cpu",
        "delay_steps": 1,
        "loss_function": "MSELoss()", 
        "lr_decay_rate":1,
    },
    "mamba": {
        "d_model": 64,
        "d_state": 16
    },
    "simulate": {
        "batch_size": 10,
        "seq_len": 500,
    }
}

hyperparam_config_TrophophasePlant = {
    "plant" :{
        "mu-max": 0.5,
        "Ks": 0.1,
        "Y": 0.6,
        "sR": 1.0,
        "Ki": 0.1,
        "D_center_min": 0.1,
        "D_center_max": 0.9,
        "u_min": 0,
        "u_max": 1
    },
    "signal": {
        "lambd": 20,
        "p": 0.09,
        "seq_len": 251,
        "dt": 0.1
    },
    "train": {
        "k_folds": 5,
        "epochs": 50,
        "batch_size": 1000,
        "lr": 1e-3,
        "device": "cuda", # if torch.cuda.is_available() else "cpu",
        "delay_steps": 1,
        "loss_function": "MSELoss()", 
        "lr_decay_rate":1,
    },
    "mamba": {
        "d_model": 64,
        "d_state": 16
    },
    "simulate": {
        "batch_size": 10,
        "seq_len": 251,
    }
}


hyperparam_config_SecondOrderLinearPlant = {
    "signal": {
        "lambd": 200,
        "p": 0.1,
        "seq_len": 1000,
        "dt": 0.01
    },
    "train": {
        "epochs": 1,
        "batch_size": 1000,
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