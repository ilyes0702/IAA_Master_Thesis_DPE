


hyperparam_config_ChemostatPlant_og = {
    "plant" :{
        "mu-max": 0.5,      # Maximum growth rate [1/h]
        "Ks": 0.2,          # Half-saturation constant 
        "Y": 0.6,           # Yield coefficient
        "sR": 1.0,

        "u_1_D_center_min": 0.15,
        "u_1_D_center_max": 0.8,

        "u_1_hard_min": 0.0,
        "u_1_hard_max": None,

        "x_1_hard_min" : 0,
        "x_2_hard_min" : None,

        "x_1_hard_min" : 0,
        "x_2_hard_min" : None,

        "y_1_hard_min": 0,
        "y_1_hard_max": 0.5,

        "input_dim": 1,   # number of plant outputs
        "output_dim": 1   # number of plant control inputs

    },
    "signal": {
        "lambd": 15,
        "p": 0.15,
        "seq_len": 1001,
        "dt": 0.1
    },
    "train": {
        "k_folds": 5,
        "epochs": 100,
        "batch_size": 1000,
        "lr": 1e-3,
        "device": "cuda", # if torch.cuda.is_available() else "cpu",
        "delay_steps": 1,
        "loss_function": "MSELoss()", 
        "lr_decay_rate":1,
        "min_correlation_threshold": 0.7
    },
    "mamba": {
        "d_state": 16,
        "input_dim": 1,  # y
        "output_dim": 1,  # u
        "expand": 32
    },
    "simulate": {
        "batch_size": 10,
        "seq_len": 401,
    }
}
hyperparam_config_SimpleLinearPlant = {
    "plant" : {
        # Linear State Space Parameter Matrices (Simplified as scalar parts)
        "a11": -0.1,       # Self-damping / decay component of State 1
        "a12": 1.0,        # Coupling from State 2 to State 1
        "a21": -0.5,       # Restoring force / feedback from State 1 to State 2
        "a22": -0.2,       # Damping coefficient of State 2
        "b1": 0.0,         # Direct control input mapping to State 1
        "b2": 1.0,         # Direct control input mapping to State 2
        "c1": 1.0,         # Output observation matrix component for State 1
        "c2": 0.0,         # Output observation matrix component for State 2

        # Active Shielding & Generation Configurations
        "u_1_D_center_min": -2.0,   # Minimum allowable raw baseline center
        "u_1_D_center_max": 2.0,    # Maximum allowable raw baseline center

        "u_1_hard_min": -5.0,       # Strict physical input lower limit
        "u_1_hard_max": 5.0,        # Strict physical input upper limit

        "x_1_hard_min": -10.0,      # Dynamic State 1 limits
        "x_1_hard_max": 10.0,
        "x_2_hard_min": -10.0,      # Dynamic State 2 limits
        "x_2_hard_max": 10.0,

        "y_1_hard_min": -10.0,      # Output observable constraint boundary
        "y_1_hard_max": 10.0,

        "input_dim": 1,    # Number of plant outputs tracking (y)
        "output_dim": 1    # Number of plant control inputs forcing (u)
    },
    "signal": {
        "lambd": 15,
        "p": 0.5,          # Scaled for linear stability margins
        "seq_len": 1001,
        "dt": 0.1
    },
    "train": {
        "k_folds": 5,
        "epochs": 100,
        "batch_size": 1000,
        "lr": 1e-3,
        "device": "cuda",  # Auto fallback handled inside class if preferred
        "delay_steps": 1,
        "loss_function": "MSELoss()", 
        "lr_decay_rate": 1,
        "min_correlation_threshold": 0.8
    },
    "mamba": {
        "d_state": 16,
        "input_dim": 1,    # y
        "output_dim": 1,   # u
        "expand": 32
    },
    "simulate": {
        "batch_size": 10,
        "seq_len": 401,
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