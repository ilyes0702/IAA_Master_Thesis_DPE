
hyperparam_config_ChemostatPlant = {
    "plant" :{
        "mu-max": 0.5,
        "Ks": 0.2,
        "Y": 0.6,
        "sR": 1.0,
        "D_center_min": 0.25,
        "D_center_max": 0.3,
        "u_hard_min": 0.0,
        "u_hard_max": 1.0,
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
        "d_state": 16,
        "input_dim": 1,  # y
        "output_dim": 1  # u
    },
    "simulate": {
        "batch_size": 10,
        "seq_len": 500,
    }
}


hyperparam_config_FedBatchYeastPlant = {
    "plant": {
        # Stoichiometric Yield Coefficients (Table 1)
        "a1": 0.396,       # g O2 / g S
        "b1": 0.490,       # g X / g S
        "c1": 0.590,       # g CO2 / g S
        "b2": 0.050,       # g X / g S
        "c2": 0.462,       # g CO2 / g S
        "d2": 0.480,       # g P / g S
        "a3": 1.104,       # g O2 / g P
        "b3": 0.720,       # g X / g P
        "c3": 0.625,       # g CO2 / g P

        # Kinetic Parameters (Table 2)
        "ks": 3.500,       # g S / g X h
        "ko": 0.256,       # g O2 / g X h
        "kp": 0.170,       # g P / g X h
        "Ks_val": 0.100,   # g S / L
        "Ko": 0.001,       # g O2 / L
        "Kp": 0.100,       # g P / L

        # Process / Feed Concentrations (Table 3)
        "Sin": 300.0,      # g S / L (Inlet substrate concentration)
        "Pin": 10.0,       # g P / L (Inlet ethanol concentration)
        "O2_star": 0.039,  # g O2 / L (Equilibrium oxygen concentration)
        "kla": 250.0,      # h^-1 (Mass transfer coefficient)

        # Operational Boundaries & Limits (Table 3 / Section 3)
        "V_init": 4.0,     # L (Initial liquid volume)
        "V_max": 8.0,      # L (Maximum reactor volume constraint)
        "u_1_hard_min": 0.0, # L/h (Minimum volumetric feed rate)
        "u_1_hard_max": 3.0, # L/h (Maximum volumetric feed rate F_max),
        "u_1_D_center_min": 0.5, # Relative position of D_center within [u_hard_min, u_hard_max]
        "u_1_D_center_max": 2.5, # Relative position of D
    },
    "signal": {
        "lambd": 1,
        "p": 0.5,
        "seq_len": 161,    # Tailored to match the ~16h total batch time at dt=0.1
        "dt": 0.1          # Hours (Sampling period h matching simulation timeframes)
    },
    "train": {
        "k_folds": 5,
        "epochs": 50,
        "batch_size": 20000, # Adjusted downward for a more complex 5-state system
        "lr": 1e-3,
        "device": "cuda",   # "cuda" if torch.cuda.is_available() else "cpu"
        "delay_steps": 1,
        "loss_function": "MSELoss()",
        "lr_decay_rate": 1,
    },
    "mamba": {
        "d_model": 256,
        "d_state": 256,
        "input_dim": 1,    # Observed feedback target (P)
        "output_dim": 1,
        "n_layers": 1      # Manipulated value (Feed rate F)
    },
    "simulate": {
        "batch_size": 10,
        "seq_len": 150,    # Capable of capturing the optimal tf* = 16.12 h profile
    }
}

hyperparam_config_IdiophasePlant = {
        "signal": {
            "seq_len": 351,
            "dt": 0.1,
            #/ 🕹️ Channel 1 Signal Parameters (e.g., highly dynamic)
            "u_1_lambd": 5.0,        
            "u_1_p": 0.05,            
            
            #// 🕹️ Channel 2 Signal Parameters (e.g., highly filtered, slow moving)
            "u_2_lambd": 5.0,        
            "u_2_p": 0.05
        },
        "train": {
            "batch_size": 10000,
            "device": "cuda",
            "delay_steps": 1,
            "epochs": 50,
            "lr": 1e-3,
            "loss_function": "MSELoss()",
            "k_folds": 5,
            "lr_decay_rate":1,
        },
        "plant": {
            "mu_max": 0.12,
            "Ks": 50.0,
            "p1": 0.00047,
            "p2": 200000.0,
            "p5": 0.9,
            "p6": 100.0,
            "p7": 0.04,
            "q": 2000.0,
            "mu_Pen": 3.0,
            "V_idiophase": 170.0,
            
            "u_1_hard_min": 0.0,
            "u_1_hard_max": 1.0,
            "u_2_hard_min": 0.0,
            "u_2_hard_max": 1.0,
            "y_1_hard_min": 0.0,
            "y_1_hard_max": None,
            "y_2_hard_min": 0.0,
            "y_2_hard_max": None,

            "u_1_D_center_min": 0.8,
            "u_1_D_center_max": 0.9,

            "u_2_D_center_min": 0.8,
            "u_2_D_center_max": 0.9,
        },
        "mamba": {
            "d_model": 64,
            "d_state": 16,
            "input_dim": 2,  # y1, y2
            "output_dim": 2  # u1, u2
        },
        "simulate": {
            "batch_size": 10,
            "seq_len": 351,
        }
    }

hyperparam_config_MassSpringDamperPlant = {
    "plant" :{
        "m": 0.1,
        "d": 0.1,
        "k": 1.0,
        "D_center_min": 0.5,
        "D_center_max": 0.5,
        "u_hard_min": 0,
        "u_hard_max": 1.0
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
        "batch_size":5000,
        "lr": 1e-3,
        "device": "cuda", # if torch.cuda.is_available() else "cpu",
        "delay_steps": 1,
        "loss_function": "MSELoss()", 
        "lr_decay_rate":1,
    },
    "mamba": {
        "d_model": 64,
        "d_state": 16,
        "input_dim": 1,  # y
        "output_dim": 1  # u
    },
    "simulate": {
        "batch_size": 10,
        "seq_len": 10,
    }
}

hyperparam_config_TrophophasePlant = {
    "plant" :{
        "mu_max": 0.12,
        "Ks": 50,
        "m_S": 23.0,
        "p1": 0.00047,
        "p2": 200000.0,
        "D_center_min": 0.12,
        "D_center_max": 0.88,
        "u_hard_min": 0.00001,
        "u_hard_max": 0.99999,
        "y_1_hard_min": 0.0,
        "y_1_hard_max": None
    },
    "signal": {
        "lambd": 20,
        "p": 0.1,
        "seq_len": 251,
        "dt": 0.1
    },
    "train": {
        "k_folds": 5,
        "epochs": 50,
        "batch_size": 10000,
        "lr": 1e-3,
        "device": "cuda", # if torch.cuda.is_available() else "cpu",
        "delay_steps": 1,
        "loss_function": "MSELoss()", 
        "lr_decay_rate":1,
    },
    "mamba": {
        "d_model": 256,
        "d_state": 16,
        "input_dim": 1,  # y
        "output_dim": 1  # u
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