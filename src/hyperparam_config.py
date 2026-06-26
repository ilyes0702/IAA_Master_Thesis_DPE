
hyperparam_config_ChemostatPlant = {
    "plant" :{
        "mu-max": 0.5,      # Maximum growth rate [1/h]
        "Ks": 0.2,          # Half-saturation constant 
        "Y": 0.6,           # Yield coefficient
        "sR": 1.0,

        "u_1_D_center_min": 0.15,
        "u_1_D_center_max": 0.8,

        "u_1_hard_min": 0.0,
        "u_1_hard_max": 1.0,

        "x_1_hard_min" : 0,
        "x_2_hard_min" : None,

        "x_1_hard_min" : 0,
        "x_2_hard_min" : None,

        "y_1_hard_min": 0 

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
        "batch_size": 10000,
        "lr": 1e-3,
        "device": "cuda", # if torch.cuda.is_available() else "cpu",
        "delay_steps": 1,
        "loss_function": "MSELoss()", 
        "lr_decay_rate":1,
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

hyperparam_config_BajpaiReussPlant = {
    "plant": {
        # --- Kinematic & Yield Parameters ---
        "mu_x": 0.092,     # Maximum specific growth rate [1/h] (Contois)
        "K_x": 0.015,       # Contois saturation constant [g substrate / g biomass]
        "mu_p": 0.005,       # Maximum specific rate of product formation ( g product / (g dry wt cells)* h)
        

        
        "Y_xs": 0.45,       # Yield of biomass on substrate [g dry wt cells/g substrate]
        "Y_ps": 0.9,        # Xield of product on substrate [g product/ g substrate]

        "m_x" : 0.014,      # Maintenance requirement of substrate [g substrate/ (g dry wet cell)*h]  
        "F": 0.33,          # [g glucose/(dm^3)*h]

        "S_0": 0.1,         # [g/dm^3]                
        
        "pi_max": 0.004,    # Maximum specific product formation rate [1/h]
        "K_p": 0.0002,      # Monod saturation constants for substrate limitation of product formation [g/dm^3]
        "K_I": 0.10,        # Substrate inhibition constant for product [g/dm^3]
        "Y_ps": 1.2,        # Yield factor: penicillin produced per substrate consumed [g/g]

        "S_F": 400,       #Substrate concentration in feed stream [g/dm^3] ##NOT SURE
        
        "K": 0.04,          # First-order decay rate constant for product [1/h]

        "K_La": 60,         #[1/h]

        "C_L_star": 0.27,   # solubility of oxygen in broth [mmol/dm^3]


        "m_x": 0.029,       # Maintenance energy coefficient [g substrate / g biomass / h]
        "K_ox": 0.00111,    # Contois saturation constant for oxygen limitation of product formation [(mmol/ g dry wt cells)]
        "K_op": 3e-5,       # Contois saturation constant for oxygen limitation of product formation [(mmol/g dry wt cells)^1/p]
        "p": 2.74,          # Exponent of CL in oxygen limitation of product formation
        "m_o": 0.467,       # Maintenance requirement of oxygen [mmol O2/(g dry wt cells)*h]
        "Y_xo": 0.04,       # Yield of biomass on oxygen [g dry wt cells/ mmol O2]
        "Y_po": 0.2,        # Yield of product on oxygen [g product/mmol O2]

        # --- Volumetric Flow Rate Control Bounds (u_1 = F) ---
        "u_1_D_center_min": 0.024,  # Typical optimal feeding profile floor [L/h]
        "u_1_D_center_max": 0.024,  # Typical optimal feeding profile ceiling [L/h]

        "u_1_hard_min": 0.0,       # Pump fully off [L/h]
        "u_1_hard_max": 2.0,       # Maximum physical actuator saturation pump limit [L/h]

        # --- Hard State Lower/Upper Bounds ---
        "x_1_hard_min": 0.0,       # Biomass (X) cannot be negative
        "x_2_hard_min": 0.0,       # Substrate (S) cannot be negative
        "x_3_hard_min": 0.0,       # Penicillin (P) cannot be negative
        "x_4_hard_min": 0.0,       # C_L cannot be negative
        "x_5_hard_min": 0.1,       # Volume (V) must maintain a physical minimum heel (e.g. 0.1L)
        "x_5_hard_max": 15.0,      # Maximum structural capacity limit of the vessel tank [L]
    },
    "signal": {
        "lambd": 10,
        "p": 0.15,
        "seq_len": 1001,
        "dt": 0.01                 # Fermentations evolve slower than chemostats; a slightly higher dt is normal
    },
    "train": {
        "k_folds": 5,
        "epochs": 100,
        "batch_size": 20,
        "lr": 1e-3,
        "device": "cuda",
        "delay_steps": 1,
        "loss_function": "MSELoss()", 
        "lr_decay_rate": 1,
    },
    "mamba": {
        "d_state": 16,
        "input_dim": 1,            # Tracking 1 observable output (e.g., pi or mu)
        "output_dim": 1,           # Regulating 1 physical control output (Feed Rate F)
        "expand": 32               # Expansion factor maps core dimension (input_dim * 2) -> 64 internal tracking lines
    },
    "simulate": {
        "batch_size": 10,
        "seq_len": 2001,            # Fed-batch runs span much longer horizons (e.g., 200 hours total at dt=0.25)
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
        "n_layers": 1,      # Manipulated value (Feed rate F),
        "expand": 20         # Expansion factor for the Mamba core to handle increased complexity
    },
    "simulate": {
        "batch_size": 10,
        "seq_len": 150,    # Capable of capturing the optimal tf* = 16.12 h profile
    }
}

hyperparam_config_PenicillinPlantBirol2002 = {
    "signal": {
        "seq_len": 201,
        "dt": 0.01,  # 0.1 hours (6 minutes) integration frequency
        
        # 🕹️ Control Input Channel 1: Substrate Feed Flow Rate (F)
        "u_1_lambd": 15.0,        
        "u_1_p": 0.1,            
    },
    "train": {
        "batch_size": 100,
        "device": "cuda",  # Leverages GPU for massive 10k batch parallelism
        "delay_steps": 1,
        "epochs": 50,
        "lr": 1e-3,
        "loss_function": "MSELoss()",
        "k_folds": 5,
        "lr_decay_rate": 1,
    },
    "plant": {
        # --- Kinematics & Growth Parameters (From Paper Table 2) ---
        "mu_x": 0.092,     # Maximum specific biomass growth rate (1/h)
        "K_x": 0.15,       # Contois saturation constant for biomass substrate (g/g)
        "mu_p": 0.005,     # Specific rate of penicillin production (1/h)
        "K_p": 0.0002,     # Substrate inhibition constant for production (g/L)
        "K_I": 0.10,       # High substrate concentration inhibition parameter (g/L)
        "p_pow": 3.0,      # Yield equation exponential factor 'p'
        "K_h": 0.04,       # Penicillin product hydrolysis decay constant (1/h)
        
        # --- Yield Coefficients & Operational Stream Values ---
        "Y_xs": 0.45,      # Yield factor: grams of Biomass formed per gram of Glucose
        "Y_ps": 0.90,      # Yield factor: grams of Penicillin formed per gram of Glucose
        "m_x": 0.014,      # Maintenance coefficient requirement on substrate (1/h)
        "s_f": 600.0,      # Highly concentrated feed substrate solution stream (g/L)
        "F_loss": 2.5e-4,  # Constant evaporative volumetric loss rate (L/h) at 25°C
        
        # --- Physical Hard Operational Actuator / Tracker Constraints ---
        # u1: Feed Flow Rate F (L/h)
        "u_1_hard_min": 0.0,   # Valve fully closed
        "u_1_hard_max": 2.0,   # Maximum physical volumetric pump capacity (L/h)
        
        # y1: Specific Growth Rate mu (1/h)
        "y_1_hard_min": 0.0,
        "y_1_hard_max": 0.15,  # Upper biological thermodynamic limit
        
        # y2: Penicillin Product Concentration P (g/L)
        "y_2_hard_min": 0.0,
        "y_2_hard_max": 40.0,  # Saturation limit before extreme viscosity degradation
        
        # --- Normal Operating Center Bounds for Initial Trajectory Exploration ---
        # Dictates nominal operating flow bounds for steady fed-batch maintenance
        "u_1_D_center_min": 0.04,  
        "u_1_D_center_max": 0.06,  
    },
    "mamba": {
        "d_model": 64,
        "d_state": 16,
        "input_dim": 2,   # Tracking inputs: [mu, P]
        "output_dim": 1   # Control actions computed: [F]
    },
    "simulate": {
        "batch_size": 10,
        "seq_len": 351,
    }
}

hyperparam_config_IdiophasePlant = {
        "signal": {
            "seq_len": 51,
            "dt": 0.1,
            #/ 🕹️ Channel 1 Signal Parameters (e.g., highly dynamic)
            "u_1_lambd": 10,        
            "u_1_p": 0.0999,            
            
            #// 🕹️ Channel 2 Signal Parameters (e.g., highly filtered, slow moving)
            "u_2_lambd": 5.0,        
            "u_2_p": 0.29
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
            "m_S": 23,

            "x10": 1000,
            "x20": 2000,
            "x30": 25,
            "x40": 1600,

            "x_1_hard_min": 0.0,
            "x_1_hard_max": None,

            "x_2_hard_min": 0.0,
            "x_2_hard_max": None,

            "x_3_hard_min": 0.0,
            "x_3_hard_max": None,

            "x_4_hard_min": 0.0,
            "x_4_hard_max": None,
            
            "u_1_hard_min": 0.0,
            "u_1_hard_max": 1.0,

            "u_2_hard_min": 0.0,
            "u_2_hard_max": 1.0,

            "y_1_hard_min": 0.0,
            "y_1_hard_max": None,

            "y_2_hard_min": 0.0,
            "y_2_hard_max": None,

            "u_1_D_center_min": 0.001,
            "u_1_D_center_max": 0.999,

            "u_2_D_center_min": 0.3,
            "u_2_D_center_max": 0.7,
        },
        "mamba": {
            "expand": 32,
            "d_state": 16,
            "input_dim": 2,  # y1, y2
            "output_dim": 2  # u1, u2
        },
        "simulate": {
            "batch_size": 10,
            "seq_len": 101,
        }
    }

hyperparam_config_MassSpringDamperPlant = {
    "plant" :{
        "m": 0.1,
        "d": 0.1,
        "k": 1.0,
        "u_1_D_center_min": 0.5,
        "u_1_D_center_max": 0.5,
        "u_1_hard_min": 0,
        "u_1_hard_max": 1.0
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
        
        "u_1_D_center_min": 0.6,
        "u_1_D_center_max": 0.9,

        "u_1_hard_min": 0.0,
        "u_1_hard_max": 1,

        "x_1_hard_min": 0,
        "x_1_hard_max": None,

        "y_1_hard_min": 0,
        "y_1_hard_max": 0.12,

        "x10": 1600.0,   #wenn trainiert mit 1500 aber getesttet mit 1600, gute performnce
        "x20": 2000.0,

        "input_dim": 1,  # y
        "output_dim": 1  # u
    },
    "signal": {
        "lambd": 4,
        "p": 0.5,
        "seq_len": 2001,
        "dt": 0.01
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
    },
    "mamba": {
        "expand": 32,
        "d_state": 16,
        "input_dim": 1,  # y
        "output_dim": 1  # u
    },
    "simulate": {
        "batch_size": 10,
        "seq_len": 2001,
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