import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

plt.style.use("src/sample/style.mplstyle")

# Define file names, labels, and B&W friendly configurations
# We use different line styles and markers so they are distinct in grayscale
datasets = {
    'src/hfco.csv': {
        'label': 'HFCO-1233zd(E)', 
        'color': '#000000',          # Pure Black
        'style_left': '-',           # Solid
        'style_right': '-.',         # Dashed
        'marker': None
    },  
    'src/hcf.csv':  {
        'label': 'HFC-152',  
        'color': '#e66101',          # Dark Grey
        'style_left': '-',          # Dash-dot
        'style_right': ':',          # Dotted
        'marker': None
    },  
    'src/neopentane.csv': {
        'label': 'Neopentane', 
        'color': '#5e3c99',          # Medium Grey
        'style_left': '-',           # Solid with markers to differentiate from HFCO
        'style_right': '--',         # Dashed with markers to differentiate from HFCO
        'marker': 'o',               # Circle markers
        'markevery': 10              # Only show marker every 10 points so it's not crowded
    }   
}

fig, ax = plt.subplots(figsize=(7, 7))

# 1. Plot all the data
for filename, config in datasets.items():
    try:
        df = pd.read_csv(filename, skiprows=[1])
        
        t_left = pd.to_numeric(df['Température'])
        s_left = pd.to_numeric(df['Entropie molaire.1'])   
        t_right = pd.to_numeric(df['Température.1'])
        s_right = pd.to_numeric(df['Entropie molaire.3']) 
        
        # Determine marker settings if applicable
        marker = config.get('marker', None)
        markevery = config.get('markevery', 10)
        
        # Plot left branch
        ax.plot(s_left, t_left, 
                linestyle=config['style_left'], 
                color=config['color'], 
                marker=marker, 
                markevery=markevery,
                linewidth=2,
                label=f"{config['label']} (Liquid)")
        
        # Plot right branch
        ax.plot(s_right, t_right, 
                linestyle=config['style_right'], 
                color=config['color'], 
                marker=marker, 
                markevery=markevery,
                linewidth=2,
                label=f"{config['label']} (Vapor)")
        
    except FileNotFoundError:
        print(f"File {filename} not found. Skipping...")

# 2. Calculate and set the aspect ratio (outside the loop)
asp = 1.0  
x_range = np.diff(ax.get_xlim())[0]
y_range = np.diff(ax.get_ylim())[0]
range_ratio = x_range / y_range
ax.set_aspect(asp * range_ratio)

# Labeling and styling (With upright roman units)
ax.set_xlabel(r'Molar Entropy [$\mathrm{J \cdot mol^{-1} \cdot K^{-1}}$]')
ax.set_ylabel(r'Temperature [$\mathrm{K}$]')
ax.grid(True, linestyle=':', alpha=0.4)

# Legend placed perfectly in the top left corner
ax.legend(loc='upper left', frameon=True) 

plt.tight_layout()

# Save and display
plt.savefig('combined_ts_diagram_bw.png', dpi=300)
plt.show()