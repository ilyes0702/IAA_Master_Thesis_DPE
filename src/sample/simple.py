import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os

plt.style.use("src/sample/style.mplstyle")

# 1. Load Data - Hardcoded to skip the first 2 rows of headers/units
def load_csv_fixed(path):
    # skiprows=2 ignores the text headers and unit definitions
    # names=[...] provides clean internal headers for the numeric columns
    return pd.read_csv(path, skiprows=2, header=None)

df_hcf = load_csv_fixed('src/hcf.csv')
df_neopentane = load_csv_fixed('src/neopentane.csv')
df_hfco = load_csv_fixed('src/hfco.csv')

output_dir = "reports/thermodynamics"
os.makedirs(output_dir, exist_ok=True)

fluid_colors = {
    "HFC-152": "#1f77b4",
    "Neopentane": "#ff7f0e",
    "HFCO-1233zd(E)": "#2ca02c"
}

fig, ax = plt.subplots(figsize=(12, 8))

datasets = [
    ("HFC-152", df_hcf),
    ("Neopentane", df_neopentane),
    ("HFCO-1233zd(E)", df_hfco)
]

for name, df in datasets:
    color = fluid_colors[name]
    
    # Liquid Phase: Col 2 (index 1) over Col 3 (index 2)
    # x = index 2 (Entropy), y = index 1 (Temp)
    ax.plot(df.iloc[:, 2].dropna(), df.iloc[:, 1].dropna(), 
            color=color, linestyle='-', linewidth=2, label=name)
    
    # Vapor Phase: Col 5 (index 4) over Col 6 (index 5)
    # x = index 5 (Entropy), y = index 4 (Temp)
    ax.plot(df.iloc[:, 5].dropna(), df.iloc[:, 4].dropna(), 
            color=color, linestyle='--', linewidth=2)

# === TICK OPTIMIZATION ===
# Now that data is numeric, MaxNLocator will produce clean numbers (300, 310, etc.)
ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=10))
ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=10))
ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())

# === FIXED ASPECT RATIO ===
x_left, x_right = ax.get_xlim()
y_bottom, y_top = ax.get_ylim()
ax.set_aspect((x_right - x_left) / (y_top - y_bottom))

# Formatting
ax.set_xlabel(r"Molar Entropy [$J \cdot mol^{-1} \cdot K^{-1}$]", fontsize=13)
ax.set_ylabel(r"Temperature [K]", fontsize=13)
ax.grid(True, which='major', linestyle='-', alpha=0.4)
ax.legend(title="")

save_path = os.path.join(output_dir, "Combined_TS_Diagram_Clean.png")
plt.savefig(save_path, dpi=300, bbox_inches="tight")
plt.show()