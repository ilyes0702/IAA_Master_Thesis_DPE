# IMPORT SYSTEM MODULES
import sys
import os
import seaborn as sns

from src.sample.utils.saving_utils import save_plot_image
# ADD PROJECT DIRECTORY TO SYSTEM PATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# IMPORT DATA PROCESSING MODULES
import numpy as np
import pandas as pd
from matplotlib.colors import to_hex
from matplotlib import cm

# IMPORT VISUALIZATION MODULES
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patches as patches
from PIL import Image
from io import BytesIO

# IMPORT MACHINE LEARNING MODULES
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import PredictionErrorDisplay

# IMPORT CUSTOM MODULES
from src.sample.decorators.general_decorators import *

import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image


@save_image_decorator
def plot_signals_flexible(
    t,                # Can be a single array OR a list of arrays
    signals,          # List of arrays
    labels=None,
    title=None,
    xlabel="Time",
    ylabel="Value",
    figsize=(10, 5),
    save_path=None,
    show=False,
    filename=None,
    dirname=None
):
    fig, ax = plt.subplots(figsize=figsize, layout="constrained")

    for i, sig in enumerate(signals):
        # Determine which x-axis to use for this specific signal
        # If t is a list, take the i-th element; otherwise, use t for all.
        current_t = t[i] if isinstance(t, list) else t
        
        if labels is not None:
            ax.plot(current_t, sig, label=labels[i])
        else:
            ax.plot(current_t, sig)

    # === FIXED ASPECT RATIO === #
    # Note: If x and y scales are vastly different, this can make the plot very thin.
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()
    x_range = x_max - x_min
    y_range = y_max - y_min
    if y_range != 0:
        ax.set_aspect(x_range / y_range)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if title:
        ax.set_title(title)
    if labels:
        ax.legend()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()

    # Save to buffer and return as PIL Image
    buf = BytesIO()
    plt.savefig(buf, format="PNG", dpi=600)
    buf.seek(0)
    plt.close()
    return Image.open(buf)

#=== FUNCTION TO PLOT SIGNALS ===#
def plot_signals(
    t,
    signals,
    labels=None,
    title=None,
    xlabel="Time",
    ylabel="Value",
    figsize=(10, 5),
    save_path=None,
    show=False,
    filename=None,
    dirname=None
):
    """
    Generic plotting function for time-series or x-y signals.
    """

    fig, ax = plt.subplots(figsize=figsize, layout= "constrained")

    for i, sig in enumerate(signals):
        if labels is not None:
            ax.scatter(t, sig, label=labels[i])
        else:
            ax.scatter(t, sig)

    # === FIXED ASPECT RATIO === #
    x_range = np.diff(ax.get_xlim())[0]
    y_range = np.diff(ax.get_ylim())[0]
    ax.set_aspect(x_range / y_range)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if title:
        ax.set_title(title)

    if labels:
        ax.legend()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight", pad_inches=0.05)

    if show:
        plt.show()

    
    #plt.tight_layout()

    # Save the plot to a buffer
    buf = BytesIO()
    plt.savefig(buf, format="PNG", dpi=600)
    buf.seek(0)
    plt.close()

    # Convert the buffer to a PIL Image
    image = Image.open(buf)
    save_plot_image(image=image, filename=filename, dirname=dirname)
    return image




