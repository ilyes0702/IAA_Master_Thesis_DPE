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
            ax.plot(t, sig, label=labels[i])
        else:
            ax.plot(t, sig)

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




#=== FUNCTION TO PLOT 2D MISCLASSIFICATION PLOTS ===#
@save_image_decorator
def create_misclassification_plot(y_true, y_pred, X, dirname, filename):
    """
    Generate a color-coded scatter plot showing misclassifications with the categories:
    False Negative (FN), False Positive (FP), True Positive (TP), True Negative (TN).

    Parameters:
        - y_true (array-like): Ground truth labels.
        - y_pred (array-like): Predicted labels.
        - X (pd.DataFrame): Feature data for the scatter plot (must have at least two columns for plotting).
        - dirname (str): Directory where the plot image will be saved.
        - filename (str): Filename for storing the generated plot.

    Returns:
        - PIL.Image: The generated misclassification plot as a PIL Image.
    """
    # Ensure X has at least two columns for plotting
    if X.shape[1] < 2:
        raise ValueError("X must have at least two columns for plotting.")

    # Define categories and their colors
    categories = {
        "True Positive (TP)": {"condition": (y_true == 1) & (y_pred == 1), "color": "green"},
        "True Negative (TN)": {"condition": (y_true == -1) & (y_pred == -1), "color": "blue"},
        "False Positive (FP)": {"condition": (y_true == -1) & (y_pred == 1), "color": "orange"},
        "False Negative (FN)": {"condition": (y_true == 1) & (y_pred == -1), "color": "red"},
    }

    # Create the scatter plot
    fig, ax = plt.subplots(figsize=(8, 6))
    for category, props in categories.items():
        condition = props["condition"]
        color = props["color"]
        ax.scatter(
            X.loc[condition, X.columns[0]],
            X.loc[condition, X.columns[1]],
            label=category,
            color=color,
            alpha=0.7,
            s=50,
        )

    # Customize the plot
    ax.set_xlabel(X.columns[0])
    ax.set_ylabel(X.columns[1])
    ax.legend(loc="upper right", fontsize="small")
    
    ax.set_aspect(np.diff(ax.get_xlim()) / np.diff(ax.get_ylim()))

    from src.sample.utils.general_utils import apply_axis_label_mapping
    apply_axis_label_mapping(ax)
    plt.tight_layout()

    # Save the plot to a buffer
    buf = BytesIO()
    plt.savefig(buf, format="PNG", dpi=600)
    buf.seek(0)
    plt.close()

    # Convert the buffer to a PIL Image
    image = Image.open(buf)

    return image




#=== FUNCTION TO CREATE INDIVIDUAL SCATTER PLOTS ===#           
def create_individual_scatter_plots(ds, features, label_, color_by_A12=False, color_by_label=False):
    """
    Generate individual scatter plots for all pairs of features with scales from 0 to 1 and 0.2 ticks.

    Parameters:
    - df (pd.DataFrame): The DataFrame containing the data.
    - features (list): List of feature column names to plot.
    - label_DIDF (str): Column name for the DI/DF label.
    - color_by_A12 (bool): Whether to color code plots involving 'C3' based on the value of 'A12'.
    - color_by_label (bool): Whether to color code plots based on the value of label_DIDF.

    Returns:
    None
    """
    # Define color mapping for DI and DF
    color_map = {-1: "red", 1: "green"}  # DI = red, DF = green

    # Iterate over all pairs of features
    for i, feature_x in enumerate(features):
        for j, feature_y in enumerate(features):
            if i != j:  # Avoid plotting a feature against itself
                fig, ax = plt.subplots()

                if color_by_A12 and ("C3" in [feature_x, feature_y]) and "A12" in ds.df.columns:
                    # Color code based on A12 values
                    scatter = ax.scatter(
                        ds.df[feature_x],
                        ds.df[feature_y],
                        c=ds.df["A12"],
                        cmap="viridis",
                        s=1,
                        alpha=0.7
                    )
                    cbar = plt.colorbar(scatter, ax=ax)
                    cbar.set_label("A12")
                elif color_by_label:
                    # Scatter plot with color coding for DI/DF
                    for label, color in color_map.items():
                        subset = ds.df[ds.df[label_] == label]
                        ax.scatter(
                            subset[feature_x],
                            subset[feature_y],
                            s=1,
                            alpha=0.4,
                            label=f"{label_} = {label}",
                            color=color
                        )
                else:
                    # Scatter plot without color coding
                    ax.scatter(
                        ds.df[feature_x],
                        ds.df[feature_y],
                        s=1,
                        alpha=0.4
                    )

                # Add quadrant lines
                xlim = ax.get_xlim()
                ylim = ax.get_ylim()
                if xlim[0] < 0 < xlim[1]:
                    ax.axvline(x=0, color="black", linestyle="--", linewidth=1)
                if ylim[0] < 0 < ylim[1]:
                    ax.axhline(y=0, color="black", linestyle="--", linewidth=1)

                ax.set_xlabel(feature_x)
                ax.set_ylabel(feature_y)
                from src.sample.utils.general_utils import apply_axis_label_mapping
                apply_axis_label_mapping(ax)

                # Add background regions
                if ylim[0] < 0 < ylim[1]:
                    ax.axhspan(
                        xmin=0,
                        xmax=ax.get_xlim()[1],
                        ymin=0,
                        ymax=ax.get_ylim()[1],
                        facecolor="grey",
                        alpha=0.3,
                        label="Positive Region (Y)"
                    )
                    ax.axhspan(
                        xmin=0,
                        xmax=ax.get_xlim()[1],
                        ymax=0,
                        ymin=ax.get_ylim()[0],
                        facecolor="white",
                        alpha=0.3,
                        label="Negative Region (Y)"
                    )

                ax.set_aspect(np.diff(ax.get_xlim()) / np.diff(ax.get_ylim()))

                # Save the plot
                filename = f"images/{date}/{date_and_time}/{ds.name}/dataset_visualizations/scatter_plots/{date_and_time}_scatter_plot_{feature_y}_vs_{feature_x}.png"
                os.makedirs(os.path.dirname(filename), exist_ok=True)
                plt.savefig(filename, dpi=300)
                plt.close(fig)


@save_image_decorator
def plot_learning_curve(model, ds, dirname, filename, num_epochs=50):
    """
    Trains the given model over a specified number of epochs and plots the learning curve.

    Parameters:
        - model - The machine learning model to train.
        - ds - An object containing training and testing data (X_train, y_train, X_test, y_test).
        - num_epochs - Number of epochs for training (default is 50).
        - save_path - Path to save the learning curve plot (default is "Learning_curve.png").
    """
    from math import sqrt
    from sklearn.metrics import mean_squared_error
    # Lists to store training and validation losses
    train_losses = []
    val_losses = []
    
    epochs = range(1, num_epochs + 1)
    for epoch in epochs:
        model.fit(ds.X_train, ds.y_train)
        
        # Predict and calculate the training loss
        train_pred = model.predict(ds.X_train)
        train_loss = sqrt(mean_squared_error(ds.y_train, train_pred))
        train_losses.append(train_loss)
        
        # Predict and calculate the validation loss
        val_pred = model.predict(ds.X_test)
        val_loss = sqrt(mean_squared_error(ds.y_test, val_pred))
        val_losses.append(val_loss)
    
    # Plot the learning curve
    # Create figure and axes
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(epochs, train_losses, label='Training RMSE')
    ax.plot(epochs, val_losses, label='Validation RMSE', linestyle='--')
    ax.set_aspect(np.diff(ax.get_xlim()) / np.diff(ax.get_ylim()))
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Root Mean Squared Error (RMSE)')
    ax.legend()

    plt.tight_layout()
    
    # Save the plot to a buffer
    buf = BytesIO()
    plt.savefig(buf, format='PNG', dpi=600)
    buf.seek(0)
    plt.close()
    
    # Convert the buffer to a PIL Image
    image = Image.open(buf)

    return image

#=== FUNCTION TO CREATE 3D SCATTER PLOT ===#
@save_image_decorator
def create3DScatterPlot(df, label_name, class_names, dirname, filename):
    """
    Creates a 3D scatter plot of feature vectors, visualizing data points with different class labels.

    The function takes a dataset containing three features (A1, A2, A3) and class labels, 
    then generates a 3D scatter plot with different colors representing distinct classes.

    Parameters:
        df (pd.DataFrame): DataFrame containing feature vectors and class labels.
                           Must include columns ["A1", "A2", "A3"] for plotting.
        label_name (str): Column name of the class labels in the DataFrame.
        class_names (list): A list of class names corresponding to unique labels (binary classification assumed).
                            Example: `["Class 0", "Class 1"]`
        dirname (str): Directory where the plot image should be saved.
        filename (str): Name of the file for saving the plot.

    Returns:
        PIL.Image: The generated 3D scatter plot as a PIL Image object, allowing further processing or saving.
    """
    
    x = df["A1"].to_numpy()
    y = df["A2"].to_numpy()
    z = df["A3"].to_numpy()
    labels = df[label_name].to_numpy()
    labels = [1 if x == 1 else 0 for x in labels]
    # Creating figure
    fig = plt.figure(figsize = (10, 7))
    ax = fig.add_subplot(111, projection ="3d")
    
    # Add x, y gridlines
    ax.grid(b = True, color ='grey',
            linestyle ='-.', linewidth = 0.3,
            alpha = 0.2)
    
    ax.set_xlabel('A1', fontweight ='bold')
    ax.set_ylabel('A2', fontweight ='bold')
    ax.set_zlabel('A3', fontweight ='bold')

    from src.sample.utils.general_utils import apply_axis_label_mapping
    apply_axis_label_mapping(ax)    
    # Mapping class labels to custom names
    label_mapping = {0: class_names[0], 1: class_names[1]}
    # Create a colormap for the classes
    colors = ['red', 'green']
    for label in np.unique(labels):
        ax.scatter(x[labels == label], y[labels == label], z[labels == label],
               label=f'{label_mapping[label]}', color=colors[label])

    # Display legend outside the plot for better visibility
    ax.legend(loc='upper left', bbox_to_anchor=(0.75, 1))

    # Save the plot to a buffer
    buf = BytesIO()
    plt.savefig(buf, format='PNG', dpi=600)
    buf.seek(0)
    plt.close()
    
    # Convert the buffer to a PIL Image
    image = Image.open(buf)

    return image

#=== FUNCTION TO SAVE HISTOGRAM AND KDE ===#
@save_image_decorator
def plot_histogram_and_kde(df, feature,  dirname, filename, title=None,bins=30, kde=True):
    """
    Plot the histogram and KDE of a feature in a dataset.

    Parameters:
        df (pd.DataFrame): The dataset containing the feature.
        feature (str): The name of the feature to plot.
        bins (int): Number of bins for the histogram. Default is 30.
        kde (bool): Whether to include the KDE plot. Default is True.
        title (str): Title of the plot. Default is None.
        save_path (str): Path to save the plot as an image. If None, the plot is not saved. Default is None.

    Returns:
        None
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plot histogram with KDE
    sns.histplot(df[feature], bins=bins, kde=kde, color="blue", alpha=0.6, edgecolor="black")
    
    # Add title and labels
    ax.set_xlabel(feature, fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_aspect(np.diff(ax.get_xlim())/np.diff(ax.get_ylim()))
    from src.sample.utils.general_utils import apply_axis_label_mapping
    apply_axis_label_mapping(ax)
    # Save the plot to a buffer
    buf = BytesIO()
    plt.savefig(buf, format='PNG', dpi=600)
    buf.seek(0)
    plt.close(fig)

    # Convert the buffer to a PIL Image
    image = Image.open(buf)

    return image

from matplotlib.path import Path


#=== FUNCTION TO CREATE PARETO PLOT (SIMPLE) ===#
@save_image_decorator
def createParetoPlot_simple(df, indexList, dirname, filename):
    """
    Create Pareto plot from dataframe and shade the hypervolume region.

    Parameters:
        df (DataFrame): The dataset containing the objective functions F1 and F2.
        indexList (list): List of indices representing the Pareto front.
        dirname (str): The directory name to save the plot.
        filename (str): The base name of the file to save the plot.

    Returns:
        tuple: A tuple containing the Pareto plot as a PIL Image and the hypervolume area.
    """
    import matplotlib.patches as patches
    from matplotlib.path import Path

    fig, ax = plt.subplots()

    # Plot all points
    ax.scatter(df["F1"], df["F2"], color="blue", s=15, label="Points")

    # Highlight Pareto front
    pareto_scatter = ax.scatter(
        df.loc[indexList, "F1"], df.loc[indexList, "F2"],
        label="Pareto front", s=50, marker="*", color="black"
    )

    # Calculate and plot the ideal objective vector
    ideal_obj = df[["F1", "F2"]].min()
    ax.scatter(ideal_obj["F1"], ideal_obj["F2"], color='green', marker='o', s=100, label='Ideal Objective Vector')

    # Calculate and plot the nadir point
    nadir_point = df[["F1", "F2"]].max()
    ax.scatter(nadir_point["F1"], nadir_point["F2"], color='red', marker='o', s=100, label='Nadir Point')

    # Calculate the points (x_ideal, y_nadir) and (x_nadir, y_ideal)
    x_ideal = ideal_obj["F1"]
    y_nadir = nadir_point["F2"]
    x_nadir = nadir_point["F1"]
    y_ideal = ideal_obj["F2"]

    point1 = (x_ideal, y_nadir)
    point2 = (x_nadir, y_ideal)

    # Get Pareto front points
    pareto_points = df.loc[indexList, ["F1", "F2"]].values

    # Sort Pareto front points by F1
    pareto_points = pareto_points[np.argsort(pareto_points[:, 0])]

    # Construct the polygon in the correct sequence
    hypervolume_points = np.vstack([
        [nadir_point.values],   # Step 1: Start at nadir point
        # Step 2: Move to 
        [x_nadir, y_ideal],
        pareto_points[::-1],        # Step 3: Follow Pareto front from lower to upper end
        [x_ideal, y_nadir],     # Step 4: Go to (x_ideal, y_nadir)
        [nadir_point.values]    # Step 5: Return to nadir point
    ])

    # Create a Path object and a PathPatch from the hypervolume points
    path = Path(hypervolume_points)
    patch = patches.PathPatch(path, facecolor='gray', alpha=0.5, label='Hypervolume Region')

    # Add the patch to the plot
    ax.add_patch(patch)
    ax.set_aspect(np.diff(ax.get_xlim())/np.diff(ax.get_ylim()))
    # Create legend
    #ax.legend(loc='upper right')
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax.set_xlabel("-PESR [%]")
    ax.set_ylabel("LCOE [USD/MWh]")

    plt.tight_layout()

    # Save the plot to a buffer
    buf = BytesIO()
    plt.savefig(buf, format='PNG', dpi=600)
    buf.seek(0)
    plt.close()

    # Convert the buffer to a PIL Image
    image = Image.open(buf)

    return image

#=== FUNCTION TO CREATE PARETO PLOT ===#
@save_image_decorator
def createParetoPlot(df,indexList, color_column, dirname, filename):
    """
    Create Pareto plot from dataframe, color-coded by a specified column.

    Parameters:
        df (DataFrame) : The dataset containing the objective functions F1 and F2.
        dirname (str) : The directory name to save the plot.
        filename (str) : The base name of the file to save the plot.
        color_column (str) : The column to use for color-coding the points.

    Returns:
        image (PIL Image) : The Pareto plot as a PIL Image.
    """
    #df = df.reset_index(drop=True)

    # Create a mapping for titles based on the color_column argument
    title_mapping = {
        "A8": "Variable I",
        "A9": "Variable II",
        "A10": "Heating with turbine bleeding",
        "A11": "Regeneration",
        "A12": "Fluid variable",
        "A13": "Closed bleeding",
    }

    # Fetch the title based on color_column
    plot_title = title_mapping.get(color_column, f"Pareto Plot by {color_column.capitalize()}")
    # Extract unique values
    unique_values = np.unique(df[color_column])
    num_values = len(unique_values)

    # Use tab10 or tab20 colormap based on number of categories
    cmap = plt.cm.get_cmap("tab10" if num_values <= 10 else "tab20", num_values)

    # Create a dictionary mapping unique values to color indices
    color_dict = {val: idx for idx, val in enumerate(unique_values)}

    # Assign colors dynamically based on values
    colors = [color_dict[val] for val in df[color_column]]

    # Compute Pareto front

    # Calculate padding and set limits based on Pareto front
    padding_factor = 0.1
    x_min = df.loc[indexList, "F1"].min()
    x_max = df.loc[indexList, "F1"].max()
    y_min = df.loc[indexList, "F2"].min()
    y_max = df.loc[indexList, "F2"].max()

    x_padding = padding_factor * (x_max - x_min)
    y_padding = padding_factor * (y_max - y_min)

    x_min -= x_padding
    x_max += x_padding
    y_min -= y_padding
    y_max += y_padding

    # Plot
    fig, ax = plt.subplots()
    scatter = ax.scatter(df["F1"], df["F2"], c=colors, cmap=cmap, s=15)

    # Highlight Pareto front
    pareto_scatter = ax.scatter(
        df.loc[indexList, "F1"], df.loc[indexList, "F2"],
        label="Pareto front", s=50, marker="*", color="black"
    )

    # Create legend dynamically
    legend_patches = [patches.Patch(color=cmap(idx / num_values), label=f"Value = {val}") for val, idx in color_dict.items()]
    ax.legend(handles=legend_patches + [pareto_scatter], loc='upper right')

    # Set limits based on Pareto front with padding
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    ax.set_xlabel("F1")
    ax.set_ylabel("F2")
    # Set title
    # ax.set_title(plot_title)
    ax.set_aspect(np.diff(ax.get_xlim())/np.diff(ax.get_ylim()))
    plt.tight_layout()
    from src.sample.utils.general_utils import apply_axis_label_mapping
    apply_axis_label_mapping(ax)
    # Save the plot to a buffer
    buf = BytesIO()
    plt.savefig(buf, format='PNG', dpi=600)
    buf.seek(0)
    plt.close()

    # Convert the buffer to a PIL Image
    image = Image.open(buf)

    return image


@save_image_decorator
def create_multiple_pareto_curves_new(
    datasets, dirname, filename,
    color_by_flow_rate=False, color_by_structure=False,
    legend_loc='upper right',
    global_structure_color_map=None,
    global_flow_bounds=None
):
    """
    Plot Pareto fronts from multiple Dataset objects on the same plot with consistent coloring across sessions.

    Parameters:
        datasets (list): List of Dataset objects with .pareto_front_df attribute.
        dirname (str): Directory to save the figure.
        filename (str): File name for the saved image.
        color_by_flow_rate (bool): If True, color points by flow rate.
        color_by_structure (bool): If True, color points by unique structures.
        legend_loc (str): Location of the legend.
        global_structure_color_map (dict): Persistent map of structures to colors.
        global_flow_bounds (dict): Persistent min/max bounds for flow rate coloring.

    Returns:
        PIL.Image.Image: The rendered plot.
    """
    import os
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    from matplotlib.patches import Patch
    from matplotlib.colors import to_hex
    from src.sample.utils.general_utils import apply_axis_label_mapping
    from io import BytesIO
    from PIL import Image

    if global_structure_color_map is None:
        global_structure_color_map = {}
    if global_flow_bounds is None:
        global_flow_bounds = {'min': None, 'max': None}

    fig, ax = plt.subplots(figsize=(8, 6))
    markers = ['o', 's', 'D', '^', 'v', 'P', '*', 'X', 'h']
    color_palette = cm.tab20
    color_index = len(global_structure_color_map)

    if color_by_flow_rate and (global_flow_bounds['min'] is None or global_flow_bounds['max'] is None):
        all_flows = pd.concat([ds.pareto_front_df["Fluid (kg/s) "] for ds in datasets])
        global_flow_bounds['min'] = all_flows.min()
        global_flow_bounds['max'] = all_flows.max()

    norm = None
    if color_by_flow_rate:
        cmap = plt.get_cmap("viridis")
        norm = plt.Normalize(global_flow_bounds['min'], global_flow_bounds['max'])

    structure_patches = []
    scatter_obj = None

    for idx, ds in enumerate(datasets):
        df = ds.pareto_front_df.reset_index(drop=True)

        if color_by_structure:
            for _, row in df.iterrows():
                structure = tuple(row[ds.actvar_struct])
                if structure not in global_structure_color_map:
                    global_structure_color_map[structure] = to_hex(color_palette.colors[color_index % len(color_palette.colors)])
                    color_index += 1

                ax.scatter(
                    row["F1"], row["F2"],
                    color=global_structure_color_map[structure],
                    s=20,
                    marker=markers[idx % len(markers)],
                    alpha=0.7
                )

            df_sorted = df.sort_values(by="F1")
            ax.plot(
                df_sorted["F1"], df_sorted["F2"],
                color='grey', linewidth=1, alpha=0.4, zorder=1
            )

        elif color_by_flow_rate:
            scatter = ax.scatter(
                df["F1"], df["F2"],
                c=df["Fluid (kg/s) "],
                cmap=cmap, norm=norm,
                s=50, marker=markers[idx % len(markers)], alpha=0.7,
                label=f"{ds.name} (Pareto Front)"
            )
            scatter_obj = scatter

        else:
            ax.scatter(
                df["F1"], df["F2"],
                label=f"{ds.name} (Pareto Front)",
                s=50, marker=markers[idx % len(markers)], alpha=0.7
            )

        representative_point = df.iloc[0]
        ax.annotate(
            ds.name,
            xy=(representative_point["F1"], representative_point["F2"]),
            xytext=(0, 10),
            textcoords="offset points",
            fontsize=10,
            color='black'
        )

    if color_by_structure:
        for structure, color in global_structure_color_map.items():
            # Convert each value in the structure to an integer, increment the last one by 1
            modified_structure = list(structure)
            modified_structure[-1] = int(modified_structure[-1]) + 1  # Increment the last element by 1
            label = "-".join(str(int(val)) for val in modified_structure)
            structure_patches.append(Patch(color=color, label=label))

        ax.legend(handles=structure_patches, loc=legend_loc, title="Unique structure-fluid combinations")

    elif color_by_flow_rate and scatter_obj:
        cbar = plt.colorbar(scatter_obj, ax=ax)
        cbar.set_label("Fluid flow rate [kg/s]")

    elif not color_by_structure:
        ax.legend(loc=legend_loc)

    ax.set_xlabel("F1")
    ax.set_ylabel("F2")
    apply_axis_label_mapping(ax)

    ax.grid(False)
    ax.set_aspect(np.diff(ax.get_xlim()) / np.diff(ax.get_ylim()))
    plt.tight_layout()

    os.makedirs(dirname, exist_ok=True)
    buf = BytesIO()
    plt.savefig(buf, format='PNG', dpi=600, bbox_inches='tight')
    buf.seek(0)
    plt.close()

    image = Image.open(buf)
    return image


@save_image_decorator
def create_multiple_pareto_curves(datasets, dirname, filename, color_by_flow_rate=False, color_by_structure=False, legend_loc='upper right', global_structure_color_map=None):
    """
    Plot Pareto fronts from multiple Dataset objects on the same plot, using their precomputed pareto_front_df attribute.

    Parameters:
        datasets (list of Dataset objects): Each must have a .pareto_front_df attribute with columns 'F1' and 'F2'.
        dirname (str): Directory path where the plot will be saved.
        filename (str): Filename of the saved plot image.
        color_by_flow_rate (bool, optional): If `True`, colors points based on flow rate values.
        color_by_structure (bool, optional): If `True`, colors points based on unique structures.
        legend_loc (str, optional): Location of the legend. Defaults to 'lower right'.

    Returns:
        PIL.Image: The generated plot as a PIL Image.
    """
    from src.sample.utils.general_utils import apply_axis_label_mapping
    import matplotlib.cm as cm
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from io import BytesIO
    from PIL import Image
    import pandas as pd
    import numpy as np
    if global_structure_color_map is None:
            global_structure_color_map = {}
    fig, ax = plt.subplots(figsize=(8, 6))
    markers = ['o', 's', 'D', '^', 'v', 'P', '*', 'X', 'h']
    # global_structure_color_map = {}
    color_palette = cm.tab20
    color_index = 0

    if color_by_flow_rate:
        cmap = plt.get_cmap("viridis")

    for idx, ds in enumerate(datasets):
        df = ds.pareto_front_df.reset_index(drop=True)

        if color_by_structure:
            structure_counts = df[ds.actvar_struct].value_counts()
            repeated_structures = structure_counts[structure_counts > 1].index
            unique_structures = pd.DataFrame(list(repeated_structures), columns=ds.actvar_struct)

            for _, structure_row in unique_structures.iterrows():
                structure = tuple(structure_row)
                if structure not in global_structure_color_map:
                    global_structure_color_map[structure] = to_hex(color_palette(color_index / 20))
                    color_index += 1

            for _, row in df.iterrows():
                structure = tuple(row[ds.actvar_struct])
                if structure in global_structure_color_map:
                    ax.scatter(
                        row["F1"], row["F2"],
                        color=global_structure_color_map[structure],
                        s=20,
                        marker=markers[idx % len(markers)],
                        alpha=0.7
                    )

            structure_patches = []
            for structure, color in global_structure_color_map.items():
                label = "-".join(str(int(val)) for val in structure)
                structure_patches.append(Patch(color=color, label=label))

            ax.legend(handles=structure_patches, loc=legend_loc, title="Unique structure-fluid combinations")

            df_sorted = df.sort_values(by="F1")
            ax.plot(
                df_sorted["F1"],
                df_sorted["F2"],
                color='grey',
                linewidth=1,
                alpha=0.4,
                zorder=1
            )

        elif color_by_flow_rate:
            norm = plt.Normalize(df["Fluid (kg/s) "].min(), df["Fluid (kg/s) "].max())
            scatter = ax.scatter(
                df["F1"], df["F2"],
                c=df["Fluid (kg/s) "],
                cmap=cmap, norm=norm, s=50, marker=markers[idx % len(markers)], alpha=0.7,
                label=f"{ds.name} (Pareto Front)"
            )
        else:
            ax.scatter(
                df["F1"], df["F2"],
                label=f"{ds.name} (Pareto Front)",
                s=50, marker=markers[idx % len(markers)], alpha=0.7
            )

        # Annotate the dataset name next to the curve
        representative_point = df.iloc[0]
        ax.annotate(
            ds.name,
            xy=(representative_point["F1"], representative_point["F2"]),
            xytext=(-5, 0),
            textcoords="offset points",
            fontsize=10,
            color='black'
        )

    if color_by_flow_rate:
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label("Fluid flow rate [kg/s]")

    ax.set_xlabel("F1")
    ax.set_ylabel("F2")

    if not color_by_structure and not color_by_flow_rate:
        ax.legend(loc=legend_loc)

    plt.grid(False)
    ax.set_aspect(np.diff(ax.get_xlim()) / np.diff(ax.get_ylim()))
    plt.tight_layout()
    apply_axis_label_mapping(ax)

    buf = BytesIO()
    plt.savefig(buf, format='PNG', dpi=600, bbox_inches='tight')
    buf.seek(0)
    plt.close()

    image = Image.open(buf)
    return image

@save_image_decorator
def plot_multiple_pareto_subplots(
    dataset_groups, dirname, filename,
    color_by_flow_rate=False, color_by_structure=False,
    legend_loc='center left',  # Adjusted default location
    global_structure_color_map=None,
    global_flow_bounds=None
):
    """
    Plot Pareto fronts from multiple Dataset objects in subplots, using global coloring across subplots and calls.

    Parameters:
        dataset_groups (list of lists): Each inner list contains Dataset objects for one subplot.
        dirname (str): Directory to save the figure.
        filename (str): Name of the saved image file.
        color_by_flow_rate (bool): If True, use flow rate for coloring.
        color_by_structure (bool): If True, color by unique structures.
        legend_loc (str): Legend location.
        global_structure_color_map (dict): Maps structure tuples to colors.
        global_flow_bounds (dict): {min: float, max: float} bounds for flow rate normalization.

    Returns:
        PIL.Image.Image: The rendered figure.
    """
    import os
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    from matplotlib.patches import Patch
    from matplotlib.colors import to_hex
    from PIL import Image
    from io import BytesIO
    from src.sample.utils.general_utils import apply_axis_label_mapping
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    if global_structure_color_map is None:
        global_structure_color_map = {}

    if global_flow_bounds is None:
        global_flow_bounds = {'min': None, 'max': None}

    if color_by_flow_rate and (global_flow_bounds['min'] is None or global_flow_bounds['max'] is None):
        all_flows = []
        for group in dataset_groups:
            for ds in group:
                all_flows.append(ds.pareto_front_df["Fluid (kg/s) "])
        combined_flows = pd.concat(all_flows)
        global_flow_bounds['min'] = combined_flows.min()
        global_flow_bounds['max'] = combined_flows.max()

    norm = plt.Normalize(global_flow_bounds['min'], global_flow_bounds['max'])
    cmap = plt.get_cmap("viridis")
    color_palette = cm.tab20
    color_index = len(global_structure_color_map)
    markers = ['o', 's', 'D', '^', 'v', 'P', '*', 'X', 'h']

    fig, axes = plt.subplots(1, len(dataset_groups), figsize=(8 * len(dataset_groups), 6))

    if len(dataset_groups) == 1:
        axes = [axes]

    structure_patches = []
    scatter_obj = None

    for ax, group in zip(axes, dataset_groups):
        for idx, ds in enumerate(group):
            df = ds.pareto_front_df.reset_index(drop=True)

            if color_by_structure:
                for _, row in df.iterrows():
                    structure = tuple(row[ds.actvar_struct])
                    if structure not in global_structure_color_map:
                        global_structure_color_map[structure] = to_hex(color_palette(color_index / 20))
                        color_index += 1

                    ax.scatter(
                        row["F1"], row["F2"],
                        color=global_structure_color_map[structure],
                        s=30,
                        marker=markers[idx % len(markers)],
                        alpha=0.7
                    )

                df_sorted = df.sort_values(by="F1")
                ax.plot(
                    df_sorted["F1"], df_sorted["F2"],
                    color='grey', linewidth=1, alpha=0.4, zorder=1
                )

            elif color_by_flow_rate:
                scatter = ax.scatter(
                    df["F1"], df["F2"],
                    c=df["Fluid (kg/s) "],
                    cmap=cmap, norm=norm,
                    s=50, marker=markers[idx % len(markers)], alpha=0.7,
                    label=f"{ds.name} (Pareto Front)"
                )
                scatter_obj = scatter

            else:
                ax.scatter(
                    df["F1"], df["F2"],
                    label=f"{ds.name} (Pareto Front)",
                    s=50, marker=markers[idx % len(markers)], alpha=0.7
                )

            representative_point = df.iloc[0]
            ax.annotate(
                ds.name,
                xy=(representative_point["F1"], representative_point["F2"]),
                xytext=(-5, 0),
                textcoords="offset points",
                fontsize=10,
                color='black'
            )

        ax.set_xlabel("F1")
        ax.set_ylabel("F2")
        apply_axis_label_mapping(ax)

        if not color_by_structure and not color_by_flow_rate:
            ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1))

        ax.grid(False)
        ax.set_aspect(np.diff(ax.get_xlim()) / np.diff(ax.get_ylim()))

    if color_by_structure:
        for structure, color in global_structure_color_map.items():
            label = "-".join(str(int(val)) for val in structure)
            structure_patches.append(Patch(color=color, label=label))

        fig.legend(handles=structure_patches, loc='center left', bbox_to_anchor=(0.86, 0.5), title="Unique Structures")

    if color_by_flow_rate and scatter_obj:
        divider = make_axes_locatable(axes[-1])
        cax = divider.append_axes("right", size="5%", pad=0.1)
        fig.colorbar(scatter_obj, cax=cax).set_label("Fluid flow rate [kg/s]")

    fig.tight_layout(rect=[0, 0, 0.85, 1])  # leave room on the right

    buf = BytesIO()
    os.makedirs(dirname, exist_ok=True)
    full_path = os.path.join(dirname, filename)
    plt.savefig(buf, format='PNG', dpi=600, bbox_inches='tight')
    buf.seek(0)
    plt.close()
    image = Image.open(buf)
    return image


# Helper function to convert RGBA to hex
def to_hex(color):
    return "#{:02x}{:02x}{:02x}".format(int(color[0] * 255), int(color[1] * 255), int(color[2] * 255))


# def create_multiple_pareto_curves_old(datasets,
#                                   dirname,
#                                   filename,
#                                   color_by_flow_rate=False,
#                                   color_by_structure=False):
#     """
#     Plot Pareto fronts from multiple Dataset objects on the same plot, using their precomputed pareto_front_df attribute.

#     Parameters:
#         datasets (list of Dataset objects): Each must have a .pareto_front_df attribute with columns 'F1' and 'F2'.
#         dirname (str): Directory path where the plot will be saved.
#         filename (str): Filename of the saved plot image.
#         color_by_flow_rate (bool, optional): If `True`, colors points based on flow rate values.
#         color_by_structure (bool, optional): If `True`, colors points based on unique structures.

#     Returns:
#         PIL.Image: The generated plot as a PIL Image.
#     """
#     from src.utils.general_utils import apply_axis_label_mapping
#     fig, ax = plt.subplots(figsize=(8, 6))

#     markers = ['o', 's', 'D', '^', 'v', 'P', '*', 'X', 'h']
    
#     color_palette = cm.tab20
#     color_index = 0
#     if color_by_flow_rate:
#         cmap = plt.get_cmap("viridis")

#     for idx, ds in enumerate(datasets):
#         df = ds.pareto_front_df.reset_index(drop=True)
#         if color_by_structure:
#             # Find unique structures that appear at least twice in this dataset
#             structure_counts = df[ds.actvar_struct].value_counts()
#             repeated_structures = structure_counts[structure_counts > 1].index
#             unique_structures = pd.DataFrame(list(repeated_structures), columns=ds.actvar_struct)
#             # Assign colors for new structures
#             for _, structure_row in unique_structures.iterrows():
#                 structure = tuple(structure_row)
#                 if structure not in global_structure_color_map:
#                     global_structure_color_map[structure] = to_hex(color_palette(color_index / 20))
#                     color_index += 1
#             # Plot only points of repeated structures
#             for _, row in df.iterrows():
#                 structure = tuple(row[ds.actvar_struct])
#                 if structure in global_structure_color_map:
#                     ax.scatter(
#                         row["F1"], row["F2"],
#                         color=global_structure_color_map[structure],
#                         s=20,
#                         marker="o",
#                         alpha=0.7
#                     )
#             from matplotlib.patches import Patch
#             structure_patches = []
#             for structure, color in global_structure_color_map.items():
#                 label = "-".join(str(val) for val in structure)
#                 structure_patches.append(Patch(color=color, label=label))
#             ax.legend(handles=structure_patches, loc='lower right', title="Unique structure-fluid combinations")
#             # Sort by F1 to connect points in logical order (you can choose F2 instead if preferred)
#             df_sorted = df.sort_values(by="F1")

#             # Add transparent grey line connecting Pareto points
#             ax.plot(
#                 df_sorted["F1"],
#                 df_sorted["F2"],
#                 color='grey',
#                 linewidth=1,
#                 alpha=0.4,
#                 zorder=1  # ensures the line stays behind the scatter points
#             )
#         elif color_by_flow_rate:
#             norm = plt.Normalize(df["Fluid (kg/s) "].min(), df["Fluid (kg/s) "].max())
#             scatter = ax.scatter(
#                 df["F1"], df["F2"],
#                 c=df["Fluid (kg/s) "],
#                 cmap=cmap, norm=norm, s=50, marker=markers[idx % len(markers)], alpha=0.7,
#                 label=f"{ds.name} (Pareto Front)"
#             )
#         else:
#             ax.scatter(
#                 df["F1"], df["F2"],
#                 label=f"{ds.name} (Pareto Front)",
#                 s=50, marker="*", alpha=0.7
#             )

#     if color_by_flow_rate:
#         cbar = plt.colorbar(scatter, ax=ax)
#         cbar.set_label("Fluid flow rate [kg/s]")

#     ax.set_xlabel("F1")
#     ax.set_ylabel("F2")
#     if not color_by_structure and not color_by_flow_rate:
#         ax.legend(loc="lower right")
#     plt.grid(False)
#     ax.set_aspect(np.diff(ax.get_xlim())/np.diff(ax.get_ylim()))
#     plt.tight_layout()
#     apply_axis_label_mapping(ax)
#     buf = BytesIO()
#     plt.savefig(buf, format='PNG', dpi=600, bbox_inches='tight')
#     buf.seek(0)
#     plt.close()
#     image = Image.open(buf)
#     return image


#=== FUNCTION TO PLOT CONFUSION MATRIX ===#
@save_image_decorator
def plot_confusion_matrix(cm, dirname, filename):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    buf = BytesIO()
    plt.savefig(buf, format='PNG', dpi=600, bbox_inches='tight')
    buf.seek(0)
    plt.close()
    image = Image.open(buf)
    return image


#=== FUNCTION TO CREATE SCATTER MATRIX ===#
@save_image_decorator
def createScatterMatrix(ds, columns, dirname, filename):
    """
    Generates a scatter matrix plot to visualize pairwise relationships between features.

    The scatter matrix provides an overview of feature interactions while differentiating 
    data points based on class labels using color coding.

    Parameters:
        ds (object): Dataset object containing feature data (`ds.df`), including columns for features and class labels.
        columns (list): List of feature column names to include in the scatter matrix plot.
        dirname (str): Directory path where the plot image should be saved.
        filename (str): Filename for storing the generated plot.

    Returns:
        PIL.Image: The scatter matrix plot as a PIL Image, allowing further processing or saving.

    Additional Details:
        - Uses class labels from `ds.df["class"]` for coloring (`-1` → red, `1` → green).
        - Displays **KDE (Kernel Density Estimate)** on diagonal elements for better distribution analysis.
        - Enhances axis labels for improved readability.
        - Includes a **legend** mapping class labels to custom names.
    """
    
    df = pd.DataFrame(ds.df[columns])
    df.columns = columns
    label_colors = ds.df['class'].map({-1: "red", 1: "green"})
    axes = pd.plotting.scatter_matrix(pd.DataFrame(df),
                               c=label_colors, 
                               figsize=(10, 10), 
                               marker='o', 
                               s=5, 
                               diagonal = 'hist', 
                               alpha=.8)
    
    # Increase the size and boldness of axis labels
    #plt.rcParams.update({'font.size': 14, 'axes.labelsize': 14, 'axes.titlesize': 14, 'font.weight': 'bold'})

    # Set the font size for axis titles
    # for ax in axes.flatten():
    #     ax.set_xlabel(ax.get_xlabel(), fontsize=20)
    #     ax.set_ylabel(ax.get_ylabel(), fontsize=20)   
    
    # Get unique class labels
    unique_classes = ds.df['class'].unique() 
    # Map class labels to custom names 
    class_labels = {-1: "DI", 1: "DF"}  
     # Assign red for DI and green for DF
    colors = ['red' if label == -1 else 'green' for label in unique_classes] 
    
    # Create patches for the legend with custom labels
    patches = [
        mpatches.Patch(color=colors[i], label=class_labels[unique_classes[i]])
        for i in range(len(unique_classes))
        ]
    
    
    # Add the legend to the plot
    axes[0, 0].legend(handles=patches, loc='upper left')
    # Save the plot to a buffer
    buf = BytesIO()
    plt.savefig(buf, format='PNG', dpi = 600)
    buf.seek(0)
    plt.close()
    
    # Convert the buffer to a PIL Image
    image = Image.open(buf)
    
    return image

@save_image_decorator
def plot_pareto_front(
    df,
    obj1,
    obj2,
    dirname,
    filename,
    color_by=None,
    cmap="viridis",
    marker="*",
    color="red",
    size=60,
    alpha=0.7,
    title=None,
    xlabel=None,
    ylabel=None,
    legend_labels=None,
):
    import matplotlib.pyplot as plt
    from io import BytesIO
    from PIL import Image
    import matplotlib.patches as mpatches
    import numpy as np

    fig, ax = plt.subplots(figsize=(8, 6))

    legend_handles = []

    # Plot all points
    if color_by is not None and color_by in df.columns:
        # Convert string categories to numeric codes for coloring
        categories = df[color_by].astype('category')
        color_values = categories.cat.codes
        scatter = ax.scatter(
            df[obj1], df[obj2], c=color_values, cmap=cmap, alpha=alpha,
            marker=marker, s=size
        )

        # Build legend handles for each unique category
        cat_labels = categories.cat.categories
        for code, label in enumerate(cat_labels):
            patch_color = scatter.cmap(scatter.norm(code))
            legend_handles.append(mpatches.Patch(color=patch_color, label=str(label)))
    else:
        scatter = ax.scatter(
            df[obj1], df[obj2], alpha=alpha, marker=marker, color=color, s=size, edgecolor="k", label="Points"
        )
        legend_handles.append(scatter)

    # Calculate and plot the ideal objective vector
    ideal_obj = df[[obj1, obj2]].min()
    ax.scatter(ideal_obj[obj1], ideal_obj[obj2], color='green', marker='o', s=100, label='Ideal Objective Vector')
    legend_handles.append(mpatches.Patch(color='green', label='Ideal Objective Vector'))

    # Calculate and plot the nadir point
    nadir_point = df[[obj1, obj2]].max()
    ax.scatter(nadir_point[obj1], nadir_point[obj2], color='blue', marker='o', s=100, label='Nadir Point')
    legend_handles.append(mpatches.Patch(color='blue', label='Nadir Point'))

    ax.set_xlabel(xlabel if xlabel else obj1)
    ax.set_ylabel(ylabel if ylabel else obj2)

    from src.sample.utils.general_utils import apply_axis_label_mapping
    apply_axis_label_mapping(ax)

    # Always show the legend if handles exist
    if legend_handles:
        ax.legend(handles=legend_handles, title=color_by if color_by else None)

    ax.set_aspect(np.diff(ax.get_xlim()) / np.diff(ax.get_ylim()))
    plt.tight_layout()
    buf = BytesIO()
    plt.savefig(buf, format="PNG", dpi=600)
    buf.seek(0)
    plt.close()
    image = Image.open(buf)
    return image

#=== FUNCTION TO PLOT PARETO FRONT WITH RECTANGLES ===#
@save_image_decorator
def plot_pareto_front_with_rectangles(
    df,
    obj1,
    obj2,
    dirname,
    filename,
    color_by=None,
    cmap="viridis",
    marker="*",
    color="red",
    size=60,
    alpha=0.7,
    title=None,
    xlabel=None,
    ylabel=None,
    legend_labels=None,
    structure_cols=None,  # <-- NEW ARGUMENT
):
    """
    Plot a Pareto front with optional rectangle overlays representing grouped structures.

    Parameters:
        df (pandas.DataFrame): DataFrame containing the Pareto front data.
        obj1 (str): Column name representing the x-axis objective.
        obj2 (str): Column name representing the y-axis objective.
        dirname (str): Directory path where the plot will be saved.
        filename (str): Filename of the saved plot image.
        color_by (str, optional): Feature to color by; default is None.
        cmap (str, optional): Colormap for color mapping; default is "viridis".
        marker (str, optional): Marker style for scatter points; default is "*".
        color (str, optional): Default scatter point color; default is "red".
        size (int, optional): Scatter point size; default is 60.
        alpha (float, optional): Transparency of points; default is 0.7.
        title (str, optional): Plot title.
        xlabel (str, optional): Label for the x-axis.
        ylabel (str, optional): Label for the y-axis.
        legend_labels (list, optional): Custom legend labels.
        structure_cols (list of str, optional): Column names that define unique structures 
        to be grouped and assigned colors.

    Returns:
        PIL.Image: The generated plot as a PIL Image object.

    The function plots a Pareto front with flexible coloring options. If structure_cols
    is provided, unique structures are assigned colors, and rectangles are drawn around 
    their corresponding points to highlight their presence.
    """ 
    

    fig, ax = plt.subplots(figsize=(8, 6))

    # --- Assign colors for structures if needed ---
    structure_color_map = {}
    color_palette = cm.tab20
    color_index = 0

    # If structure_cols is given, color by structure
    if structure_cols is not None and all(col in df.columns for col in structure_cols):
        # Find unique structures that appear more than once
        structure_counts = df[structure_cols].value_counts()
        repeated_structures = structure_counts[structure_counts > 1].index
        unique_structures = pd.DataFrame(list(repeated_structures), columns=structure_cols)
        # unique_structures = ds.pareto_front_unique_structures
        for _, structure_row in unique_structures.iterrows():
            structure = tuple(structure_row)
            if structure not in structure_color_map:
                structure_color_map[structure] = to_hex(color_palette(color_index / 20))
                color_index += 1
        # Plot points and rectangles for each structure
        for structure, color_struct in structure_color_map.items():
            mask = (df[structure_cols] == pd.Series(structure, index=structure_cols)).all(axis=1)
            # Plot points
            ax.scatter(
                df.loc[mask, obj1], df.loc[mask, obj2],
                color=color_struct,
                s=size, marker=marker, alpha=alpha, label="-".join(str(val) for val in structure)
            )
            # Draw rectangle (same color, more transparent)
            x_min, x_max = df.loc[mask, obj1].min(), df.loc[mask, obj1].max()
            y_min, y_max = df.loc[mask, obj2].min(), df.loc[mask, obj2].max()
            rect = mpatches.Rectangle(
                (x_min, y_min),
                x_max - x_min,
                y_max - y_min,
                linewidth=1.5,
                edgecolor=color_struct,
                facecolor=color_struct,
                alpha=0.15,
                zorder=0
            )
            ax.add_patch(rect)
    else:
        # Default coloring
        scatter = ax.scatter(
            df[obj1], df[obj2], alpha=alpha, marker=marker, color=color, s=size, edgecolor="k", label="Pareto front"
        )

    ax.set_xlabel(xlabel if xlabel else obj1)
    ax.set_ylabel(ylabel if ylabel else obj2)
    from src.sample.utils.general_utils import apply_axis_label_mapping
    apply_axis_label_mapping(ax)

    # Only show legend for points
    ax.legend(title=color_by if color_by else None)
    ax.set_aspect(np.diff(ax.get_xlim()) / np.diff(ax.get_ylim()))
    plt.tight_layout()
    buf = BytesIO()
    plt.savefig(buf, format="PNG", dpi=600)
    buf.seek(0)
    plt.close()
    image = Image.open(buf)
    return image


#=== FUNCTION TO CREATE CONFUSION MATRIX IMAGE ===#
@save_image_decorator
def generate_confusion_matrix(model, ds, dirname, filename):
    """
    Generates and returns a confusion matrix as a PIL Image object.

    The function calculates the confusion matrix based on the model’s predictions 
    and visualizes it using a heatmap-style plot. The resulting image helps analyze 
    classification performance by displaying misclassifications and correct predictions.

    Parameters:
        model (object): A trained classification model with a `predict` method 
                        and a `classes_` attribute for retrieving class labels.
        ds (object): Dataset object containing test data (`ds.X_test_transformed`) 
                     and ground truth labels (`ds.y_test`).
        dirname (str): Directory where the confusion matrix image should be saved.
        filename (str): Filename for storing the generated confusion matrix image.

    Returns:
        PIL.Image: The confusion matrix plot as a PIL Image object for further processing or saving.

    Additional Details:
        - Uses `confusion_matrix(ds.y_test, model.predict(ds.X_test_transformed))` to compute errors.
        - Visualizes the matrix using `ConfusionMatrixDisplay` with a `"Blues"` colormap.
        - Saves the figure to a buffer before converting it to a PIL Image.
    """
    # Compute the confusion matrix
    cm = confusion_matrix(ds.y_test, model.predict(ds.X_test_transformed), labels=model.classes_)
    
    # Create the confusion matrix display
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
    
    # Plot the confusion matrix
    fig, ax = plt.subplots(figsize=(8, 8))
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    
    # Save the plot to a buffer
    buf = BytesIO()
    plt.savefig(buf, format="PNG", dpi = 600)
    buf.seek(0)
    plt.close()
    
    # Convert the buffer to a PIL Image
    image = Image.open(buf)
    
    return image




# #=== FUNCTION TO CREATE DECISION TREE PLOT ===#
# @save_image_decorator
# @apply_axis_labels
# def createDecisionTreePlot(classTree, ds, dirname, filename):
#     """
#     Generates a visual representation of a trained decision tree classifier.

#     The function creates a structured tree plot displaying feature splits and class 
#     distributions. The decision tree is color-filled to indicate class regions, making 
#     it easier to interpret model decision boundaries.

#     Parameters:
#         classTree (DecisionTreeClassifier): A trained scikit-learn decision tree classifier.
#         ds (object): Dataset object containing feature names (`ds.features`) used for training.
#         dirname (str): Directory path where the generated plot will be saved.
#         filename (str): Name of the file for saving the decision tree visualization.

#     Returns:
#         PIL.Image: The generated decision tree plot as a PIL Image, allowing further processing or saving.

#     Additional Details:
#         - Uses `tree.plot_tree()` for visualization with proportionally scaled nodes.
#         - Labels feature splits with corresponding feature names from `ds.features`.
#         - Color-codes class labels using `class_names=["DI", "DF"]` for readability.
#     """
    
#     # Plot the decision tree
#     tree.plot_tree(classTree, proportion=True, filled=True, feature_names=ds.features, class_names=["DI","DF"])

#     # Save the plot to a buffer
#     buf = BytesIO()
#     plt.savefig(buf, format='PNG', dpi = 600)
#     buf.seek(0)
#     plt.close()
    
#     # Convert the buffer to a PIL Image
#     image = Image.open(buf)
    
#     return image



#=== FUNCTION TO CREATE A CONTOUR PLOT ===#
@save_image_decorator
def create_contour_plot(data,dirname, filename):
    """
    Create a contour plot from a NumPy array with shape (n, 3).

    Parameters:
        data (numpy.ndarray): A NumPy array with shape (n, 3), where:
                              - Column 0: X values
                              - Column 1: Y values
                              - Column 2: Z values
        xlabel (str): Label for the X-axis.
        ylabel (str): Label for the Y-axis.
        zlabel (str): Label for the Z-axis (color bar).
        title (str): Title of the plot.

    Returns:
        image (PIL.Image.Image): The contour plot as a PIL Image.
    """
    # Ensure the input data has the correct shape
    if data.shape[1] != 3:
        raise ValueError("Input data must have shape (n, 3).")

    # Extract X, Y, Z values
    X = data[:, 0]
    Y = data[:, 1]
    Z = data[:, 2]

    # Create a grid for X and Y
    x_unique = np.unique(X)
    y_unique = np.unique(Y)
    X_grid, Y_grid = np.meshgrid(x_unique, y_unique)

    # Reshape Z to match the grid
    Z_grid = Z.reshape(len(y_unique), len(x_unique))

    # Create the contour plot
    plt.figure(figsize=(8, 6))
    contour = plt.contourf(X_grid, Y_grid, Z_grid, levels=20, cmap="viridis")
    plt.colorbar(contour, label="Z")
    plt.xlabel("X")
    plt.ylabel("Y")
    
    # Save the plot to a buffer
    buf = BytesIO()
    plt.savefig(buf, format="PNG", dpi = 600)
    buf.seek(0)
    plt.close()
    
    # Convert the buffer to a PIL Image
    image = Image.open(buf)
    
    return image


#=== FUNCTION TO CREATE A PARITY PLOT AND A RESIDUAL PLOT ===# 
@save_image_decorator
def create_parity_plot(y, y_pred, dirname, filename, include_residuals=False):
    """
    Generates a parity plot to evaluate regression model performance.

    The function creates two subplots:
    1. **Actual vs. Predicted Values:** Visualizes how well predictions align with true values.
    2. **Relative Residuals vs. Predicted Values (optional):** Shows the percentage difference between predictions and actual values.

    Parameters:
        y (array-like): Ground truth values.
        y_pred (array-like): Predicted values from the regression model.
        dirname (str): Directory where the parity plot image will be saved.
        filename (str): Filename for storing the generated plot.
        include_residuals (bool): Whether to include the residual plots. Default is True.

    Returns:
        PIL.Image: The generated parity plot as a PIL Image, allowing further processing or saving.
    """
    if include_residuals:
        fig, axs = plt.subplots(ncols=2, figsize=(12, 4))  # Two subplots if residuals are included
    else:
        fig, axs = plt.subplots(ncols=1, figsize=(6, 4))  # One subplot if residuals are excluded
        axs = [axs]  # Wrap single axis in a list for consistency

    # Actual vs. Predicted Values
    PredictionErrorDisplay.from_predictions(
        y,
        y_pred=y_pred,
        kind="actual_vs_predicted",
        subsample=None,
        ax=axs[0],
        random_state=42,
    )

    #axs[0].set_xlim(0,1)
    #axs[0].set_ylim(0,1)
    #axs[0].set_xticks(np.arange(0, 1.1, 0.2))
    #axs[0].set_yticks(np.arange(0, 1.1, 0.2))
    #axs[0].set_aspect('equal', adjustable='box')
    if include_residuals:
        # Compute relative residuals as a percentage
        relative_residuals = 100 * abs((y - y_pred) / y)

        # Residuals vs. Predicted Values
        PredictionErrorDisplay.from_predictions(
            y,
            y_pred=y_pred + relative_residuals,
            kind="residual_vs_predicted",
            subsample=None,
            ax=axs[1],
            random_state=0,
        )


    # Set aspect ratio after all plotting is done
    axs[0].set_aspect('equal', adjustable='box')
    if include_residuals:
        axs[1].set_aspect('equal', adjustable='box')


     # --- Set round-number ticks and padding ---
    from src.sample.utils.general_utils import get_tick_step
    for ax in axs:
        # Get current limits
        x_min, x_max = ax.get_xlim()
        y_min, y_max = ax.get_ylim()
        # Find min/max for both axes
        axis_min = min(x_min, y_min)
        axis_max = max(x_max, y_max)
        # Compute dynamic tick step
        tick_step = get_tick_step(axis_min, axis_max)
        # Round down/up to nearest multiple of tick_step
        tick_min = tick_step * np.floor(axis_min / tick_step)
        tick_max = tick_step * np.ceil(axis_max / tick_step)
        # Set limits with a little extra padding if desired
        ax.set_xlim(tick_min, tick_max)
        ax.set_ylim(tick_min, tick_max)
        # Set ticks at intervals of tick_step
        ticks = np.arange(tick_min, tick_max + tick_step, tick_step)
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)

    plt.tight_layout()
    # Save the plot to a buffer
    buf = BytesIO()
    plt.savefig(buf, format="PNG", dpi=600)
    buf.seek(0)
    plt.close()

    # Convert the buffer to a PIL Image
    image = Image.open(buf)

    return image






