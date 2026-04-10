# import standard libraries
import os
import sys
import numpy as np
import pandas as pd

# import machine learning modules
import joblib


# ADD PROJECT DIRECTORY TO SYSTEM PATH
sys.path.append(os.path.abspath(os.path.join(
    os.path.dirname(__file__), 
    "C:/Users/Stagiaire/Documents/Stage_Ilyes_Ait_Aissa_Repo"
)))

# Import custom modules

from sample.decorators.general_decorators import *
from src.sample.utils.saving_utils import *
from src.sample.config import *


#=== FUNCTION TO CHOOSE NICE TICK STEP ===#
def get_tick_step(data_min, data_max, n_ticks=6):
    """
    Determine a 'nice' tick step for axis labeling based on the data range 
    and the desired number of ticks.

    Parameters:
    - data_min (float): The minimum value of the data range.
    - data_max (float): The maximum value of the data range.
    - n_ticks (int, optional): The desired number of ticks (default is 6).

    Returns:
    - float: A 'nice' tick step value that ensures well-distributed axis labels.
    
    The method works by computing the raw step size, then adjusting it to the 
    nearest 'nice' value (such as 1, 2, 5, or 10 multiplied by a power of 10), 
    ensuring clear and readable axis divisions.
    """
    raw_step = (data_max - data_min) / (n_ticks - 1)
    magnitude = 10 ** np.floor(np.log10(raw_step))
    residual = raw_step / magnitude
    if residual < 1.5:
        nice_step = 1 * magnitude
    elif residual < 3:
        nice_step = 2 * magnitude
    elif residual < 7:
        nice_step = 5 * magnitude
    else:
        nice_step = 10 * magnitude
    return nice_step





#=== FUNCTION TO ADD CLASS LABELS DI/DF TO DATAFRAME ===#
@log_execution_time
def addLabelsToDataframe(df, constraints, label):
    """
    Assigns binary class labels to a DataFrame based on constraint satisfaction.

    This function checks whether each row satisfies all specified constraints—
    specifically, whether all values in the given columns are less than or equal to zero.
    If a row meets all conditions, it is labeled with a `1` (positive class); otherwise, `0` (negative class).

    Parameters:
        df (pd.DataFrame): The input DataFrame.
        constraints (list of str): List of column names to evaluate.
        label (str): Name of the new column that stores assigned class labels.

    Returns:
        pd.DataFrame: The updated DataFrame with a new column containing binary class labels.

    Example:
        >>> df = pd.DataFrame({'A': [1, -2], 'B': [0, -1]})
        >>> addLabelsToDataframe(df, ['A', 'B'], 'label')
           A  B  label
        0  1  0      0
        1 -2 -1      1
    """
    # Reset the index to ensure it is continuous
    df = df.reset_index(drop=True)  
    df[label] = (df[constraints].le(0).all(axis=1)).astype(int)
    return df


#=== FUNCTION TO ADD LABELS ABOUT MISCLASSIFICATION TO DATAFRAME ===#
@log_execution_time
def addLabelsToDataFrame_clf(df, col_class, col_predicted, col_classification_result, col_classification_type):
    """
    Adds a classification result column to indicate whether a data point was correctly classified
    and a column to indicate if it is a False Positive, False Negative, True Positive, or True Negative.

    The function compares the "Predicted_Label" column with the actual "class" column.
    If a prediction matches the true label, the classification result is set to `1` (correctly classified).
    If the prediction is incorrect, it is set to `-1` (misclassified).

    Parameters:
        - df (pd.DataFrame): DataFrame containing classification results with columns
                           "Predicted_Label" (predicted category) and "class" (true category).

    Returns:
        pd.DataFrame: The updated DataFrame with additional "classification_result" and "classification_type" columns.
                      - `1` indicates correct classification.
                      - `-1` indicates misclassification.
                      - "classification_type" indicates FP, FN, TP, or TN.
    """
    # Reset the index of the DataFrame
    df = df.reset_index(drop=True)
    df[col_classification_result] = [0] * len(df.index)
    df[col_classification_type] = [""] * len(df.index)

    for i in range(len(df.index)):
        predicted = df.loc[i, col_predicted]
        actual = df.loc[i, col_class]

        if predicted == actual:
            df.loc[i, col_classification_result] = 1
            if predicted == 1:  # Assuming 1 is the positive class
                df.loc[i, col_classification_type] = "TP"
            else:
                df.loc[i, col_classification_type] = "TN"
        else:
            df.loc[i, col_classification_result] = -1
            if predicted == 1:  # Assuming 1 is the positive class
                df.loc[i, col_classification_type] = "FP"
            else:
                df.loc[i, col_classification_type] = "FN"

    return df





#=== FUNCTION TO DETERMINE SET OF DOMINATED VECTORS ===#    
@log_execution_time
def is_dominated_by_k(costs, k, return_mask=True):
    """
    Identify points in a cost space that are dominated by exactly 'k' other points.

    Parameters:
    - costs (array-like): An array of shape (n_points, n_costs), where each row represents 
      a point and each column corresponds to a cost dimension.
    - k (int): The number of dominating points a data point must have.
    - return_mask (bool, optional): 
      - If True, returns a boolean mask indicating dominated points.
      - If False, returns an array of indices for dominated points.

    Returns:
    - numpy.ndarray:
      - If `return_mask` is True: A boolean array of shape (n_points,) marking dominated points.
      - If `return_mask` is False: An integer array containing indices of dominated points.

    The function computes the number of times each point is dominated by others in 
    the cost space, then selects points that are dominated exactly `k` times.
    """
    costs = np.array(costs)
    n_points = costs.shape[0]
    domination_counts = np.zeros(n_points, dtype=int)

    # Count how many times each point is dominated
    for i, point in enumerate(costs):
        domination_counts[i] = np.sum(np.all(costs < point, axis=1))

    # Get the mask or indices of points dominated by exactly k others
    if return_mask:
        return domination_counts == k
    else:
        return np.where(domination_counts == k)[0]


#=== FUNCTION TO GET UNION OF DOMINATED POINTS UP TO K ===#
@log_execution_time
def is_dominated_by_up_to_k(costs, k, return_mask=True):
    """
    Identify points in a cost space that are dominated by up to 'k' others.

    Parameters:
    - costs (array-like): An array of shape (n_points, n_costs), where each row represents 
      a point and each column corresponds to a cost dimension.
    - k (int): The maximum number of dominating points allowed.
    - return_mask (bool, optional): 
      - If True, returns a boolean mask indicating selected points.
      - If False, returns an array of the actual subset of selected points.

    Returns:
    - numpy.ndarray:
      - If `return_mask` is True: A boolean array of shape (n_points,) marking selected points.
      - If `return_mask` is False: A (n_selected_points, n_costs) array containing the selected points.

    The function computes how many times each point is dominated by others in the cost space. 
    It then selects points that are dominated at most `k` times.
    """
    costs = np.array(costs)
    n_points = costs.shape[0]
    domination_counts = np.zeros(n_points, dtype=int)

    # Count how many times each point is dominated
    for i, point in enumerate(costs):
        domination_counts[i] = np.sum(np.all(costs < point, axis=1))

    # Create mask for points dominated by up to 'k' others
    mask = domination_counts <= k

    return mask if return_mask else costs[mask]




#=== FUNCTION TO FIND UNIQUE STRUCTURES CORRESPONDING TO PARETO POINTS ===#
@log_execution_time
def find_frequent_combinations_with_min_max(ds, df, group_columns, value_column, second_value_column, threshold=1):
    """
    Identify frequent value combinations in specified columns, calculate the 
    minimum and maximum values for two target columns, and return the results 
    sorted by the minimum value of the first target column.

    Parameters:
    - df (pd.DataFrame): The input DataFrame.
    - group_columns (list of str): List of column names to group by (e.g., ['A', 'C', 'D']).
    - value_column (str): Name of the first column for which to find min and max values (e.g., 'F').
    - second_value_column (str): Name of the second column for which to find min and max values.
    - threshold (int, optional): Minimum frequency for a combination to be considered frequent (default is 1).

    Returns:
    - pd.DataFrame: DataFrame containing the grouped combinations, their min/max values for 
      both specified columns, and count of occurrences.

    The function groups the data by `group_columns`, counts occurrences, and filters 
    combinations that appear more than `threshold` times. For each frequent combination, 
    it calculates min/max values of `value_column` and `second_value_column`, then sorts the results 
    by the growing order of the minimum value in `value_column`.
    """
    # Group by all columns in 'group_columns' and count occurrences
    combinations = df.groupby(group_columns).size().reset_index(name='Count')

    # Filter only the combinations that occur more than the threshold
    frequent_combinations = combinations[combinations['Count'] > threshold]

    # For each frequent combination, find the min and max of the specified columns
    result = []
    for _, row in frequent_combinations.iterrows():
        # Dynamically filter using group_columns
        condition = (df[group_columns] == row[group_columns].values).all(axis=1)
        subset = df[condition]
        min_value = subset[value_column].min()
        max_value = subset[value_column].max()
        min_value2 = subset[second_value_column].min()
        max_value2 = subset[second_value_column].max()
        result.append({
            **row.to_dict(),
            'Min_' + value_column: min_value,
            'Max_' + value_column: max_value,
            'Min_' + second_value_column: min_value2,
            'Max_' + second_value_column: max_value2
        })

    # Create the results as a DataFrame and order by the 'Min_value_column'
    result_df = pd.DataFrame(result)
    if not result_df.empty:
        result_df = result_df.sort_values(by='Min_' + value_column).reset_index(drop=True)
        return result_df
    else:
        return df



#=== FUNCTION TO CALCULATE PERCENTAGES OF UNFULFILLED CONSTRAINTS ===#
@log_execution_time
def calculate_unfulfilled_constraints_per_constraint(df, constraints):
    """
    Compute the percentage of unfulfilled constraints for each constraint column.

    Parameters:
    - df (pandas.DataFrame): The dataset containing constraint columns.
    - constraints (list of str): List of column names representing constraints.

    Returns:
    - dict: A dictionary where keys are constraint names and values are the percentage 
      of rows where the constraint is unfulfilled (i.e., the value > 0).

    The function iterates over each constraint column and calculates the percentage 
    of rows where the constraint is violated. The result helps evaluate the frequency 
    of unfulfilled constraints in the dataset.
    """
    unfulfilled_percentages = {}
    
    for constraint in constraints:
        # Calculate the percentage of rows where the constraint is unfulfilled (value > 0)
        unfulfilled_count = (df[constraint] > 0).sum()
        total_rows = len(df)
        unfulfilled_percentages[constraint] = (unfulfilled_count / total_rows) * 100

    return unfulfilled_percentages





#=== FUNCION TO SPLIT DATAFRAME INTO SUBDATAFRAMES WITH UNIQUE COMBINATIONS OF COLUMN VALUES ===#
@log_execution_time
def split_dataframe(df, group_columns):
    """
    Split a DataFrame into multiple sub-DataFrames based on unique combinations 
    of values in the specified columns.

    Parameters:
    - df (pandas.DataFrame): The DataFrame to be split.
    - group_columns (list of str): Column names used for grouping.

    Returns:
    - dict: A dictionary where keys are tuples representing unique value combinations 
      in `group_columns`, and values are the corresponding sub-DataFrames.

    The function groups the DataFrame by the specified columns, then creates 
    separate sub-DataFrames for each unique combination of values. The results 
    are stored in a dictionary, allowing easy access to different subsets.
    """
    # Create a dictionary to store the split DataFrames
    split_dfs = {}

    # Group the DataFrame by the specified columns
    grouped = df.groupby(group_columns)

    # Iterate over each group
    for group_values, group_df in grouped:
        # Convert group values to a concatenated string of numbers
        key = ''.join(map(str, group_values))
        # Retain all columns in the resulting DataFrame
        split_dfs[key] = group_df.reset_index(drop=True)

    return split_dfs

#=== FUNCTION TO DETERMINE MODEL TYPE OF A MACINE LEARNING OBJECT ===#
def get_model_type(model):
    """
    Identify whether the given machine learning model is a classifier or a regressor.

    Parameters:
    - model (object): A machine learning model instance, typically from scikit-learn.

    Returns:
    - str: A string indicating the model type:
      - `"classification"` if the model is a classifier.
      - `"regression"` if the model is a regressor.
      - `"unknown"` if the model type cannot be determined.

    The function checks the `_estimator_type` attribute, which is commonly found in 
    scikit-learn models, to determine whether the model is intended for classification 
    or regression.
    """
    # Check if the model has the '_estimator_type' attribute
    if hasattr(model, '_estimator_type'):
        estimator_type = model._estimator_type
        if estimator_type == 'classifier':
            return "classification"
        elif estimator_type == 'regressor':
            return "regression"
        else:
            return "unknown"
    else:
        return "unknown"







#=== FUNCTION FOR LATIN HYPERCUBE SAMPLING WITH MIXED VARIABLE TYPES ===#
@log_execution_time
def latin_hypercube_sampling_with_bounds_and_types(num_samples, lower_bounds, upper_bounds, var_types, seed=None):
    """
    Perform Latin Hypercube Sampling (LHS) for mixed variable types (both float and integer).

    Parameters:
    - num_samples (int): The number of samples to generate.
    - lower_bounds (list or numpy.ndarray): Lower bounds for each variable.
    - upper_bounds (list or numpy.ndarray): Upper bounds for each variable.
    - var_types (list of str): Specifies whether each variable is `"int"` or `"float"`.
    - seed (int, optional): Random seed for reproducibility (default is None).

    Returns:
    - numpy.ndarray: A sampled array of shape (num_samples, n_vars), where 
      integer variables are rounded appropriately.

    The function first generates samples using Latin Hypercube Sampling, scales 
    them based on the provided bounds, and rounds integer variables while 
    keeping float variables unchanged.
    """
    from scipy.stats import qmc
    import numpy as np

    n_vars = len(lower_bounds)
    sampler = qmc.LatinHypercube(d=n_vars, seed=seed)
    samples = sampler.random(n=num_samples)
    scaled = samples * (np.array(upper_bounds) - np.array(lower_bounds)) + np.array(lower_bounds)
    for i, vtype in enumerate(var_types):
        if vtype == "int":
            scaled[:, i] = np.round(scaled[:, i]).astype(int)
    return scaled