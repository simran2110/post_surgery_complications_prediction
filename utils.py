import pandas as pd 
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import re 
import os
def ohe_transform(df, ohe, list_features, prefix_label=""):
    """
    Given a onehot encoder (ohe), encode specific columns of a dataframe.
    This allows train and test data to be encoded using the same encoder.
    Return a dataframe that includes encoded values, with labels, indexed 
    as per the original dataframe.
    Assumptions: ohe was fit prior to calling this function, with relevant data.
    """
    encoded_categories = ohe.transform(df[list_features])
    column_labels = np.concatenate(ohe.categories_).ravel()
    column_labels = [prefix_label + str(lbl) for lbl in column_labels]
    return pd.DataFrame(encoded_categories,
                        columns=column_labels,
                        index=df.index)
    

def convert_col_to_binary(df, colnames):
    for col in colnames:
        df[f'{col}_binary'] = df[col].notna().astype(int)
    return df

def log_std_transform(df, colnames):
    df_log = np.log(suburb_df[feature_cols])
    return df 

def boxcox_std_transform(df, colnames):# scaling using boxcox transformation
    df_boxcox = {}
    df_boxcox = {col: boxcox(suburb_df[col])[0] for col in feature_cols}
    df_boxcox = pd.DataFrame(df_boxcox)
    df_boxcox.head()
    return df

def impute_zero(df, column):
    """Impute 0 where there are NaN or null values in the specified column."""
    column = column.strip()
    
    # Replace NaN or empty strings with 0
    df[column] = df[column].replace({np.nan: 0, None: 0, '': 0})
    return df

def check_count_of_nan_and_empty_string(df, col):
    print("col", col)
    # check unique values of anemias icd
    total_rows = len(df)  # Total rows in the column
    unique_values = df[col].unique() # Unique values in the column
    nan_count = df[col].isna().sum()  # Count NaNs
    empty_str_count = (df[col] == "").sum()  # Count empty strings
    space_count = (df[col] == " ").sum()  # Count single spaces
    
    # Valid entries: Total rows - (NaNs + empty strings + spaces)
    valid_count = total_rows - (nan_count + empty_str_count + space_count)

    # Calculate percentage of missing values
    missing_count = nan_count + empty_str_count + space_count
    missing_percentage = (missing_count / total_rows) * 100
    
    print(f"Column: {col}")
    print(f"  Total rows: {total_rows}")
    # print(f"Unique Values: {unique_values}")
    print(f"  NaN count: {nan_count}")
    print(f"  Empty string count: {empty_str_count}")
    print(f"  Single space count: {space_count}")
    print(f"  Valid entries count: {valid_count}")
    print(f"  Missing values count: {missing_count} ({missing_percentage:.2f}%)")
    print("-" * 30)
    
def validate_and_impute_binary_numeric_column(df, column):
    """Validate if a column can be binary (i.e., contains only NaN, empty string, None, and 1).
       If it can, impute 0 for non-1 values and return validation results."""
    
    print(f"\nValidating if {column} can be binary")

    # Clean the column by replacing None, empty strings, and NaN with NaN, then drop NaNs
    cleaned_column = df[column].replace([None, ''], np.nan).dropna()
    
    # Get unique non-NaN values
    unique_values = cleaned_column.unique()

    # Check if the unique values are just 1 (allowing NaN, empty, None)
    can_be_binary = set(unique_values) == {1}

    if can_be_binary:
        # Impute 0 for non-1 values in the column
        df[column] = df[column].replace([None, ''], 0)
        
    return {
        "canBeBinary": can_be_binary,
        "Unique_values": unique_values
    }
def validate_and_impute_binary_for_categorical_column(df, column):
    """
    Converts a column to binary if it has only two values: one actual value and NaN.
    If the column has two values (e.g., 'value' and NaN), it will be converted
    to 1 for 'value' and 0 for NaN.

    Args:
    - df (pandas.DataFrame): DataFrame containing the column to check.
    - column (str): The column name to check and convert.

    Returns:
    - df (pandas.DataFrame): DataFrame with the updated column.
    """
    unique_values = df[column].dropna().unique()  # Get unique non-NaN values
    
    if len(unique_values) == 1:  # Check if there is exactly one unique value other than NaN
        value = unique_values[0]
        # Convert the column to binary (1 for the actual value, 0 for NaN)
        df[column] = df[column].apply(lambda x: 1 if x == value else 0)
        return (df, True)
        print(f"Column '{column}' converted to binary.")
    else:
        print(f"Column '{column}' does not meet the condition (only one actual value and NaN).")
        return df, False
    
    
def one_hot_encode(df, column_name):
    """
    This function performs one-hot encoding on the specified column of the dataframe.
    
    Parameters:
    - df: The dataframe containing the categorical column to encode.
    - column_name: The name of the categorical column to one-hot encode.
    
    Returns:
    - The dataframe with one-hot encoded columns.
    """
    # Perform one-hot encoding using pandas get_dummies
    one_hot_encoded_df = pd.get_dummies(df, columns=[column_name], drop_first=False)
    
    return one_hot_encoded_df
    
# Function to check if a column is completely empty (all NaN or empty string)
def is_empty_column(column):
    # Check if all values in the column are NaN or empty strings
    return column.isna().all() or (column == "").all()

def remove_identifier_columns(df):
    """
    Removes identifier columns from the DataFrame.
    
    Parameters:
    - df (pd.DataFrame): The input DataFrame.
    
    Returns:
    - pd.DataFrame: DataFrame with identifier columns removed.
    """
    identifier_columns_to_exclude = [
        "mrn", "admission_id", "surname", "first_name", "dob", "death_flag", 
        "age", "date_of_death", "gender", "race", "ethnicity", "marital_status", 
        "pcode", "language", "interpreter", "phone1", "phone2", "t_phone", "email", 
        "emerg_name", "emerg_phone1", "gp_name", "gp_phone"
    ]

    # Remove only the columns that exist in the DataFrame
    columns_to_remove = [col for col in identifier_columns_to_exclude if col in df.columns]
    
    return df.drop(columns=columns_to_remove, errors="ignore")

def filter_zero_variance_columns(self, df):
        print("Fitering zero variance columns...")
        print(f"Columns before applying operation are {len(df.columns)}")
        
        constant_cols = [col for col in df.columns if df[col].nunique() <= 1]
        if constant_cols:
            print("Dropped columns:", constant_cols)
            for col in constant_cols:
                unique_vals = df[col].unique()
                print(f" - {col}: unique value(s) = {unique_vals}")
            df = df.drop(columns=constant_cols)
            print(f"Columns after filtering zero variance column is {len(df.columns)}")
        else:
            print("No column with zero variance")
        return df

def create_output_dir_from_filename(data_file, prefix="plots"):
    """
    Create a directory with a timestamp extracted from the filename,
    or use the current timestamp if none is found.

    Parameters:
    -----------
    data_file : str
        The filename (e.g., 'MIDAS_preprocessed_2025-05-02-11-12-42.csv')
    prefix : str
        Prefix for the folder name (default: 'plots')

    Returns:
    --------
    output_dir : str
        Path to the created directory
    """

    # Regex pattern to match timestamps like 2025-05-02-11-12-42
    match = re.search(r'\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}', data_file)

    if match:
        timestamp = match.group()
    else:
        timestamp = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')

    output_dir = f"{prefix}_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    print(f"📁 Directory created: {output_dir}")
    return output_dir

