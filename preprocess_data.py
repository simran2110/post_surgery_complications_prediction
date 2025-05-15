import pandas as pd 
import numpy as np
from utils import *
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
import warnings
from datetime import datetime
from preprocess_icds import IcdsPreprocessing
from plot_metrics import *
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer
from preprocess_target import *
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
import logging
import argparse

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class DataConfig:
    """Configuration for data processing parameters"""
    # File paths
    input_file: str
    output_file: str
    target_file: str = None  # Path to target features file
    target_columns: List[str] = None  # List of target column names
    
    # Feature selection
    selected_features: List[str] = None
    
    # Data validation
    invalid_ranges: Dict[str, Tuple[float, float]] = None
    
    # Imputation settings
    numeric_imputation_strategy: str = 'mean'
    categorical_imputation_strategy: str = 'most_frequent'
    impute_method: str = 'normal'
    
    # Encoding settings
    encode_categorical: bool = True
    drop_first: bool = True  # Whether to drop first category in one-hot encoding
    
    # Automatic detection settings
    max_unique_values_for_categorical: int = 10  # Maximum unique values to consider as categorical
    min_unique_values_for_categorical: int = 2   # Minimum unique values to consider as categorical
    
    def __post_init__(self):
        if self.selected_features is None:
            self.selected_features = [
                'record_id',
                "age",
                "gender",
                "days_surgery_discharge",
                "schedule_priority",
                "duration_of_procedure",
                "asa",
                # 'drg_code', 
                # 'icd_prim_code',
                "bmi",
                "wcc",
                "hb",
                "haematocrit",
                "platelets",
                "creatinine",
                "urea",
                "albumin",
                "sodium",
                "potassium",
                "inr",
                "fibrinogen",
                # "cci_ccf",
                # "cci_dementia",
                # "cci_copd",
                # "cci_rheum",
                # "cci_liver_mild",
                # "cci_liver_mod_severe",
                # "cci_diabetes_with",
                # "cci_hemiplegia",
                # "cci_renal_mod_severe",
                # "cci_solid_blood_ca",
                # "cci_metastatic_tumour",
                # "cci_hiv",
                "cci_total",
                'hospital_los',
                'icu_admission_date_and_tim'
            ]
        
        if self.target_columns is None:
            self.target_columns = ['hospital_los', 'dd_3month', 'dd_6month', 'los_target', '180_readmission']

class DataProcessor:
    """Class for processing and cleaning medical data"""
    
    def __init__(self, config: DataConfig):
        """Initialize the data processor with configuration"""
        self.config = config
        self.df: Optional[pd.DataFrame] = None
        self._validate_config()
    
    def _validate_config(self):
        """Validate the configuration parameters"""
        if not os.path.exists(self.config.input_file):
            raise FileNotFoundError(f"Input file not found: {self.config.input_file}")
        
        if not self.config.selected_features:
            raise ValueError("No features selected for processing")
    
    def load_data(self) -> pd.DataFrame:
        """Load and validate the dataset"""
        logger.info(f"Loading dataset from {self.config.input_file}")
        try:
            df = pd.read_csv(self.config.input_file, low_memory=False)
            self.df = self._remove_invalid_patients(df)
            self.df = self.df[self.df["redcap_event_name"].str.contains("index")]
            print(f"Dataset size after filtering index events: {self.df.shape}")
        
            logger.info(f"Dataset loaded successfully: {self.df.shape}")
            return self.df
        except Exception as e:
            logger.error(f"Error loading dataset: {str(e)}")
            raise
    
    def _remove_invalid_patients(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove invalid patients from the dataset"""
        logger.info("Removing invalid patients...")
        
        # Remove opted-out patients
        optout_columns = ['optout_baseline', 'optout_baseline_3m', 'optout_baseline_6m']
        existing_optout_cols = [col for col in optout_columns if col in df.columns]
        
        if existing_optout_cols:
            opted_out_ids = df[df[existing_optout_cols].eq(1).any(axis=1)]['record_id'].unique()
            df = df[~df['record_id'].isin(opted_out_ids)]
            logger.info(f"Removed {len(opted_out_ids)} opted-out patients")
        
        # Remove patients with certain procedures
        procedures_to_remove = ["gastroscopy", "colonoscopy", "bronchoscopy", "flexi cystoscopy"]
        if 'primary_procedure' in df.columns:
            procedure_ids = df[df['primary_procedure'].str.lower().isin([p.lower() for p in procedures_to_remove])]['record_id'].unique()
            df = df[~df['record_id'].isin(procedure_ids)]
            logger.info(f"Removed {len(procedure_ids)} patients with excluded procedures")
        
        return df
    
    def filter_zero_variance_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove columns with zero variance"""
        logger.info("Filtering zero variance columns...")
        logger.info(f"DataFrame shape before filtering: {df.shape}")
        
        # Now identify columns with zero variance
        constant_cols = [col for col in df.columns if df[col].nunique() <= 1]
        
        if constant_cols:
            logger.info(f"\nRemoving {len(constant_cols)} constant columns")
            
            # Check if any selected features are being dropped
            selected_features_dropped = [col for col in constant_cols if col in self.config.selected_features]
            if selected_features_dropped:
                logger.warning(f"\nWARNING: The following selected features are being dropped due to zero variance: {selected_features_dropped}")
                logger.warning("These features have the following values:")
                for col in selected_features_dropped:
                    unique_values = df[col].unique()
                    value_counts = df[col].value_counts()
                    logger.warning(f"{col}:")
                    logger.warning(f"  Unique values: {unique_values}")
                    logger.warning(f"  Value counts:\n{value_counts}")
                    logger.warning(f"  Data type: {df[col].dtype}")
            
            # Log all columns being dropped
            for col in constant_cols:
                if col not in selected_features_dropped:  # Skip already logged selected features
                    logger.debug(f"Column {col}: unique value(s) = {df[col].unique()}")
            
            df = df.drop(columns=constant_cols)
        
        logger.info(f"\nDataFrame shape after filtering: {df.shape}")
        return df
    
    def remove_low_variance_features(self, df: pd.DataFrame, exclude_na=True, top_feature_prop_threshold=80):
        """
        Removes low-variance features based on:
        1. Zero variance (only one unique value)
        2. High dominance of a single value (> threshold %)

        Parameters:
            df: Input DataFrame
            exclude_na: Whether to ignore NaN in uniqueness checks
            top_feature_prop_threshold: % threshold for dominance (e.g., 95 = 95%)

        Returns:
            Tuple of:
                - Filtered DataFrame
                - List of removed columns
                - List of retained columns
        """
        logger.info("Starting low-variance feature filtering...")
        logger.info(f"DataFrame shape before filtering: {df.shape}")

        df_clean = df.copy()
        dropped_columns = []

        # Step 1: Remove zero-variance columns (constant values)
        zero_var_cols = [col for col in df_clean.columns if df_clean[col].nunique(dropna=exclude_na) <= 1]
        if zero_var_cols:
            logger.info(f"Removing {len(zero_var_cols)} zero-variance columns: {zero_var_cols}")
            dropped_columns.extend(zero_var_cols)
            df_clean = df_clean.drop(columns=zero_var_cols)

            selected_dropped = [col for col in zero_var_cols if col in getattr(self.config, 'selected_features', [])]
            for col in selected_dropped:
                logger.warning(f"Selected feature dropped due to zero variance: {col}")
                logger.warning(f"Unique values: {df[col].unique()} | Value counts:\n{df[col].value_counts()}")

        # Step 2: Remove high-dominance columns
        dominance_cols = []
        for col in df_clean.columns:
            series = df_clean[col].dropna() if exclude_na else df_clean[col]
            if len(series) == 0:
                continue
            top_freq_ratio = series.value_counts(normalize=True).iloc[0] * 100
            if top_freq_ratio > top_feature_prop_threshold:
                dominance_cols.append(col)
                logger.info(f"Dropping column '{col}' due to high dominance ({top_freq_ratio:.2f}%)")

        dropped_columns.extend(dominance_cols)
        df_clean = df_clean.drop(columns=dominance_cols)

        retained_columns = list(df_clean.columns)

        logger.info(f"DataFrame shape after filtering: {df_clean.shape}")
        logger.info(f"Total columns dropped: {len(dropped_columns)}")
        logger.info(f"Remaining columns: {len(retained_columns)}")

        return df_clean, dropped_columns, retained_columns

    
    def remove_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove outliers based on configured ranges"""
        if not self.config.invalid_ranges:
            return df
        
        logger.info("Removing outliers...")
        logger.info(f"DataFrame shape before removing outliers: {df.shape}")
        
        # Create a copy of the dataframe
        df_filtered = df.copy()
        
        for column, (lower, upper) in self.config.invalid_ranges.items():
            if column in df.columns:
                # Create mask for valid values
                mask = (df[column] >= lower) & (df[column] <= upper)
                # Apply mask to all columns including record_id
                df_filtered = df_filtered[mask]
                removed = len(df) - len(df_filtered)
                logger.info(f"Removed {removed} rows for column {column}")
        
        logger.info(f"DataFrame shape after removing outliers: {df_filtered.shape}")
        return df_filtered
    
    def apply_column_transformations(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply standard transformations to columns"""
        logger.info("Applying column transformations...")
        logger.info(f"DataFrame shape before transformations: {df.shape}")
        
        # Height transformation
        if 'height' in df.columns:
            df['height'] = df['height'].apply(
                lambda x: x / 100 if pd.notnull(x) and x > 3 else x
            )
            logger.info("Transformed height to meters")
        
        # ICU admission transformation
        if 'icu_admission_date_and_tim' in df.columns:
            df['icu_admission_date_and_tim'] = df['icu_admission_date_and_tim'].notna().astype(int)
            logger.info("Transformed ICU admission to binary")
        
        logger.info(f"DataFrame shape after transformations: {df.shape}")
        return df
    
    def impute_missing_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing data using appropriate imputation strategies"""
        logger.info("Imputing missing data...")
        logger.info(f"DataFrame shape before imputation: {df.shape}")
        
        # Create a copy of the dataframe to avoid modifying the original
        df_imputed = df.copy()
        
        # Get columns with missing values (exclude record_id)
        missing_columns = df_imputed[self.config.selected_features].columns[
            df_imputed[self.config.selected_features].isnull().any()
        ].tolist()
        
        # Remove record_id from missing_columns if it's there
        if 'record_id' in missing_columns:
            missing_columns.remove('record_id')
        
        # Define imputation strategies for different column types
        median_impute_cols = ['age']
        distribution_impute_cols = ['gender']
        calculation_impute_cols = ['bmi']
        mean_impute_cols = ['days_surgery_discharge', 'duration_of_procedure', 'albumin', 'potassium', 'inr', 'fibrinogen', 'asa', 'duration_of_time_in_recove', 'sas_score']
        drop_missing_cols = ['wcc', 'hb', 'haematocrit', 'platelets', 'creatinine', 'urea', 'sodium', 'albumin', 'inr', 'drg_code', 'icd_prim_code']
        
        # First handle special cases
        for col in missing_columns:
            logger.info(f"Processing missing values for {col}")
            logger.info(f"DataFrame shape before processing {col}: {df_imputed.shape}")
            
            if col in median_impute_cols:
                df_imputed = self._impute_with_median(df_imputed, col)
            elif col in distribution_impute_cols:
                df_imputed = self._impute_with_distribution(df_imputed, col)
            elif col in calculation_impute_cols:
                df_imputed = self._impute_with_calculation(df_imputed, col)
            elif col in mean_impute_cols:
                df_imputed = self._impute_with_mean(df_imputed, col)
            elif col in drop_missing_cols:
                df_imputed = self._drop_missing_rows(df_imputed, col)
            else:
                # Handle remaining columns based on imputation method
                if self.config.impute_method == 'normal':
                    df_imputed = self._impute_with_normal_distribution(df_imputed, col)
                elif self.config.impute_method == 'mean':
                    df_imputed = self._impute_with_mean(df_imputed, col)
                else:
                    logger.warning(f"No imputation strategy defined for {col}")
            
            logger.info(f"DataFrame shape after processing {col}: {df_imputed.shape}")
        
        # Check for any remaining missing values (excluding record_id)
        missing_after = df_imputed[self.config.selected_features].drop(columns=['record_id']).isnull().mean()
        if any(missing_after > 0):
            logger.warning(f"Some missing values remain after imputation:\n{missing_after}")
        else:
            logger.info("All missing values have been handled or imputed successfully.")
        
        logger.info(f"DataFrame shape after imputation: {df_imputed.shape}")
        return df_imputed
    
    def _impute_with_median(self, df: pd.DataFrame, col: str) -> pd.DataFrame:
        """Impute missing values with median"""
        logger.info(f"Imputing {col} with the median value")
        logger.info(f"DataFrame shape before median imputation: {df.shape}")
        
        # Convert column to numeric, coercing errors to NaN
        df[col] = pd.to_numeric(df[col], errors='coerce')
        
        median_value = df[col].median()
        logger.info(f"Median {col} is {median_value}")
        df[col].fillna(median_value, inplace=True)
        
        logger.info(f"DataFrame shape after median imputation: {df.shape}")
        return df
    
    def _impute_with_distribution(self, df: pd.DataFrame, col: str) -> pd.DataFrame:
        """Impute missing values based on distribution of existing values"""
        logger.info(f"Imputing {col} with the value that keeps proportion same as before imputation")
        logger.info(f"DataFrame shape before distribution imputation: {df.shape}")
        
        # Get the distribution of existing values
        value_dist = df[col].dropna().value_counts(normalize=True)
        logger.info(f"Original {col} Distribution:\n{value_dist}")

        # Randomly impute missing values based on distribution
        missing_values = np.random.choice(
            value_dist.index, 
            size=df[col].isna().sum(), 
            p=value_dist.values
        )
        df.loc[df[col].isna(), col] = missing_values
        
        logger.info(f"DataFrame shape after distribution imputation: {df.shape}")
        logger.info(f"Final {col} Distribution:\n{df[col].value_counts(normalize=True)}")
        return df
    
    def _impute_with_calculation(self, df: pd.DataFrame, col: str) -> pd.DataFrame:
        """Impute missing values using calculations from other columns"""
        logger.info(f"DataFrame shape before calculation imputation: {df.shape}")
        
        if col == 'bmi':
            # Convert height and weight to numeric, coercing errors to NaN
            df['height'] = pd.to_numeric(df['height'], errors='coerce')
            df['weight'] = pd.to_numeric(df['weight'], errors='coerce')
            
            missing_mask = df[col].isnull()
            valid_mask = (
                (df['height'].notnull()) & 
                (df['weight'].notnull()) & 
                (df['height'] != 0) & 
                (df['weight'] != 0)
            )
            
            # Calculate BMI where possible
            df.loc[missing_mask & valid_mask, col] = (
                df['weight'] / (df['height'] ** 2)
            )
            
            # Drop rows where calculation wasn't possible
            df = df.dropna(subset=[col])
            logger.info(f"Imputed {col} using height and weight calculations")
        
        logger.info(f"DataFrame shape after calculation imputation: {df.shape}")
        return df
    
    def _impute_with_mean(self, df: pd.DataFrame, col: str) -> pd.DataFrame:
        """Impute missing values with mean"""
        logger.info(f"Imputing Mean for missing values of col {col}")
        logger.info(f"DataFrame shape before mean imputation: {df.shape}")
        
        # Convert column to numeric, coercing errors to NaN
        df[col] = pd.to_numeric(df[col], errors='coerce')
        
        imputer = SimpleImputer(strategy='mean')
        df[col] = imputer.fit_transform(df[[col]])
        
        logger.info(f"DataFrame shape after mean imputation: {df.shape}")
        return df
    
    def _drop_missing_rows(self, df: pd.DataFrame, col: str) -> pd.DataFrame:
        """Drop rows with missing values for specified column"""
        logger.info(f"Dropping rows with missing values for {col}")
        logger.info(f"DataFrame shape before dropping rows: {df.shape}")
        
        df = df.dropna(subset=[col])
        
        logger.info(f"DataFrame shape after dropping rows: {df.shape}")
        return df
    
    def _impute_with_normal_distribution(self, df: pd.DataFrame, col: str) -> pd.DataFrame:
        """Impute missing values using normal distribution based on existing values"""
        logger.info(f"Imputing {col} using normal distribution")
        logger.info(f"DataFrame shape before normal distribution imputation: {df.shape}")
        
        # Convert column to numeric, coercing errors to NaN
        df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Get non-null values
        non_null_values = df[col].dropna()
        
        if len(non_null_values) < 2:
            logger.warning(f"Insufficient data for {col} to perform normal distribution imputation, using mean instead")
            mean_val = non_null_values.mean() if len(non_null_values) > 0 else 0
            df[col] = df[col].fillna(mean_val)
            logger.info(f"DataFrame shape after fallback mean imputation: {df.shape}")
            return df
        
        # Calculate mean and standard deviation
        mean_val = non_null_values.mean()
        std_val = non_null_values.std()
        
        # Get indices of null values
        null_indices = df[col].isnull()
        null_count = null_indices.sum()
        
        logger.info(f"Generating {null_count} normal distribution random values for {col} (mean={mean_val:.2f}, std={std_val:.2f})")
        
        # Generate random values from normal distribution
        random_values = np.random.normal(mean_val, std_val, size=null_count)
        
        # Ensure generated values are within reasonable bounds (e.g., non-negative for certain columns)
        if col in ['days_surgery_discharge', 'duration_of_procedure', 'duration_of_time_in_recove']:
            random_values = np.maximum(random_values, 0)
        
        # Replace null values with generated values
        df.loc[null_indices, col] = random_values
        
        logger.info(f"DataFrame shape after normal distribution imputation: {df.shape}")
        logger.info(f"Normal distribution imputation completed for {col}")
        return df
    
    def analyze_feature_importance(self, X: pd.DataFrame, y: pd.Series, task: str = 'classification', 
                                 method: str = 'mutual_info', top_n: int = 10) -> pd.DataFrame:
        """
        Analyze and return feature importance using specified method.

        Parameters:
            X (pd.DataFrame): Feature matrix
            y (pd.Series or np.array): Target vector
            task (str): 'classification' or 'regression'
            method (str): 'mutual_info', 'random_forest', or 'linear_model'
            top_n (int): Number of top features to return

        Returns:
            pd.DataFrame: Feature importance scores sorted by importance
        """
        if task not in ['classification', 'regression']:
            raise ValueError("task must be 'classification' or 'regression'")

        if method == 'mutual_info':
            if task == 'classification':
                scores = mutual_info_classif(X, y, discrete_features='auto')
            else:
                scores = mutual_info_regression(X, y)
            importance_df = pd.DataFrame({'feature': X.columns, 'importance': scores})

        elif method == 'random_forest':
            model = RandomForestClassifier() if task == 'classification' else RandomForestRegressor()
            model.fit(X, y)
            importance_df = pd.DataFrame({'feature': X.columns, 'importance': model.feature_importances_})

        elif method == 'linear_model':
            model = LogisticRegression(max_iter=1000) if task == 'classification' else LinearRegression()
            model.fit(X, y)
            coefs = model.coef_[0] if task == 'classification' else model.coef_
            importance_df = pd.DataFrame({'feature': X.columns, 'importance': np.abs(coefs)})

        else:
            raise ValueError("method must be one of: 'mutual_info', 'random_forest', 'linear_model'")

        return importance_df.sort_values(by='importance', ascending=False).head(top_n)
    
    def detect_categorical_columns(self, df: pd.DataFrame) -> List[str]:
        """
        Automatically detect categorical columns based on:
        1. Number of unique values
        2. Data type
        3. Column name patterns
        4. Value distribution
        """
        logger.info("Detecting categorical columns...")
        
        categorical_cols = []
        
        for col in df.columns:
            # Skip record_id and target column
            if col in ['record_id', self.config.target_columns]:
                continue
                
            # Get number of unique values
            n_unique = df[col].nunique()
            
            # Check if column is already categorical type
            is_categorical_type = pd.api.types.is_categorical_dtype(df[col]) or \
                                pd.api.types.is_object_dtype(df[col])
            
            # Check column name patterns
            is_categorical_name = any(pattern in col.lower() for pattern in [
                'gender', 'type', 'code', 'flag', 'status', 'category', 'class',
                'priority', 'level', 'grade', 'score', 'stage', 'phase', 'cci_'
            ])
            
            # Check value distribution
            if n_unique > 0:
                value_distribution = df[col].value_counts(normalize=True)
                is_uniform_distribution = value_distribution.std() < 0.3  # Low standard deviation suggests categorical
            
            # Determine if column is categorical
            is_categorical = (
                # Binary columns (2 unique values)
                n_unique == 2 or
                # Low cardinality categorical columns
                (n_unique >= self.config.min_unique_values_for_categorical and
                 (is_categorical_type or is_categorical_name or is_uniform_distribution)) or
                # Explicitly categorical columns
                is_categorical_type
            )
            
            if is_categorical:
                categorical_cols.append(col)
                logger.info(f"Detected categorical column: {col} (unique values: {n_unique})")
                if n_unique <= 5:  # Log value distribution for low cardinality columns
                    logger.info(f"Value distribution for {col}:")
                    logger.info(df[col].value_counts(normalize=True).to_string())
        
        logger.info(f"Detected {len(categorical_cols)} categorical columns")
        return categorical_cols

    def encode_categorical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical features:
        - Binary columns (2 unique values) -> 0/1
        - Categorical columns (more than 2 unique values) -> one-hot encoding
        """
        if not self.config.encode_categorical:
            return df
        
        logger.info("Encoding categorical features...")
        logger.info(f"DataFrame shape before encoding: {df.shape}")
        
        # Automatically detect categorical columns
        categorical_cols = self.detect_categorical_columns(df)
        
        # Filter categorical columns to only include those in the current dataframe
        categorical_cols = [col for col in categorical_cols if col in df.columns]
        
        if not categorical_cols:
            logger.info("No categorical columns to encode")
            return df
        
        # Create a copy of the dataframe
        df_encoded = df.copy()
        
        # Separate binary and categorical columns
        binary_cols = []
        multi_cat_cols = []
        
        for col in categorical_cols:
            n_unique = df[col].nunique()
            if n_unique == 2:
                binary_cols.append(col)
            else:
                multi_cat_cols.append(col)
        
        logger.info(f"Found {len(binary_cols)} binary columns and {len(multi_cat_cols)} categorical columns")
        
        # 1. Handle binary columns (convert to 0/1)
        for col in binary_cols:
            logger.info(f"Converting binary column {col} to 0/1")
            # Get unique values
            unique_vals = df[col].unique()
            # Create mapping (first value -> 0, second value -> 1)
            mapping = {unique_vals[0]: 0, unique_vals[1]: 1}
            # Apply mapping
            df_encoded[col] = df_encoded[col].map(mapping)
            logger.info(f"Mapping for {col}: {mapping}")
        
        # 2. Handle multi-category columns (one-hot encoding)
        if multi_cat_cols:
            # Initialize OneHotEncoder
            encoder = OneHotEncoder(
                sparse_output=False,
                drop='first' if self.config.drop_first else None,
                handle_unknown='ignore'
            )
            
            try:
                # Get categorical data
                cat_data = df_encoded[multi_cat_cols]
                
                # Fit and transform
                encoded_data = encoder.fit_transform(cat_data)
                
                # Create new column names
                new_columns = []
                for i, col in enumerate(multi_cat_cols):
                    categories = encoder.categories_[i]
                    if self.config.drop_first:
                        categories = categories[1:]
                    new_columns.extend([f"{col}_{cat}" for cat in categories])
                
                # Create DataFrame with encoded data
                encoded_df = pd.DataFrame(
                    encoded_data,
                    columns=new_columns,
                    index=df_encoded.index
                )
                
                # Drop original categorical columns and add encoded ones
                df_encoded = df_encoded.drop(columns=multi_cat_cols)
                df_encoded = pd.concat([df_encoded, encoded_df], axis=1)
                
                logger.info(f"One-hot encoded {len(multi_cat_cols)} categorical columns into {len(new_columns)} binary columns")
                
                # Log the mapping of original to encoded columns
                for i, col in enumerate(multi_cat_cols):
                    categories = encoder.categories_[i]
                    if self.config.drop_first:
                        categories = categories[1:]
                    logger.info(f"\nOne-hot encoding for {col}:")
                    for cat in categories:
                        logger.info(f"  {col}_{cat}")
                
            except Exception as e:
                logger.error(f"Error in categorical encoding: {str(e)}")
                raise
        
        logger.info(f"Final DataFrame shape after encoding: {df_encoded.shape}")
        return df_encoded

    def scale_numerical_features(self, df: pd.DataFrame, categorical_cols: List[str]) -> pd.DataFrame:
        """Scale numerical features using StandardScaler"""
        logger.info("Scaling numerical features...")
        logger.info(f"DataFrame shape before scaling: {df.shape}")
        
        # Create a copy of the dataframe
        df_scaled = df.copy()
        
        # Detect categorical columns
        # categorical_cols = self.detect_categorical_columns(df)
        
        # Identify numerical columns
        numerical_cols = [col for col in df.columns 
                        if col not in categorical_cols 
                        and col != 'record_id' 
                        and col not in self.config.target_columns
                        and pd.api.types.is_numeric_dtype(df[col])]
        
        if not numerical_cols:
            logger.info("No numerical columns to scale")
            return df_scaled
        
        try:
            # Initialize StandardScaler
            scaler = StandardScaler()
            
            # Fit and transform numerical columns
            df_scaled[numerical_cols] = scaler.fit_transform(df[numerical_cols])
            
            # Log scaling information
            logger.info(f"Scaled {len(numerical_cols)} numerical features")
            logger.info("Numerical columns scaled:")
            for col in numerical_cols:
                logger.info(f"  {col}")
            
            # Log scaling statistics
            scaling_stats = pd.DataFrame({
                'Feature': numerical_cols,
                'Mean': scaler.mean_,
                'Scale': scaler.scale_
            })
            logger.info("\nScaling statistics:")
            logger.info(scaling_stats.to_string())
            
            logger.info(f"DataFrame shape after scaling: {df_scaled.shape}")
            return df_scaled
            
        except Exception as e:
            logger.error(f"Error in scaling numerical features: {str(e)}")
            raise

    def convert_categorical_to_numeric(self, X):
        """
        Converts all categorical (non-numeric) columns in the DataFrame to numeric using encoding.
        - For categorical columns, uses Label Encoding or One-Hot Encoding depending on the type of variable.

        Parameters:
        -----------
        X : pandas.DataFrame
            The feature matrix containing categorical and numeric features.

        Returns:
        --------
        X_encoded : pandas.DataFrame
            DataFrame with all categorical variables converted to numeric format.
        """
        print("Categorical columns:")
        print(X.select_dtypes(include=['object', 'category']).columns)
        # Iterate over columns to identify categorical features
        for col in X.select_dtypes(include=['object', 'category']).columns:
            print(col)
            # If the column has more than 2 unique values, we apply One-Hot Encoding
            if X[col].nunique() > 2:
                print(f"Applying One-Hot Encoding to '{col}'")
                X = pd.get_dummies(X, columns=[col], drop_first=True)
            else:
                # Apply Label Encoding for binary categorical columns
                print(f"Applying Label Encoding to '{col}'")
                X[col] = X[col].astype('category').cat.codes
        
        return X
    
    def preprocess_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess features including encoding categorical variables"""
        logger.info("Preprocessing features...")
        
        # 1. Apply column transformations first
        df = self.apply_column_transformations(df)
        if 'record_id' not in df.columns:
            raise ValueError("record_id column missing after column transformations")
        logger.info("Applied column transformations")
        
        # 2. Remove outliers
        df = self.remove_outliers(df)
        if 'record_id' not in df.columns:
            raise ValueError("record_id column missing after removing outliers")
        logger.info("Removed outliers")
        
        # 3. Impute missing data
        df = self.impute_missing_data(df)
        if 'record_id' not in df.columns:
            raise ValueError("record_id column missing after imputation")
        logger.info("Imputed missing data")
        
        # 4. Filter zero variance columns after transformations and imputation
        df = self.filter_zero_variance_columns(df)
        if 'record_id' not in df.columns:
            raise ValueError("record_id column missing after filtering zero variance columns")
        logger.info("Filtered zero variance columns")
        
        # 5. Select only the features we want (after all cleaning steps)
        # First check which selected features are actually in the dataframe
        available_features = [col for col in self.config.selected_features if col in df.columns]
        missing_features = set(self.config.selected_features) - set(available_features)
        if missing_features:
            logger.warning(f"\nThe following selected features are not available in the dataframe and will be skipped: {missing_features}")
        
        # Update selected_features to only include available columns
        self.config.selected_features = available_features
        df = df[self.config.selected_features]
        if 'record_id' not in df.columns:
            raise ValueError("record_id column missing after feature selection")
        logger.info(f"Selected {len(self.config.selected_features)} features")
        
        df = self.convert_categorical_to_numeric(df)
        
        # df = self.encode_categorical_features(df)
        # # 6. Encode categorical features
        # df = self.encode_categorical_features(df)
        # if 'record_id' not in df.columns:
        #     raise ValueError("record_id column missing after categorical encoding")
        # logger.info("Encoded categorical features")
        
        # # 7. Scale numerical features
        # df = self.scale_numerical_features(df)
        # if 'record_id' not in df.columns:
        #     raise ValueError("record_id column missing after numerical scaling")
        # logger.info("Scaled numerical features")
        
        return df
    
    def has_value_in_columns(self, df, columns):
        """
        Check if there are non-null values in the specified columns.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame to check columns in
        columns : list
            List of column names to check
            
        Returns:
        --------
        pandas.Series
            Boolean series indicating whether each row has any non-null values in the specified columns
        """
        valid_cols = [col for col in columns if col in df.columns]
        if not valid_cols:
            return pd.Series(False, index=df.index)
        return df[valid_cols].notna().any(axis=1)
    
    # Add various disease multiclass feature creation methods
    def create_heart_failure_multiclass(self, df_index, df_index_selected, columns):
        """
        Create multiclass features for heart failure.
        
        Parameters:
        -----------
        df_index : pandas.DataFrame
            Original DataFrame containing index events
        df_index_selected : pandas.DataFrame
            DataFrame storing processed features
        columns : list
            List of columns related to heart failure
        """
        print("Creating heart failure multiclass features...")
        
        # Define columns related to specific categories
        nyha_columns = [col for col in columns if 'nyha' in col.lower()]
        
        # Initialize multiclass feature
        df_index_selected['heart_failure_class'] = 0
        
        # First mark all heart failure records as unspecified (1)
        mask_hf = self.has_value_in_columns(df_index, columns)
        df_index_selected.loc[mask_hf, 'heart_failure_class'] = 1
        
        # NYHA Class I (2)
        mask_nyha1 = self.has_value_in_columns(df_index, [col for col in nyha_columns if 'i' in col.lower() and 'ii' not in col.lower()])
        df_index_selected.loc[mask_nyha1, 'heart_failure_class'] = 2
        
        # NYHA Class II (3)
        mask_nyha2 = self.has_value_in_columns(df_index, [col for col in nyha_columns if 'ii' in col.lower() and 'iii' not in col.lower()])
        df_index_selected.loc[mask_nyha2, 'heart_failure_class'] = 3
        
        # NYHA Class III (4)
        mask_nyha3 = self.has_value_in_columns(df_index, [col for col in nyha_columns if 'iii' in col.lower() and 'iv' not in col.lower()])
        df_index_selected.loc[mask_nyha3, 'heart_failure_class'] = 4
        
        # NYHA Class IV (5)
        mask_nyha4 = self.has_value_in_columns(df_index, [col for col in nyha_columns if 'iv' in col.lower()])
        df_index_selected.loc[mask_nyha4, 'heart_failure_class'] = 5
        
        # Output distribution
        class_counts = df_index_selected['heart_failure_class'].value_counts().sort_index()
        print("Heart failure class distribution:")
        for cls, count in class_counts.items():
            print(f"  Class {cls}: {count} rows")
    
    def create_hypertension_multiclass(self, df_index, df_index_selected, columns):
        """
        Create multiclass features for hypertension.
        
        Parameters:
        -----------
        df_index : pandas.DataFrame
            Original DataFrame containing index events
        df_index_selected : pandas.DataFrame
            DataFrame storing processed features
        columns : list
            List of columns related to hypertension
        """
        print("Creating hypertension multiclass features...")
        
        # Define columns related to treatment
        treated_columns = [col for col in columns if 'treated' in col.lower()]
        untreated_columns = [col for col in columns if 'untreated' in col.lower()]
        partially_treated_columns = [col for col in columns if 'partially' in col.lower()]
        
        # Initialize multiclass feature
        df_index_selected['hypertension_class'] = 0
        
        # First mark all hypertension records as unspecified (1)
        mask_htn = self.has_value_in_columns(df_index, columns)
        df_index_selected.loc[mask_htn, 'hypertension_class'] = 1
        
        # Treated hypertension (2)
        mask_treated = self.has_value_in_columns(df_index, treated_columns)
        df_index_selected.loc[mask_treated, 'hypertension_class'] = 2
        
        # Untreated hypertension (3)
        mask_untreated = self.has_value_in_columns(df_index, untreated_columns)
        df_index_selected.loc[mask_untreated, 'hypertension_class'] = 3
        
        # Partially treated hypertension (4)
        mask_partial = self.has_value_in_columns(df_index, partially_treated_columns)
        df_index_selected.loc[mask_partial, 'hypertension_class'] = 4
        
        # Output distribution
        class_counts = df_index_selected['hypertension_class'].value_counts().sort_index()
        print("Hypertension class distribution:")
        for cls, count in class_counts.items():
            print(f"  Class {cls}: {count} rows")

    def create_heart_disease_multiclass(self, df_index, df_index_selected, columns):
        """
        Create multiclass features for heart disease.
        
        Parameters:
        -----------
        df_index : pandas.DataFrame
            Original DataFrame containing index events
        df_index_selected : pandas.DataFrame
            DataFrame storing processed features
        columns : list
            List of columns related to heart disease
        """
        print("Creating heart disease multiclass features...")
        
        # Define columns related to specific categories
        angina_columns = [col for col in columns if 'angina' in col.lower()]
        ccs_columns = [col for col in columns if 'ccs' in col.lower()]
        
        # Initialize multiclass feature
        df_index_selected['heart_disease_class'] = 0
        
        # First mark all heart disease records as unspecified (1)
        mask_hd = self.has_value_in_columns(df_index, columns)
        df_index_selected.loc[mask_hd, 'heart_disease_class'] = 1
        
        # No angina (2)
        mask_no_angina = self.has_value_in_columns(df_index, [col for col in angina_columns if 'no_angina' in col.lower()])
        df_index_selected.loc[mask_no_angina, 'heart_disease_class'] = 2
        
        # CCS Class I (3)
        mask_ccs1 = self.has_value_in_columns(df_index, [col for col in ccs_columns if 'i' in col.lower() and 'ii' not in col.lower()])
        df_index_selected.loc[mask_ccs1, 'heart_disease_class'] = 3
        
        # CCS Class II (4)
        mask_ccs2 = self.has_value_in_columns(df_index, [col for col in ccs_columns if 'ii' in col.lower() and 'iii' not in col.lower()])
        df_index_selected.loc[mask_ccs2, 'heart_disease_class'] = 4
        
        # CCS Class III (5)
        mask_ccs3 = self.has_value_in_columns(df_index, [col for col in ccs_columns if 'iii' in col.lower() and 'iv' not in col.lower()])
        df_index_selected.loc[mask_ccs3, 'heart_disease_class'] = 5
        
        # CCS Class IV (6)
        mask_ccs4 = self.has_value_in_columns(df_index, [col for col in ccs_columns if 'iv' in col.lower()])
        df_index_selected.loc[mask_ccs4, 'heart_disease_class'] = 6
        
        # Output distribution
        class_counts = df_index_selected['heart_disease_class'].value_counts().sort_index()
        print("Heart disease class distribution:")
        for cls, count in class_counts.items():
            print(f"  Class {cls}: {count} rows")
    
    def create_diabetes_multiclass(self, df_index, df_index_selected, columns):
        """
        Create multiclass features for diabetes.
        
        Parameters:
        -----------
        df_index : pandas.DataFrame
            Original DataFrame containing index events
        df_index_selected : pandas.DataFrame
            DataFrame storing processed features
        columns : list
            List of columns related to diabetes
        """
        print("Creating diabetes multiclass features...")
        
        # Define columns related to specific types
        type1_columns = [col for col in columns if 'type1' in col.lower()]
        type2_columns = [col for col in columns if 'type2' in col.lower()]
        complication_columns = [col for col in columns if 'end_organ_damage' in col.lower()]
        
        # Initialize multiclass feature
        df_index_selected['diabetes_class'] = 0
        
        # First mark all diabetes records as unspecified (1)
        mask_diabetes = self.has_value_in_columns(df_index, columns)
        df_index_selected.loc[mask_diabetes, 'diabetes_class'] = 1
        
        # Type 1 diabetes (2)
        mask_type1 = self.has_value_in_columns(df_index, type1_columns)
        df_index_selected.loc[mask_type1, 'diabetes_class'] = 2
        
        # Type 2 diabetes (3)
        mask_type2 = self.has_value_in_columns(df_index, type2_columns)
        df_index_selected.loc[mask_type2, 'diabetes_class'] = 3
        
        # Diabetes with complications (4)
        mask_complications = self.has_value_in_columns(df_index, complication_columns)
        df_index_selected.loc[mask_complications & mask_diabetes, 'diabetes_class'] = 4
        
        # Output distribution
        class_counts = df_index_selected['diabetes_class'].value_counts().sort_index()
        print("Diabetes class distribution:")
        for cls, count in class_counts.items():
            print(f"  Class {cls}: {count} rows")

    def create_cancer_multiclass(self, df_index, df_index_selected, columns):
        """
        Create multiclass features for cancer.
        
        Parameters:
        -----------
        df_index : pandas.DataFrame
            Original DataFrame containing index events
        df_index_selected : pandas.DataFrame
            DataFrame storing processed features
        columns : list
            List of columns related to cancer
        """
        print("Creating cancer multiclass features...")
        
        # Define columns related to specific types
        unspecified_columns = [col for col in columns if 'unspecified' in col.lower()]
        primary_columns = [col for col in columns if 'primary' in col.lower() or 'localised' in col.lower()]
        metastatic_columns = [col for col in columns if 'metastatic' in col.lower() or 'secondary' in col.lower() or 'mets' in col.lower()]
        
        # Initialize multiclass feature
        df_index_selected['cancer_class'] = 0
        
        # First mark all cancer records as unspecified (1)
        mask_cancer = self.has_value_in_columns(df_index, columns)
        df_index_selected.loc[mask_cancer, 'cancer_class'] = 1
        
        # Primary/localized cancer (2)
        mask_primary = self.has_value_in_columns(df_index, primary_columns)
        df_index_selected.loc[mask_primary, 'cancer_class'] = 2
        
        # Metastatic cancer (3)
        mask_metastatic = self.has_value_in_columns(df_index, metastatic_columns)
        df_index_selected.loc[mask_metastatic, 'cancer_class'] = 3
        
        # Output distribution
        class_counts = df_index_selected['cancer_class'].value_counts().sort_index()
        print("Cancer class distribution:")
        for cls, count in class_counts.items():
            print(f"  Class {cls}: {count} rows")
    
    def create_cerebrovascular_multiclass(self, df_index, df_index_selected, columns):
        """
        Create multiclass features for cerebrovascular disease.
        
        Parameters:
        -----------
        df_index : pandas.DataFrame
            Original DataFrame containing index events
        df_index_selected : pandas.DataFrame
            DataFrame storing processed features
        columns : list
            List of columns related to cerebrovascular disease
        """
        print("Creating cerebrovascular disease multiclass features...")
        
        # Define columns related to specific types
        stroke_columns = [col for col in columns if 'stroke' in col.lower()]
        tia_columns = [col for col in columns if 'tia' in col.lower()]
        
        # Initialize multiclass feature
        df_index_selected['cerebrovascular_class'] = 0
        
        # First mark all cerebrovascular disease records as unspecified (1)
        mask_cerebrovascular = self.has_value_in_columns(df_index, columns)
        df_index_selected.loc[mask_cerebrovascular, 'cerebrovascular_class'] = 1
        
        # Stroke (2)
        mask_stroke = self.has_value_in_columns(df_index, stroke_columns)
        df_index_selected.loc[mask_stroke, 'cerebrovascular_class'] = 2
        
        # Transient Ischemic Attack (TIA) (3)
        mask_tia = self.has_value_in_columns(df_index, tia_columns)
        df_index_selected.loc[mask_tia, 'cerebrovascular_class'] = 3
        
        # Both stroke and TIA (4)
        df_index_selected.loc[mask_stroke & mask_tia, 'cerebrovascular_class'] = 4
        
        # Output distribution
        class_counts = df_index_selected['cerebrovascular_class'].value_counts().sort_index()
        print("Cerebrovascular disease class distribution:")
        for cls, count in class_counts.items():
            print(f"  Class {cls}: {count} rows")
    
    def load_target_feature(self) -> pd.DataFrame:
        """
        Load target features from preprocessed target file.
        
        Returns:
        --------
        pandas.DataFrame or None
            DataFrame containing target features (if the file exists), otherwise None
        """
        if not self.config.target_file:
            logger.warning("No target file specified in config")
            return None
            
        if not os.path.exists(self.config.target_file):
            logger.warning(f"Warning: Target feature file not found: {self.config.target_file}")
            return None
            
        logger.info(f"Loading target features from {self.config.target_file}...")
        target_features = pd.read_csv(self.config.target_file)
        
        # Ensure record_id column is included
        if 'record_id' not in target_features.columns:
            logger.error("Error: 'record_id' column not found in target feature file")
            return None
        
        # Check which columns are available
        available_columns = ['record_id']
        
        # Add target columns based on what's available
        target_columns = ['dd_3month', 'dd_6month', 'los_target', '180_readmission']

        for col in target_columns:
            if col in target_features.columns:
                available_columns.append(col)
        
        # Only select needed columns
        target_features = target_features[available_columns]
        logger.info(f"Target features loaded, shape: {target_features.shape}")
        
        # Show statistics for target features
        target_cols = [col for col in target_features.columns if col != 'record_id']
        
        if len(target_cols) > 0:
            # Count non-zero values for each feature
            non_zero_counts = (target_features[target_cols] != 0).sum()
            
            logger.info("\nTarget feature statistics:")
            for feature, count in non_zero_counts.items():
                logger.info(f"  {feature}: {count} non-zero values ({count/len(target_features)*100:.2f}%)")
        
        return target_features

    def process(self) -> pd.DataFrame:
        """Main processing pipeline"""
        try:
            # Load and process data
            df_index = self.load_data()
            df_index_selected = self.preprocess_features(df_index)
            
            # Check if there are relevant columns in the dataset and create multiclass features
            # Heart failure multiclass features
            hf_cols = [col for col in df_index.columns if 'heart_failure' in col.lower() or 'nyha' in col.lower()]
            if hf_cols:
                self.create_heart_failure_multiclass(df_index, df_index_selected, hf_cols)
            
            # Hypertension multiclass features
            htn_cols = [col for col in df_index.columns if 'hypertension' in col.lower() or 'htn' in col.lower()]
            if htn_cols:
                self.create_hypertension_multiclass(df_index, df_index_selected, htn_cols)
            
            # Heart disease multiclass features
            hd_cols = [col for col in df_index.columns if ('heart_disease' in col.lower() or 'angina' in col.lower() or 'ccs' in col.lower()) and 'failure' not in col.lower()]
            if hd_cols:
                self.create_heart_disease_multiclass(df_index, df_index_selected, hd_cols)
            
            # Diabetes multiclass features
            dm_cols = [col for col in df_index.columns if 'diabetes' in col.lower() or 'dm' in col.lower()]
            if dm_cols:
                self.create_diabetes_multiclass(df_index, df_index_selected, dm_cols)
            
            # Cancer multiclass features
            cancer_cols = [col for col in df_index.columns if 'cancer' in col.lower() or 'tumor' in col.lower() or 'metastatic' in col.lower()]
            if cancer_cols:
                self.create_cancer_multiclass(df_index, df_index_selected, cancer_cols)
            
            # Cerebrovascular disease multiclass features
            cerebro_cols = [col for col in df_index.columns if 'cerebrovascular' in col.lower() or 'stroke' in col.lower() or 'tia' in col.lower()]
            if cerebro_cols:
                self.create_cerebrovascular_multiclass(df_index, df_index_selected, cerebro_cols)
        
            icdpreprocessor = IcdsPreprocessing()
        
            df_icd_transformed = icdpreprocessor.preprocess_icd_data(df_index)
            disease_present_columns = [col for col in df_icd_transformed.columns if col.endswith('_disease_present')] + ['record_id']
            
            
            df_index_selected = pd.merge(df_index_selected, df_icd_transformed[disease_present_columns], on='record_id', how='left')
            
            print(df_index_selected.columns)
            # Load and integrate target features if available
            target_features = self.load_target_feature()
            if target_features is not None:
                logger.info("Integrating target features with main dataset...")
                df = pd.merge(df_index_selected, target_features, on='record_id', how='left')
                logger.info(f"Dataset shape after integrating target features: {df.shape}")
            
            # Save processed data
            timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
            output_file = f"MIDAS_preprocessed_{timestamp}.csv"
            df.to_csv(output_file, index=False)
            logger.info(f"Processed data saved to {output_file}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error in data processing: {str(e)}")
            raise

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Process medical data')
    parser.add_argument('input_file', type=str, help='Path to input data file')
    parser.add_argument('--target_file', type=str, help='Path to target features file')
    parser.add_argument('--encode_categorical', action='store_true', 
                      help='Whether to encode categorical features')
    parser.add_argument('--drop_first', action='store_true',
                      help='Whether to drop first category in one-hot encoding')
    parser.add_argument('--max_unique_values', type=int, default=10,
                      help='Maximum number of unique values to consider as categorical')
    parser.add_argument('--min_unique_values', type=int, default=2,
                      help='Minimum number of unique values to consider as categorical')
    
    args = parser.parse_args()
    
    # Create configuration
    config = DataConfig(
        input_file=args.input_file,
        output_file=None,  # Will be generated with timestamp
        target_file=args.target_file,
        target_columns=None,
        encode_categorical=args.encode_categorical,
        drop_first=args.drop_first,
        max_unique_values_for_categorical=args.max_unique_values,
        min_unique_values_for_categorical=args.min_unique_values
    )
    
    # Process data
    processor = DataProcessor(config)
    processor.process()

if __name__ == "__main__":
    main()
       
# processed_csv.fetch_binary_cols()
# processed_csv.check_null_in_cat_cols()


#drg_desc ?? as we already have drg_code
