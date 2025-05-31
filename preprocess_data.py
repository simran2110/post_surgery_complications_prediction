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
# from modules.Old_model_Results.plot_metrics import *
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
    icd_file: str = 'processed_features.csv'  # Path to ICD features file
    target_columns: List[str] = None  # List of target column names
    
    # Feature selection
    selected_features: List[str] = None
    
    # Data validation
    invalid_ranges: Dict[str, Tuple[float, float]] = None
    exclude_icds: List[str] = None
    
    # Imputation settings
    numeric_imputation_strategy: str = 'mean'
    categorical_imputation_strategy: str = 'most_frequent'
    impute_method: str = 'normal'
    
    # Encoding settings
    encode_categorical: bool = True
    deep_learning: bool = False
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
                # "days_surgery_discharge",
                "schedule_priority",
                # "duration_of_procedure",
                # "asa",
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
                "cci_total",
                # 'hospital_los',
                'icu_admission_date_and_tim',
                'score1'
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
        initial_cols = len(df.columns)
        logger.info(f"Initial number of columns: {initial_cols}")
        
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
        
        final_cols = len(df.columns)
        logger.info(f"Columns dropped: ")
        print(df.columns.tolist())
        logger.info(f"\nFinal number of columns: {final_cols}")
        logger.info(f"Total columns removed: {initial_cols - final_cols} ({(initial_cols - final_cols) / initial_cols * 100:.2f}%)")
        return df
    
    def remove_low_variance_features(self, df: pd.DataFrame, exclude_na=True, top_feature_prop_threshold=90):
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
        initial_cols = len(df.columns)
        logger.info(f"Initial number of columns: {initial_cols}")

        df_clean = df.copy()
        dropped_columns = []

        # Step 1: Remove zero-variance columns (constant values)
        zero_var_cols = [col for col in df_clean.columns if df_clean[col].nunique(dropna=exclude_na) <= 1]
        if zero_var_cols:
            zero_var_count = len(zero_var_cols)
            zero_var_pct = (zero_var_count / initial_cols) * 100
            logger.info(f"Removing {zero_var_count} zero-variance columns ({zero_var_pct:.2f}%): {zero_var_cols}")
            dropped_columns.extend(zero_var_cols)
            df_clean = df_clean.drop(columns=zero_var_cols)

            selected_dropped = [col for col in zero_var_cols if col in getattr(self.config, 'selected_features', [])]
            if selected_dropped:
                selected_dropped_pct = (len(selected_dropped) / len(self.config.selected_features)) * 100
                logger.warning(f"Selected features dropped due to zero variance: {len(selected_dropped)} ({selected_dropped_pct:.2f}%)")
                for col in selected_dropped:
                    logger.warning(f"  {col}:")
                    logger.warning(f"    Unique values: {df[col].unique()}")
                    logger.warning(f"    Value counts:\n{df[col].value_counts()}")

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

        if dominance_cols:
            dominance_count = len(dominance_cols)
            dominance_pct = (dominance_count / initial_cols) * 100
            logger.info(f"Removing {dominance_count} high-dominance columns ({dominance_pct:.2f}%)")
            dropped_columns.extend(dominance_cols)
            df_clean = df_clean.drop(columns=dominance_cols)

        final_cols = len(df_clean.columns)
        total_dropped = initial_cols - final_cols
        total_dropped_pct = (total_dropped / initial_cols) * 100

        logger.info(f"Final number of columns: {df_clean.columns}")
        logger.info(f"Columns dropped: {list(set(df.columns) - set(df_clean.columns))}")
        logger.info(f"Remaining columns: {len(df_clean.columns)}")
        logger.info(f"Remaining columns: {len(df_clean.columns)}")

        return df_clean, dropped_columns, list(df_clean.columns)
    
    def remove_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove outliers based on configured ranges"""
        if not self.config.invalid_ranges:
            return df
        
        logger.info("Removing outliers...")
        initial_rows = len(df)
        logger.info(f"Initial number of rows: {initial_rows}")
        
        # Create a copy of the dataframe
        df_filtered = df.copy()
        
        for column, (lower, upper) in self.config.invalid_ranges.items():
            if column in df.columns:
                # Create mask for valid values
                mask = (df[column] >= lower) & (df[column] <= upper)
                # Apply mask to all columns including record_id
                df_filtered = df_filtered[mask]
                removed = len(df) - len(df_filtered)
                removed_pct = (removed / len(df)) * 100
                logger.info(f"Column {column}: Removed {removed} rows ({removed_pct:.2f}%)")
        
        final_rows = len(df_filtered)
        total_removed = initial_rows - final_rows
        total_removed_pct = (total_removed / initial_rows) * 100
        logger.info(f"Final number of rows: {final_rows}")
        logger.info(f"Total rows removed: {total_removed} ({total_removed_pct:.2f}%)")
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
    
    def impute_or_drop_missing_data(self, df: pd.DataFrame) -> pd.DataFrame:
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
        median_impute_cols = []
        distribution_impute_cols = []
        calculation_impute_cols = ['bmi']
        mean_impute_cols = []
        normal_impute_cols = ['days_surgery_discharge', 'duration_of_procedure', 'albumin', 'potassium', 'inr', 'fibrinogen', 'asa', 'duration_of_time_in_recove', 'sas_score']
        drop_missing_cols = ['age', 'gender', 'wcc', 'hb', 'haematocrit', 'platelets', 'creatinine', 'urea', 'sodium', 'albumin', 'inr', 'drg_code', 'icd_prim_code', 'schedule_priority']
        zero_impute_cols = ['score1']
        if len(drop_missing_cols):
            print("\n--Drop missing rows")
            missing_pct = df[drop_missing_cols].isnull().mean()
            print("\nPercentage of missing values by feature:")
            for feature, pct in missing_pct.items():
                print(f"{feature}: {pct*100:.2f}%")
                
            df_imputed = self._drop_missing_rows(df_imputed, drop_missing_cols)
            
        if len(median_impute_cols):
            print("\n--Median imputation")
            missing_pct = df[median_impute_cols].isnull().mean()
            print("\nPercentage of missing values by feature:")
            for feature, pct in missing_pct.items():
                print(f"{feature}: {pct*100:.2f}%")
            
            df_imputed = self._impute_with_median(df_imputed, median_impute_cols)
        if len(distribution_impute_cols):
            print("\n--Distribution imputation")
            missing_pct = df[distribution_impute_cols].isnull().mean()
            print("\nPercentage of missing values by feature:")
            for feature, pct in missing_pct.items():
                print(f"{feature}: {pct*100:.2f}%")
                
            df_imputed = self._impute_with_distribution(df_imputed, distribution_impute_cols)
        if len(calculation_impute_cols):
            print("\n--Calculation imputation")
            missing_pct = df[calculation_impute_cols].isnull().mean()
            print("\nPercentage of missing values by feature:")
            for feature, pct in missing_pct.items():
                print(f"{feature}: {pct*100:.2f}%")
                
            df_imputed = self._impute_with_calculation(df_imputed, calculation_impute_cols)
        if len(mean_impute_cols):
            print("\n--Mean imputation")
            missing_pct = df[mean_impute_cols].isnull().mean()
            print("\nPercentage of missing values by feature:")
            for feature, pct in missing_pct.items():
                print(f"{feature}: {pct*100:.2f}%")
                
            df_imputed = self._impute_with_mean(df_imputed, mean_impute_cols)
        if len(normal_impute_cols): 
            print("\n--Normal imputation")
            missing_pct = df[normal_impute_cols].isnull().mean()
            print("\nPercentage of missing values by feature:")
            for feature, pct in missing_pct.items():
                print(f"{feature}: {pct*100:.2f}%")
                
            df_imputed = self._impute_with_normal_distribution(df_imputed, normal_impute_cols)
        
        if len(zero_impute_cols):
            print("\n--Zero imputation")
            missing_pct = df[zero_impute_cols].isnull().mean()
            print("\nPercentage of missing values by feature:")
            for feature, pct in missing_pct.items():
                print(f"{feature}: {pct*100:.2f}%")
                
            df_imputed[zero_impute_cols] = df_imputed[zero_impute_cols].fillna(0)
                
        missing_after = df_imputed[self.config.selected_features].drop(columns=['record_id']).isnull().mean()
        if any(missing_after > 0):
            logger.warning(f"Some missing values remain after imputation:\n{missing_after}")
        else:
            logger.info("All missing values have been handled or imputed successfully.")
        
        logger.info(f"DataFrame shape after imputation: {df_imputed.shape}")
        return df_imputed
    
    def _impute_with_median(self, df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
        """Impute missing values with median for specified columns"""
        logger.info(f"Imputing columns {cols} with median values")
        logger.info(f"DataFrame shape before median imputation: {df.shape}")
        
        for col in cols:
            logger.info(f"Processing column: {col}")
            
            # Convert to numeric, coercing errors to NaN
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
            median_value = df[col].median()
            logger.info(f"Median for {col}: {median_value}")
            
            df[col].fillna(median_value, inplace=True)
        
        logger.info(f"DataFrame shape after median imputation: {df.shape}")
        return df
    
    def _impute_with_distribution(self, df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
        """Impute missing values based on distribution of existing values"""
        logger.info(f"Imputing {cols} with the value that keeps proportion same as before imputation")
        logger.info(f"DataFrame shape before distribution imputation: {df.shape}")
        
        for col in cols:
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
    
    def _impute_with_calculation(self, df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
        """Impute missing values using calculations from other columns"""
        logger.info(f"DataFrame shape before calculation imputation: {df.shape}")
        
        for col in cols:
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
    
    def _impute_with_mean(self, df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
        """Impute missing values with mean"""
        logger.info(f"Imputing Mean for missing values of col {cols}")
        logger.info(f"DataFrame shape before mean imputation: {df.shape}")
        
        for col in cols:
            # Convert column to numeric, coercing errors to NaN
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
            imputer = SimpleImputer(strategy='mean')
            df[col] = imputer.fit_transform(df[[col]])
            
            logger.info(f"DataFrame shape after mean imputation: {df.shape}")
        return df
    
    def _drop_missing_rows(self, df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
        """Drop rows with missing values for specified columns"""
        logger.info(f"Dropping rows with missing values for columns: {cols}")
        initial_rows = len(df)
        logger.info(f"Initial number of rows: {initial_rows}")
        
        df = df.dropna(subset=cols)
        
        final_rows = len(df)
        removed_rows = initial_rows - final_rows
        removed_pct = (removed_rows / initial_rows) * 100
        logger.info(f"Final number of rows: {final_rows}")
        logger.info(f"Rows removed: {removed_rows} ({removed_pct:.2f}%)")
        return df
    
    def _impute_with_normal_distribution(self, df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
        """Impute missing values using normal distribution based on existing values"""
        logger.info(f"Imputing {cols} using normal distribution")
        logger.info(f"DataFrame shape before normal distribution imputation: {df.shape}")
        
        for col in cols:
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
    
    def check_columns_exist(self, df, columns):
        """
        Generic function to check if columns exist in dataframe
        
        Parameters:
        -----------
        df : pandas.DataFrame
            The dataframe to check columns in
        columns : list
            List of column names to check for
        
        Returns:
        --------
        tuple : (bool, list)
            Boolean indicating if any columns exist, and list of existing columns
        """
        # Check which columns exist in the dataframe
        existing_columns = [col for col in columns if col in df.columns]
        
        # Print information about found columns
        if existing_columns:
            print(f"\nFound {len(existing_columns)} columns out of {len(columns)} possible columns:")
            for col in existing_columns:
                print(f"  - {col}")
        else:
            print(f"\nNone of the {len(columns)} specified columns were found in the dataframe")
        
        return len(existing_columns) > 0, existing_columns
    
    def create_heart_failure_combined_features(self, df_index, df_index_selected):
        """
        Create hierarchical multiclass features for heart failure
        Starting with highest NYHA class and moving down to diagnosis
        """
        # Define all possible columns
        all_hf_columns = {
            'diagnosis': [
                'heart_failur_icd_10',
                'heart_failure_pac',
                'heart_failure_pac_powerfor'
            ],
            'nyha': [
                'slight_limitation_nyha_ii',
                'marked_limitation_nyha_iii',
                'symptoms_at_rest_nyha_iv'
            ],
            'symptoms': [
                'orthopnoea',
                'pnd',
                'peripheral_oedema'
            ]
        }
        
        # Check which columns exist
        existing_columns = {
            category: [col for col in cols if col in df_index.columns]
            for category, cols in all_hf_columns.items()
        }
        
        # Check if at least one column exists
        total_existing = sum(len(cols) for cols in existing_columns.values())
        if total_existing == 0:
            print("No heart failure columns found in the dataframe")
            return df_index_selected
        
        print(f"Creating heart failure multiclass features with {total_existing} available columns...")
        
        df_index_selected['heart_failure_class'] = 0
        
        # Create masks
        mask_diagnosis = self.has_value_in_columns(df_index, existing_columns['diagnosis']) if existing_columns['diagnosis'] else None
        mask_symptoms = self.has_value_in_columns(df_index, existing_columns['symptoms']) if existing_columns['symptoms'] else None
        
        # Create NYHA masks
        nyha_masks = {}
        if existing_columns['nyha']:
            for nyha_class in ['iv', 'iii', 'ii']:  # Order from highest to lowest
                cols = [col for col in existing_columns['nyha'] if nyha_class in col.lower()]
                if cols:
                    nyha_masks[nyha_class] = self.has_value_in_columns(df_index, cols)
        
        # Hierarchical classification - starting from highest severity
        # Class 4: NYHA IV (highest severity, with or without any other conditions)
        if 'iv' in nyha_masks:
            df_index_selected.loc[nyha_masks['iv'], 'heart_failure_class'] = 4
        
        # Class 3: NYHA III (excluding those already classified as IV)
        if 'iii' in nyha_masks:
            df_index_selected.loc[nyha_masks['iii'] & (df_index_selected['heart_failure_class'] == 0), 'heart_failure_class'] = 3
        
        # Class 2: NYHA II (excluding those already classified as III or IV)
        if 'ii' in nyha_masks:
            df_index_selected.loc[nyha_masks['ii'] & (df_index_selected['heart_failure_class'] == 0), 'heart_failure_class'] = 2
        
        # Class 1: Any diagnosis (excluding those with any NYHA classification)
        if mask_diagnosis is not None:
            df_index_selected.loc[mask_diagnosis & (df_index_selected['heart_failure_class'] == 0), 'heart_failure_class'] = 1
        
        # Optional: Class 5 for symptoms only
        if mask_symptoms is not None:
            df_index_selected.loc[mask_symptoms & (df_index_selected['heart_failure_class'] == 0), 'heart_failure_class'] = 5

        # Define meanings
        meanings = {
            0: 'No heart failure',
            1: 'Has diagnosis only (no NYHA)',
            2: 'Has NYHA II',
            3: 'Has NYHA III',
            4: 'Has NYHA IV',
            5: 'Has symptoms only (no diagnosis or NYHA)'
        }
        
        # Print distributions and detailed information
        class_counts = df_index_selected['heart_failure_class'].value_counts().sort_index()
        print(f"\nHeart Failure Class distribution:")
        for cls, count in class_counts.items():
            meaning = meanings.get(cls, f'Class {cls}')
            print(f"  Class {cls} ({meaning}): {count} rows")
            
            # Print additional details for each class
            if cls == 4 and 'iv' in nyha_masks:
                with_diagnosis = (nyha_masks['iv'] & mask_diagnosis).sum() if mask_diagnosis is not None else 0
                print(f"    - NYHA IV with diagnosis: {with_diagnosis}")
                print(f"    - NYHA IV total: {nyha_masks['iv'].sum()}")
            elif cls == 3 and 'iii' in nyha_masks:
                with_diagnosis = (nyha_masks['iii'] & mask_diagnosis & (df_index_selected['heart_failure_class'] == 3)).sum() if mask_diagnosis is not None else 0
                print(f"    - NYHA III with diagnosis: {with_diagnosis}")
            elif cls == 2 and 'ii' in nyha_masks:
                with_diagnosis = (nyha_masks['ii'] & mask_diagnosis & (df_index_selected['heart_failure_class'] == 2)).sum() if mask_diagnosis is not None else 0
                print(f"    - NYHA II with diagnosis: {with_diagnosis}")

        return df_index_selected

    def create_hypertension_combined_features(self, df_index, df_index_selected):
        """
        Create multiclass features for hypertension with all possible combinations
        """
        hypertension_columns = [
            'hypertension_icd',
            'hypertension_pac',
            'hypertension_treated',
            'hypertension_untreated'
        ]
        
        # Check if columns exist
        columns_exist, existing_columns = self.check_columns_exist(df_index, hypertension_columns)
        
        if not columns_exist:
            print("No hypertension columns found in the dataframe")
            return df_index_selected
        
        print("Creating hypertension multiclass features...")
        df_index_selected['hypertension_class'] = 0
        
        # Create masks only for existing columns
        # Diagnosis masks
        icd_cols = [col for col in existing_columns if 'icd' in col.lower()]
        pac_cols = [col for col in existing_columns if 'pac' in col.lower() and 'treated' not in col.lower()]
        treatment_cols = [col for col in existing_columns if 'treated' in col.lower()]
        
        mask_icd = self.has_value_in_columns(df_index, icd_cols) if icd_cols else None
        mask_pac = self.has_value_in_columns(df_index, pac_cols) if pac_cols else None
        mask_treated = self.has_value_in_columns(df_index, ['hypertension_treated']) if 'hypertension_treated' in existing_columns else None
        mask_untreated = self.has_value_in_columns(df_index, ['hypertension_untreated']) if 'hypertension_untreated' in existing_columns else None
        
        # Create combined diagnosis mask if either ICD or PAC exists
        mask_diagnosis = None
        if mask_icd is not None and mask_pac is not None:
            mask_diagnosis = mask_icd | mask_pac
        elif mask_icd is not None:
            mask_diagnosis = mask_icd
        elif mask_pac is not None:
            mask_diagnosis = mask_pac
        
        # Order from most specific to least specific based on available columns
        if mask_diagnosis is not None:
            if mask_treated is not None:
                df_index_selected.loc[mask_diagnosis & mask_treated & ~(mask_untreated if mask_untreated is not None else False), 
                                    'hypertension_class'] = 1  # Diagnosed and treated
            
            if mask_untreated is not None:
                df_index_selected.loc[mask_diagnosis & mask_untreated & ~(mask_treated if mask_treated is not None else False), 
                                    'hypertension_class'] = 2  # Diagnosed and untreated
            
            # Individual diagnosis types (only if both exist)
            if mask_icd is not None and mask_pac is not None:
                df_index_selected.loc[mask_icd & ~mask_pac & 
                                    ~(mask_treated if mask_treated is not None else False) & 
                                    ~(mask_untreated if mask_untreated is not None else False), 
                                    'hypertension_class'] = 3  # ICD only
                
                df_index_selected.loc[mask_pac & ~mask_icd & 
                                    ~(mask_treated if mask_treated is not None else False) & 
                                    ~(mask_untreated if mask_untreated is not None else False), 
                                    'hypertension_class'] = 4  # PAC only
                
                df_index_selected.loc[mask_icd & mask_pac & 
                                    ~(mask_treated if mask_treated is not None else False) & 
                                    ~(mask_untreated if mask_untreated is not None else False), 
                                    'hypertension_class'] = 5  # Both ICD and PAC
        
        # Treatment status without diagnosis (only if treatment columns exist)
        if mask_treated is not None and mask_diagnosis is not None:
            df_index_selected.loc[mask_treated & ~mask_diagnosis & 
                                ~(mask_untreated if mask_untreated is not None else False), 
                                'hypertension_class'] = 6  # Just treated
        
        if mask_untreated is not None and mask_diagnosis is not None:
            df_index_selected.loc[mask_untreated & ~mask_diagnosis & 
                                ~(mask_treated if mask_treated is not None else False), 
                                'hypertension_class'] = 7  # Just untreated

        # Define meanings based on available columns
        meanings = {0: 'No hypertension'}
        
        if mask_diagnosis is not None:
            if mask_treated is not None:
                meanings[1] = 'Diagnosed and treated'
            if mask_untreated is not None:
                meanings[2] = 'Diagnosed and untreated'
            if mask_icd is not None and mask_pac is not None:
                meanings[3] = 'ICD diagnosis only'
                meanings[4] = 'PAC diagnosis only'
                meanings[5] = 'Both ICD and PAC diagnosis'
            if mask_treated is not None:
                meanings[6] = 'Treatment record only'
            if mask_untreated is not None:
                meanings[7] = 'Untreated record only'
        
        # Print distributions and available columns
        class_counts = df_index_selected['hypertension_class'].value_counts().sort_index()
        print(f"\nHypertension Class distribution:")
        print(f"Using {len(existing_columns)} available columns:")
        for col in existing_columns:
            print(f"  - {col}")
        
        print("\nClass distribution:")
        for cls, count in class_counts.items():
            if cls in meanings:
                print(f"  Class {cls} ({meanings[cls]}): {count} rows")

        return df_index_selected

    def create_ischaemic_heart_disease_combined_features(self, df_index, df_index_selected):
        """
        Create hierarchical multiclass features for heart disease
        Starting with highest CCS class and moving down to diagnosis
        """
        
        columns = [
            'ischaemic_heart_disease_icd',
            'ischaemic_heart_disease',
            'm_i',
            'angina_with_vigorous_activ',
            'angina_with_ordinary_activ',
            'angina_with_adls_ccs_3',
            'angina_at_rest_ccs_4',
            'bare_metal_stent',
            'drug_eluting_stent',
            'cabg'
        ]
        
        # Check if columns exist
        columns_exist, existing_columns = self.check_columns_exist(df_index, columns)
        
        if not columns_exist:
            return df_index_selected
        
        print(f"Found {len(existing_columns)} heart disease related columns:")
        for col in existing_columns:
            print(f"  - {col}")
        
        df_index_selected['ischaemic_heart_disease'] = 0
        
        # Create masks for different categories
        # Diagnosis masks
        diagnosis_columns = [col for col in existing_columns if any(x in col.lower() for x in ['icd', 'ischaemic', 'm_i'])]
        mask_diagnosis = self.has_value_in_columns(df_index, diagnosis_columns) if diagnosis_columns else None
        
        # CCS angina masks (ordered from highest to lowest)
        ccs_masks = {}
        ccs_mapping = {
            'ccs_4': 'angina_at_rest_ccs_4',
            'ccs_3': 'angina_with_adls_ccs_3',
            'ccs_2': 'angina_with_ordinary_activ',
            'ccs_1': 'angina_with_vigorous_activ'
        }
        
        for ccs_level, column in ccs_mapping.items():
            if column in existing_columns:
                ccs_masks[ccs_level] = self.has_value_in_columns(df_index, [column])
        
        # Intervention masks
        intervention_columns = [col for col in existing_columns if any(x in col.lower() for x in ['stent', 'cabg'])]
        mask_intervention = self.has_value_in_columns(df_index, intervention_columns) if intervention_columns else None
        
        # Hierarchical classification - starting from highest severity
        # Class 4: CCS 4 (highest severity, with or without other conditions)
        if 'ccs_4' in ccs_masks:
            df_index_selected.loc[ccs_masks['ccs_4'], 'ischaemic_heart_disease'] = 4
        
        # Class 3: CCS 3
        if 'ccs_3' in ccs_masks:
            df_index_selected.loc[ccs_masks['ccs_3'] & (df_index_selected['ischaemic_heart_disease'] == 0), 'ischaemic_heart_disease'] = 3
        
        # Class 2: CCS 2
        if 'ccs_2' in ccs_masks:
            df_index_selected.loc[ccs_masks['ccs_2'] & (df_index_selected['ischaemic_heart_disease'] == 0), 'ischaemic_heart_disease'] = 2
        
        # Class 1: CCS 1 or diagnosis or intervention
        conditions_for_class_1 = []
        if 'ccs_1' in ccs_masks:
            conditions_for_class_1.append(ccs_masks['ccs_1'])
        if mask_diagnosis is not None:
            conditions_for_class_1.append(mask_diagnosis)
        if mask_intervention is not None:
            conditions_for_class_1.append(mask_intervention)
        
        if conditions_for_class_1:
            combined_mask = pd.concat(conditions_for_class_1, axis=1).any(axis=1)
            df_index_selected.loc[combined_mask & (df_index_selected['ischaemic_heart_disease'] == 0), 'ischaemic_heart_disease'] = 1

        # Define meanings
        meanings = {
            0: 'No heart disease',
            1: 'Has diagnosis/intervention/CCS 1',
            2: 'Has CCS 2',
            3: 'Has CCS 3',
            4: 'Has CCS 4'
        }
        
        # Print distributions and detailed information
        class_counts = df_index_selected['ischaemic_heart_disease'].value_counts().sort_index()
        print(f"\nHeart Disease Class distribution:")
        for cls, count in class_counts.items():
            meaning = meanings.get(cls, f'Class {cls}')
            print(f"  Class {cls} ({meaning}): {count} rows")
            
            # Print additional details for each class
            if cls > 0:
                if mask_diagnosis is not None and cls > 1:
                    with_diagnosis = (df_index_selected['ischaemic_heart_disease'] == cls) & mask_diagnosis
                    print(f"    - With diagnosis: {with_diagnosis.sum()}")
                if mask_intervention is not None and cls > 1:
                    with_intervention = (df_index_selected['ischaemic_heart_disease'] == cls) & mask_intervention
                    print(f"    - With intervention: {with_intervention.sum()}")

        return df_index_selected

    def create_valvular_heart_disease_combined_features(self, df_index, df_index_selected):
        """
        Create multiclass features for valvular heart disease
        """
        # Define possible columns
        columns = [
            'valvular_heart_disease_icd',
            'valvular_heart_pac',
            'valvular_heart_disease_pac',
            'mitral_stenosis',
            'aortic_regurgitation',
            'mitral_regurgitation',
            'tricuspid_regurgitation'
        ]
        
        # Check if columns exist
        columns_exist, existing_columns = self.check_columns_exist(df_index, columns)
        
        if not columns_exist:
            print("No valvular heart disease columns found in the dataframe")
            return df_index_selected
        
        print("Creating valvular heart disease features...")
        df_index_selected['valvular_disease_class'] = 0
        
        # Create masks
        # Diagnosis masks
        diagnosis_columns = [col for col in existing_columns if any(x in col.lower() for x in ['icd', 'pac'])]
        mask_diagnosis = self.has_value_in_columns(df_index, diagnosis_columns) if diagnosis_columns else None
        
        # Specific valve conditions masks
        mask_mitral = self.has_value_in_columns(df_index, [col for col in existing_columns if 'mitral' in col.lower()])
        mask_aortic = self.has_value_in_columns(df_index, [col for col in existing_columns if 'aortic' in col.lower()])
        mask_tricuspid = self.has_value_in_columns(df_index, [col for col in existing_columns if 'tricuspid' in col.lower()])
        
        # Hierarchical classification - from most to least specific
        # Class 4: Multiple valve involvement
        multiple_valve_mask = (
            (mask_mitral & mask_aortic) |
            (mask_mitral & mask_tricuspid) |
            (mask_aortic & mask_tricuspid)
        )
        df_index_selected.loc[multiple_valve_mask, 'valvular_disease_class'] = 4
        
        # Class 3: Single valve with specific condition
        single_valve_mask = (
            mask_mitral | mask_aortic | mask_tricuspid
        ) & ~multiple_valve_mask  # Exclude those already classified
        df_index_selected.loc[single_valve_mask, 'valvular_disease_class'] = 3
        
        # Class 2: Diagnosis only (if not already classified with specific valve condition)
        if mask_diagnosis is not None:
            df_index_selected.loc[mask_diagnosis & (df_index_selected['valvular_disease_class'] == 0), 'valvular_disease_class'] = 2
        
        # Define meanings
        meanings = {
            0: 'No valvular heart disease',
            2: 'Diagnosed valvular disease without specific valve involvement',
            3: 'Single valve involvement',
            4: 'Multiple valve involvement'
        }
        
        # Print distributions and detailed information
        class_counts = df_index_selected['valvular_disease_class'].value_counts().sort_index()
        print(f"\nValvular Heart Disease Class distribution:")
        for cls, count in class_counts.items():
            meaning = meanings.get(cls, f'Class {cls}')
            print(f"  Class {cls} ({meaning}): {count} rows")
            
            # Print additional details for each class
            if cls == 4:
                mitral_aortic = (mask_mitral & mask_aortic & (df_index_selected['valvular_disease_class'] == 4)).sum()
                mitral_tricuspid = (mask_mitral & mask_tricuspid & (df_index_selected['valvular_disease_class'] == 4)).sum()
                aortic_tricuspid = (mask_aortic & mask_tricuspid & (df_index_selected['valvular_disease_class'] == 4)).sum()
                print(f"    - Mitral + Aortic: {mitral_aortic}")
                print(f"    - Mitral + Tricuspid: {mitral_tricuspid}")
                print(f"    - Aortic + Tricuspid: {aortic_tricuspid}")
            elif cls == 3:
                mitral_only = (mask_mitral & (df_index_selected['valvular_disease_class'] == 3)).sum()
                aortic_only = (mask_aortic & (df_index_selected['valvular_disease_class'] == 3)).sum()
                tricuspid_only = (mask_tricuspid & (df_index_selected['valvular_disease_class'] == 3)).sum()
                print(f"    - Mitral valve only: {mitral_only}")
                print(f"    - Aortic valve only: {aortic_only}")
                print(f"    - Tricuspid valve only: {tricuspid_only}")
            elif cls == 2 and mask_diagnosis is not None:
                print(f"    - Diagnosis without specific valve: {(mask_diagnosis & (df_index_selected['valvular_disease_class'] == 2)).sum()}")

        return df_index_selected        
            
    def create_diabetes_combined_features(self, df_index, df_index_selected):
        """
        Create multiclass features for diabetes with all possible combinations
        """
        # Diabetes columns grouped by type
        diabetes_columns = {
            'type1': [
                'type1_with_icd',
                'type1_without_icd',
                'type1_pac'
            ],
            'type2': [
                'type2_with_icd',
                'type2_without_icd',
                'type2_pac',
                'type2_insulin_icd'
            ],
            'treatment': [
                'insulin_pac',
                'diet_controlled',
                'hypoglycaemics',
                'incretin_mimetics'
            ],
            'complication': [
                'end_organ_damage'
            ]
        }
        
        print("Creating diabetes multiclass features...")
        df_index_selected['diabetes_class'] = 0
        
        # Create masks for main types
        mask_type1 = self.has_value_in_columns(df_index, diabetes_columns['type1'])
        mask_type2 = self.has_value_in_columns(df_index, diabetes_columns['type2'])
        mask_treatment = self.has_value_in_columns(df_index, diabetes_columns['treatment'])
        mask_complication = self.has_value_in_columns(df_index, diabetes_columns['complication'])
        
        # Order matters! Assign from most specific to least specific combinations
        # First, combinations with all conditions
        df_index_selected.loc[mask_type1 & mask_type2 & mask_complication & mask_treatment, 'diabetes_class'] = 8  # All conditions
        
        # Then, triple combinations
        df_index_selected.loc[mask_type1 & mask_type2 & mask_complication & ~mask_treatment, 'diabetes_class'] = 7  # Both types + complications
        df_index_selected.loc[mask_type1 & mask_type2 & mask_treatment & ~mask_complication, 'diabetes_class'] = 6  # Both types + treatment
        
        # Then, double combinations
        df_index_selected.loc[mask_type1 & mask_complication & ~mask_type2 & ~mask_treatment, 'diabetes_class'] = 5  # Type 1 + complications
        df_index_selected.loc[mask_type2 & mask_complication & ~mask_type1 & ~mask_treatment, 'diabetes_class'] = 4  # Type 2 + complications
        df_index_selected.loc[mask_type1 & mask_type2 & ~mask_complication & ~mask_treatment, 'diabetes_class'] = 3  # Both types
        
        # Finally, single conditions
        df_index_selected.loc[mask_type1 & ~mask_type2 & ~mask_complication & ~mask_treatment, 'diabetes_class'] = 1  # Just Type 1
        df_index_selected.loc[mask_type2 & ~mask_type1 & ~mask_complication & ~mask_treatment, 'diabetes_class'] = 2  # Just Type 2

        # Print distributions and class meanings
        diabetes_meanings = {
            0: 'No diabetes',
            1: 'Type 1 only',
            2: 'Type 2 only',
            3: 'Both Type 1 and 2',
            4: 'Type 2 with complications',
            5: 'Type 1 with complications',
            6: 'Both types with treatment',
            7: 'Both types with complications',
            8: 'All conditions (both types + complications + treatment)'
        }
        
        # Print distributions
        class_counts = df_index_selected['diabetes_class'].value_counts().sort_index()
        print(f"\nDiabetes Class distribution:")
        for cls, count in class_counts.items():
            print(f"  Class {cls} ({diabetes_meanings[cls]}): {count} rows")

        return df_index_selected
    
    def create_cancer_combined_features(self, df_index, df_index_selected):
        """
        Create multiclass features for cancer with all possible combinations
        """
        # Cancer Feature
        cancer_columns = [
            'cancer_unspecified_pac',
            'cancer_primary_icd', 'cancer_localised_pac',
            'cancer_secondary_icd', 'cancer_metastatic_pac'
        ]
        print("Creating cancer multiclass features...")
        df_index_selected['cancer_class'] = 0
        
        # Create masks
        mask_unspecified = self.has_value_in_columns(df_index, [col for col in cancer_columns if 'unspecified' in col.lower()])
        mask_primary = self.has_value_in_columns(df_index, [col for col in cancer_columns if any(x in col.lower() for x in ['primary', 'localised'])])
        mask_secondary = self.has_value_in_columns(df_index, [col for col in cancer_columns if any(x in col.lower() for x in ['secondary', 'metastatic'])])
        
        # Order matters! Assign from most specific to least specific combinations
        # First, all three conditions present
        df_index_selected.loc[mask_unspecified & mask_primary & mask_secondary, 'cancer_class'] = 7  # All three types
        
        # Then, double combinations
        df_index_selected.loc[mask_unspecified & mask_primary & ~mask_secondary, 'cancer_class'] = 5  # Unspecified + Primary
        df_index_selected.loc[mask_unspecified & mask_secondary & ~mask_primary, 'cancer_class'] = 6  # Unspecified + Secondary
        df_index_selected.loc[mask_primary & mask_secondary & ~mask_unspecified, 'cancer_class'] = 4  # Primary + Secondary
        
        # Finally, single conditions
        df_index_selected.loc[mask_unspecified & ~mask_primary & ~mask_secondary, 'cancer_class'] = 1  # Just unspecified
        df_index_selected.loc[mask_primary & ~mask_unspecified & ~mask_secondary, 'cancer_class'] = 2  # Just primary/localised
        df_index_selected.loc[mask_secondary & ~mask_unspecified & ~mask_primary, 'cancer_class'] = 3  # Just secondary/metastatic

        # Print distributions and class meanings
        meanings = {
            0: 'No cancer',
            1: 'Unspecified cancer only',
            2: 'Primary/Localised only',
            3: 'Secondary/Metastatic only',
            4: 'Primary + Secondary (no unspecified)',
            5: 'Unspecified + Primary (no secondary)',
            6: 'Unspecified + Secondary (no primary)',
            7: 'All types present'
        }
        
        class_counts = df_index_selected['cancer_class'].value_counts().sort_index()
        print(f"\nCancer Class distribution:")
        for cls, count in class_counts.items():
            print(f"  Class {cls} ({meanings[cls]}): {count} rows")

        return df_index_selected

    def create_neuro_combined_features(self, df_index, df_index_selected):
        """
        Create separate multiclass features with all possible combinations
        """
        # 1. Cerebrovascular Feature
        cerebrovascular_columns = [
            'cerebrovascular_icd', 'cerebrovascular_pac',
            'stroke', 'tia', 'hemiplegia_icd', 'hemiplegia_pac'
        ]
        print("Creating cerebrovascular disease multiclass features...")
        df_index_selected['cerebrovascular_class'] = 0
        
        # Create masks
        mask_cerebro = self.has_value_in_columns(df_index, cerebrovascular_columns)
        mask_stroke = self.has_value_in_columns(df_index, [col for col in cerebrovascular_columns if 'stroke' in col.lower()])
        mask_tia = self.has_value_in_columns(df_index, [col for col in cerebrovascular_columns if 'tia' in col.lower()])
        
        # Order from most specific to least specific
        df_index_selected.loc[mask_cerebro & mask_stroke & mask_tia, 'cerebrovascular_class'] = 7  # All three
        df_index_selected.loc[mask_cerebro & mask_stroke & ~mask_tia, 'cerebrovascular_class'] = 5  # Unspecified + Stroke
        df_index_selected.loc[mask_cerebro & mask_tia & ~mask_stroke, 'cerebrovascular_class'] = 6  # Unspecified + TIA
        df_index_selected.loc[mask_stroke & mask_tia & ~mask_cerebro, 'cerebrovascular_class'] = 4  # Stroke + TIA
        df_index_selected.loc[mask_cerebro & ~mask_stroke & ~mask_tia, 'cerebrovascular_class'] = 1  # Just unspecified
        df_index_selected.loc[mask_stroke & ~mask_cerebro & ~mask_tia, 'cerebrovascular_class'] = 2  # Just stroke
        df_index_selected.loc[mask_tia & ~mask_cerebro & ~mask_stroke, 'cerebrovascular_class'] = 3  # Just TIA

        # 2. Neurological Feature
        neurological_columns = [
            'neurological_condition_pac', 'epilepsy',
            'epilepsy_icd', 'neuromuscular_disorder'
        ]
        print("Creating neurological conditions multiclass features...")
        df_index_selected['neurological_class'] = 0
        
        # Create masks
        mask_neuro = self.has_value_in_columns(df_index, neurological_columns)
        mask_epilepsy = self.has_value_in_columns(df_index, [col for col in neurological_columns if 'epilepsy' in col.lower()])
        mask_neuromuscular = self.has_value_in_columns(df_index, [col for col in neurological_columns if 'neuromuscular' in col.lower()])
        
        # Order from most specific to least specific
        df_index_selected.loc[mask_neuro & mask_epilepsy & mask_neuromuscular, 'neurological_class'] = 7  # All three
        df_index_selected.loc[mask_neuro & mask_epilepsy & ~mask_neuromuscular, 'neurological_class'] = 5  # Unspecified + Epilepsy
        df_index_selected.loc[mask_neuro & mask_neuromuscular & ~mask_epilepsy, 'neurological_class'] = 6  # Unspecified + Neuromuscular
        df_index_selected.loc[mask_epilepsy & mask_neuromuscular & ~mask_neuro, 'neurological_class'] = 4  # Epilepsy + Neuromuscular
        df_index_selected.loc[mask_neuro & ~mask_epilepsy & ~mask_neuromuscular, 'neurological_class'] = 1  # Just unspecified
        df_index_selected.loc[mask_epilepsy & ~mask_neuro & ~mask_neuromuscular, 'neurological_class'] = 2  # Just epilepsy
        df_index_selected.loc[mask_neuromuscular & ~mask_neuro & ~mask_epilepsy, 'neurological_class'] = 3  # Just neuromuscular

        # 3. Brain-related Feature
        brain_columns = [
            'degenerative_icd', 'congenital_icd',
            'organic_brain_icd', 'dementia_icd', 'dementia_pac'
        ]
        print("Creating brain-related conditions multiclass features...")
        df_index_selected['brain_condition_class'] = 0
        
        # Create masks
        mask_brain = self.has_value_in_columns(df_index, brain_columns)
        mask_degenerative = self.has_value_in_columns(df_index, [col for col in brain_columns if 'degenerative' in col.lower()])
        mask_dementia = self.has_value_in_columns(df_index, [col for col in brain_columns if 'dementia' in col.lower()])
        
        # Order from most specific to least specific
        df_index_selected.loc[mask_brain & mask_degenerative & mask_dementia, 'brain_condition_class'] = 7  # All three
        df_index_selected.loc[mask_brain & mask_degenerative & ~mask_dementia, 'brain_condition_class'] = 5  # Unspecified + Degenerative
        df_index_selected.loc[mask_brain & mask_dementia & ~mask_degenerative, 'brain_condition_class'] = 6  # Unspecified + Dementia
        df_index_selected.loc[mask_degenerative & mask_dementia & ~mask_brain, 'brain_condition_class'] = 4  # Degenerative + Dementia
        df_index_selected.loc[mask_brain & ~mask_degenerative & ~mask_dementia, 'brain_condition_class'] = 1  # Just unspecified
        df_index_selected.loc[mask_degenerative & ~mask_brain & ~mask_dementia, 'brain_condition_class'] = 2  # Just degenerative
        df_index_selected.loc[mask_dementia & ~mask_brain & ~mask_degenerative, 'brain_condition_class'] = 3  # Just dementia

        # Print distributions and class meanings for each feature
        for feature, meanings in {
            'cerebrovascular_class': {
                0: 'No cerebrovascular condition',
                1: 'Unspecified cerebrovascular only',
                2: 'Stroke only',
                3: 'TIA only',
                4: 'Stroke + TIA (no unspecified)',
                5: 'Unspecified + Stroke (no TIA)',
                6: 'Unspecified + TIA (no stroke)',
                7: 'All conditions present'
            },
            'neurological_class': {
                0: 'No neurological condition',
                1: 'Unspecified neurological only',
                2: 'Epilepsy only',
                3: 'Neuromuscular only',
                4: 'Epilepsy + Neuromuscular (no unspecified)',
                5: 'Unspecified + Epilepsy (no neuromuscular)',
                6: 'Unspecified + Neuromuscular (no epilepsy)',
                7: 'All conditions present'
            },
            'brain_condition_class': {
                0: 'No brain condition',
                1: 'Unspecified brain condition only',
                2: 'Degenerative only',
                3: 'Dementia only',
                4: 'Degenerative + Dementia (no unspecified)',
                5: 'Unspecified + Degenerative (no dementia)',
                6: 'Unspecified + Dementia (no degenerative)',
                7: 'All conditions present'
            }
        }.items():
            class_counts = df_index_selected[feature].value_counts().sort_index()
            print(f"\n{feature.replace('_', ' ').title()} distribution:")
            for cls, count in class_counts.items():
                print(f"  Class {cls} ({meanings[cls]}): {count} rows")

        return df_index_selected

    def load_icd_features(self):
        """
        Load ICD features from a processed CSV file.
        
        Returns:
        --------
        pandas.DataFrame or None
            DataFrame containing ICD features (if the file exists), otherwise None
        """
        print(self.config.icd_file)
        if not self.config.icd_file:
            print("ICD features file not provided")
            return None
            
        if not os.path.exists(self.config.icd_file):
            print(f"Warning: ICD features file not found: {self.config.icd_file}")
            return None
            
        print(f"Loading ICD features from {self.config.icd_file}...")
        icd_features = pd.read_csv(self.config.icd_file)
        print(f"ICD features loaded, shape: {icd_features.shape}")
        
        # Show statistics for ICD features
        icd_columns = [col for col in icd_features.columns if col != 'record_id']
        
        if len(icd_columns) > 0:
            # Count non-zero values for each feature
            non_zero_counts = (icd_features[icd_columns] != 0).sum()
            top_features = non_zero_counts.sort_values(ascending=False).head(10)
            
            print("\nTop 10 most common ICD features:")
            for feature, count in top_features.items():
                print(f"  {feature}: {count} non-zero values ({count/len(icd_features)*100:.2f}%)")
        
        return icd_features
    
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

    def normalize_numerical_features(self, df):
        """
        Standardize numerical features using StandardScaler.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame containing features to be standardized
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame containing standardized features
        """
        print("Standardizing numerical features...")
        
        # Create StandardScaler
        scaler = StandardScaler()
        
        # Identify numerical columns (excluding record_id and target columns)
        numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
        numerical_cols = [col for col in numerical_cols 
                         if col != 'record_id' 
                         and col not in self.config.target_columns]
        
        if numerical_cols:
            print(f"Standardizing {len(numerical_cols)} numerical features")
            
            # Create a copy for numerical features
            df_numeric = df[numerical_cols].copy()
            
            # Show statistics before standardization
            print("\nStatistics before standardization:")
            for feature in numerical_cols:
                mean_val = df_numeric[feature].mean()
                std_val = df_numeric[feature].std()
                min_val = df_numeric[feature].min()
                max_val = df_numeric[feature].max()
                print(f"  {feature}: mean={mean_val:.2f}, std={std_val:.2f}, range=[{min_val:.2f}, {max_val:.2f}]")
            
            # Handle NaN values with median
            df_numeric = df_numeric.fillna(df_numeric.median())
            
            # Fit and transform
            df[numerical_cols] = scaler.fit_transform(df_numeric)
            
            # Show statistics after standardization
            print("\nStatistics after standardization:")
            for feature in numerical_cols:
                mean_val = df[feature].mean()
                std_val = df[feature].std()
                min_val = df[feature].min()
                max_val = df[feature].max()
                print(f"  {feature}: mean={mean_val:.2f}, std={std_val:.2f}, range=[{min_val:.2f}, {max_val:.2f}]")
        else:
            print("No numerical features found to standardize")
        
        return df

    def preprocess_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess features including encoding categorical variables"""
        logger.info("Preprocessing features...")
        
        # 1. Apply column transformations first
        df = self.apply_column_transformations(df)
        if 'record_id' not in df.columns:
            raise ValueError("record_id column missing after column transformations")
        logger.info("Applied column transformations")
        print(f"After column transformations: {df.shape}")
        
        # 2. Impute missing data
        df = self.impute_or_drop_missing_data(df)
        if 'record_id' not in df.columns:
            raise ValueError("record_id column missing after imputation")
        logger.info("Imputed missing data")
        
        print(f"impute_or_drop_missing_data: {df.shape}")


        # 3. Remove outliers
        df = self.remove_outliers(df)
        if 'record_id' not in df.columns:
            raise ValueError("record_id column missing after removing outliers")
        logger.info("Removed outliers")
        print(f"remove_outliers: {df.shape}")
        
        # 4. Filter zero variance columns after transformations and imputation
        df = self.filter_zero_variance_columns(df)
        # df = self.remove_low_variance_features(df)
        print(f"filter_zero_variance_columns: {df.shape}")

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
        print(f"After feature selection: {df.shape}")
        df = self.convert_categorical_to_numeric(df)
        print(f"After convert_categorical_to_numeric: {df.shape}")
        return df
    
    def process(self) -> pd.DataFrame:
        """Main processing pipeline"""
        try:
            # Load and process data
            df_index = self.load_data()
            print(f"Load data shape: {df_index.shape}")
                  
            df_index_selected = self.preprocess_features(df_index)
            print(f"preprocess data shape: {df_index_selected.shape}")
            
            if self.config.deep_learning:
                df_index_selected = self.create_heart_failure_combined_features(df_index, df_index_selected)
                
                df_index_selected = self.create_hypertension_combined_features(df_index, df_index_selected)
                
                df_index_selected = self.create_ischaemic_heart_disease_combined_features(df_index, df_index_selected)
                
                df_index_selected = self.create_valvular_heart_disease_combined_features(df_index, df_index_selected)
                
                df_index_selected = self.create_diabetes_combined_features(df_index, df_index_selected)
                
                df_index_selected = self.create_cancer_combined_features(df_index, df_index_selected)
                
                df_index_selected = self.create_neuro_combined_features(df_index, df_index_selected)
                
                # Load and merge ICD features
                icd_features = self.load_icd_features()
                print(icd_features.columns)
                if icd_features is not None:
                    print(f"Merging ICD features (shape: {icd_features.shape})...")
                    
                    icd_features.drop(columns=[col for col in self.config.exclude_icds if col in icd_features.columns], inplace=True)
                    df = pd.merge(df_index_selected, icd_features, on='record_id', how='left')
                    logger.info(f"Dataset shape after integrating target features: {df.shape}")
                    
                print(df_index_selected.columns)
                
                # Get all columns except record_id
                feature_columns = [col for col in icd_features.columns if col != 'record_id']
                
                # Fill NaN values produced by the merge with 0 (for one-hot encoded features)
                null_before = df[feature_columns].isnull().sum().sum()
                
            else:
                icdpreprocessor = IcdsPreprocessing()
            
                df_icd_transformed = icdpreprocessor.preprocess_icd_data(df_index)
                disease_present_columns = [col for col in df_icd_transformed.columns if col.endswith('_disease_present')] + ['record_id']
                
                df = pd.merge(df_index_selected, df_icd_transformed[disease_present_columns], on='record_id', how='left')
            
            # # Apply standardization before merging with target features
            # df = self.normalize_numerical_features(df_index_selected)
            
            
            # 4. Filter zero variance columns after transformations and imputation
            df = self.filter_zero_variance_columns(df)
            # df = self.remove_low_variance_features(df)
            print(f"filter_zero_variance_columns: {df.shape}")
        
        
            # Load and integrate target features if available
            target_features = self.load_target_feature()
            if target_features is not None:
                logger.info("Integrating target features with main dataset...")
                df = pd.merge(df, target_features, on='record_id', how='left')
                logger.info(f"Dataset shape after integrating target features: {df.shape}")
            
            # Get all columns except record_id
            feature_columns = [col for col in target_features.columns if col != 'record_id']
            
            # Fill NaN values produced by the merge with 0 (for one-hot encoded features)
            null_before = df[feature_columns].isnull().sum().sum()
            
            if null_before > 0:
                print(f"Filled {null_before} missing values after merging (not found in ICD/Target features)")
                df[feature_columns] = df[feature_columns].fillna(0)
                
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
    parser.add_argument('--icd_file', type=str, help='Path to ICD features file')
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
        output_file=None,
        target_file=args.target_file,
        target_columns=None,
        # icd_file=args.icd_file,
        encode_categorical=args.encode_categorical,
        drop_first=args.drop_first,
        max_unique_values_for_categorical=args.max_unique_values,
        min_unique_values_for_categorical=args.min_unique_values,
        exclude_icds=['Y9224']
    )
    
    # Process data
    processor = DataProcessor(config)
    processor.process()

if __name__ == "__main__":
    main()
