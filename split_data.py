"""
Data splitting module for creating train, validation, and test sets.
Splits data for multiple targets and organizes outputs in separate directories.
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import argparse
import logging
from pathlib import Path
import json
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_balanced_groups(data, target_columns):
    """
    Create balanced groups for stratification using a hash-based approach.
    
    Args:
        data: DataFrame containing the data
        target_columns: List of target column names
        
    Returns:
        Series containing group labels for stratification
    """
    # Create a unique identifier for each combination of target values
    group_labels = pd.Series(index=data.index, dtype='int')
    
    for col in target_columns:
        # Convert each target to string and combine
        group_labels = group_labels.astype(str) + '_' + data[col].astype(str)
    
    # Convert to hash values for efficiency
    return pd.factorize(group_labels)[0]

def split_data(
    data_path: str,
    target_columns: list,
    output_dir: str = 'split_data',
    test_size: float = 0.2,
    val_size: float = 0.2,
    random_state: int = 42,
    stratify: bool = True
) -> None:
    """
    Split data into train, validation, and test sets with balanced distribution.
    Creates separate directories for each target variable, with independent splits for each target.
    For each target, other target columns are excluded from the feature set.
    
    Args:
        data_path: Path to the input data file
        target_columns: List of target column names
        output_dir: Directory to save the split datasets
        test_size: Proportion of data to use for testing
        val_size: Proportion of training data to use for validation
        random_state: Random seed for reproducibility
        stratify: Whether to stratify the split based on target variables
    """
    print(target_columns)
    print(data_path)
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load data
    logger.info(f"Loading data from {data_path}")
    data = pd.read_csv(data_path)
    
    # Remove record_id if it exists
    if 'record_id' in data.columns:
        logger.info("Removing record_id column")
        data = data.drop('record_id', axis=1)
    target_columns = [col for col in target_columns if col in data.columns]
    
    # Get all feature columns (columns that are not targets)
    feature_columns = [col for col in data.columns if col not in target_columns]
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Process each target independently
    for target in target_columns:
        logger.info(f"\nProcessing target: {target}")
        
        # Create target-specific directory
        target_dir = output_path / target
        target_dir.mkdir(parents=True, exist_ok=True)
        
        # Get feature columns excluding all target columns
        feature_columns = [col for col in data.columns if col not in target_columns]
        
        # Check class distribution for current target
        value_counts = data[target].value_counts()
        can_stratify = True
        
        if stratify:
            logger.info(f"Checking class distribution for {target}...")
            
            # Check if any class has less than 2 samples
            if value_counts.min() < 2:
                logger.warning(f"Target '{target}' has classes with insufficient samples:")
                for val, count in value_counts.items():
                    logger.warning(f"  Class {val}: {count} samples")
                can_stratify = False
                logger.warning("Disabling stratification for this target due to insufficient samples.")
        
        # Log original distribution
        logger.info("\nOriginal class distribution:")
        for val, count in value_counts.items():
            logger.info(f"  Class {val}: {count} samples ({count/len(data):.1%})")
    
        # Prepare data for this target
        X = data[feature_columns]
        y = data[target]
        
        
            
        try:
            # First split: separate test set
            logger.info(f"\nSplitting data for {target} into train+val and test sets")
            X_train_val, X_test, y_train_val, y_test = train_test_split(
                X, y,
                test_size=test_size,
                random_state=random_state,
                stratify=y if can_stratify and stratify else None
            )
            
            # Second split: separate validation set from training set
            logger.info(f"Splitting train+val for {target} into train and validation sets")
            val_size_adjusted = val_size / (1 - test_size)
            X_train, X_val, y_train, y_val = train_test_split(
                X_train_val, y_train_val,
                test_size=val_size_adjusted,
                random_state=random_state,
                stratify=y_train_val if can_stratify and stratify else None
            )
            
            # Verify class distributions
            logger.info("\nVerifying class distributions across splits:")
            
            # Calculate distributions
            original_dist = y.value_counts(normalize=True)
            train_dist = y_train.value_counts(normalize=True)
            val_dist = y_val.value_counts(normalize=True)
            test_dist = y_test.value_counts(normalize=True)
            
            # Compare distributions
            logger.info("\nClass distribution comparison:")
            logger.info("Class | Original | Train   | Val     | Test    | Max Diff")
            logger.info("-" * 60)
            
            max_diff = 0
            for cls in original_dist.index:
                orig_pct = original_dist[cls] * 100
                train_pct = train_dist[cls] * 100
                val_pct = val_dist[cls] * 100
                test_pct = test_dist[cls] * 100
                
                # Calculate maximum difference from original
                diff = max(abs(train_pct - orig_pct), 
                          abs(val_pct - orig_pct), 
                          abs(test_pct - orig_pct))
                max_diff = max(max_diff, diff)
                
                logger.info(f"{cls:5d} | {orig_pct:7.1f}% | {train_pct:7.1f}% | {val_pct:7.1f}% | {test_pct:7.1f}% | {diff:7.1f}%")
            
            # Check if stratification was successful
            if stratify and can_stratify:
                if max_diff > 5:  # More than 5% difference
                    logger.warning(f"\nWARNING: Large class distribution differences detected (max diff: {max_diff:.1f}%)")
                    logger.warning("This might indicate stratification issues.")
                else:
                    logger.info(f"\nStratification successful. Maximum distribution difference: {max_diff:.1f}%")
            
        except ValueError as e:
            logger.warning(f"\nEncountered an error during stratified split for {target}: {str(e)}")
            logger.warning("Falling back to random splitting...")
            
            X_train_val, X_test, y_train_val, y_test = train_test_split(
                X, y,
                test_size=test_size,
                random_state=random_state
            )
            
            val_size_adjusted = val_size / (1 - test_size)
            X_train, X_val, y_train, y_val = train_test_split(
                X_train_val, y_train_val,
                test_size=val_size_adjusted,
                random_state=random_state
            )

        # Reconstruct full DataFrames for each split
        train_data = pd.concat([X_train, y_train], axis=1)
        val_data = pd.concat([X_val, y_val], axis=1)
        test_data = pd.concat([X_test, y_test], axis=1)
        
        # Save datasets
        train_data.to_csv(target_dir / 'train.csv', index=False)
        val_data.to_csv(target_dir / 'val.csv', index=False)
        test_data.to_csv(target_dir / 'test.csv', index=False)
        
        # Calculate and save class distributions
        train_dist = y_train.value_counts(normalize=True).to_dict()
        val_dist = y_val.value_counts(normalize=True).to_dict()
        test_dist = y_test.value_counts(normalize=True).to_dict()
        
        # Save split information
        split_info = {
            'timestamp': timestamp,
            'target_column': target,
            'feature_columns': feature_columns,
            'train_size': len(train_data),
            'val_size': len(val_data),
            'test_size': len(test_data),
            'total_size': len(data),
            'train_ratio': len(train_data) / len(data),
            'val_ratio': len(val_data) / len(data),
            'test_ratio': len(test_data) / len(data),
            'random_state': random_state,
            'stratified': stratify and can_stratify,
            'class_distribution': {
                'original': y.value_counts(normalize=True).to_dict(),
                'train': train_dist,
                'val': val_dist,
                'test': test_dist
            }
        }
        
        with open(target_dir / 'split_info.json', 'w') as f:
            json.dump(split_info, f, indent=4)
        
        # Log information for this target
        logger.info(f"\nFiles saved for target: {target}")
        logger.info(f"Directory: {target_dir}")
        logger.info(f"Training set: {len(train_data)} samples")
        logger.info(f"Validation set: {len(val_data)} samples")
        logger.info(f"Test set: {len(test_data)} samples")

def main():
    parser = argparse.ArgumentParser(description='Split data into train, validation, and test sets')
    parser.add_argument('data_path', type=str, help='Path to the input data file')
    parser.add_argument('--target_columns', type=str, nargs='+', required=True,
                      help='Names of target columns')
    parser.add_argument('--output_dir', type=str, default='split_data',
                      help='Directory to save the split datasets')
    parser.add_argument('--test_size', type=float, default=0.2,
                      help='Proportion of data to use for testing')
    parser.add_argument('--val_size', type=float, default=0.2,
                      help='Proportion of training data to use for validation')
    parser.add_argument('--random_state', type=int, default=42,
                      help='Random seed for reproducibility')
    parser.add_argument('--no_stratify', action='store_true',
                      help='Disable stratification')
    
    args = parser.parse_args()
    
    split_data(
        data_path=args.data_path,
        target_columns=args.target_columns,
        output_dir=args.output_dir,
        test_size=args.test_size,
        val_size=args.val_size,
        random_state=args.random_state,
        stratify=not args.no_stratify
    )

if __name__ == "__main__":
    main()