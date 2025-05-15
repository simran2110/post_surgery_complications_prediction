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
    Creates separate directories for each target variable.
    
    Args:
        data_path: Path to the input data file
        target_columns: List of target column names
        output_dir: Directory to save the split datasets
        test_size: Proportion of data to use for testing
        val_size: Proportion of training data to use for validation
        random_state: Random seed for reproducibility
        stratify: Whether to stratify the split based on target variables
    """
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load data
    logger.info(f"Loading data from {data_path}")
    data = pd.read_csv(data_path)
    
    # Separate features and targets
    X = data.drop(columns=target_columns)
    y = data[target_columns]
    
    # Check class distributions and decide on stratification
    if stratify:
        logger.info("Checking class distributions for stratification...")
        class_distributions = {}
        can_stratify = True
        
        for target in target_columns:
            value_counts = data[target].value_counts()
            class_distributions[target] = value_counts
            
            # Check if any class has less than 2 samples
            if value_counts.min() < 2:
                logger.warning(f"\nTarget '{target}' has classes with insufficient samples:")
                for val, count in value_counts.items():
                    logger.warning(f"  Class {val}: {count} samples")
                can_stratify = False
        
        if not can_stratify:
            logger.warning("\nDisabling stratification due to insufficient samples in some classes.")
            logger.warning("Proceeding with random splitting to ensure all classes are represented.")
            stratify = False
        else:
            logger.info("All classes have sufficient samples for stratification.")
            for target, dist in class_distributions.items():
                logger.info(f"\nClass distribution for {target}:")
                for val, count in dist.items():
                    logger.info(f"  Class {val}: {count} samples ({count/len(data):.1%})")
    
    # Perform the splits
    try:
        # First split: separate test set
        logger.info("\nSplitting data into train+val and test sets")
        if stratify:
            strat_groups = create_balanced_groups(data, target_columns)
        else:
            strat_groups = None
            
        train_val_data, test_data = train_test_split(
            data,  # Keep all columns together
            test_size=test_size,
            random_state=random_state,
            stratify=strat_groups
        )
        
        # Second split: separate validation set from training set
        logger.info("Splitting train+val into train and validation sets")
        val_size_adjusted = val_size / (1 - test_size)
        
        if stratify:
            strat_groups_train = create_balanced_groups(train_val_data, target_columns)
        else:
            strat_groups_train = None
        
        train_data, val_data = train_test_split(
            train_val_data,  # Keep all columns together
            test_size=val_size_adjusted,
            random_state=random_state,
            stratify=strat_groups_train
        )
        
    except ValueError as e:
        logger.warning(f"\nEncountered an error during stratified split: {str(e)}")
        logger.warning("Falling back to random splitting...")
        
        train_val_data, test_data = train_test_split(
            data,
            test_size=test_size,
            random_state=random_state
        )
        
        val_size_adjusted = val_size / (1 - test_size)
        train_data, val_data = train_test_split(
            train_val_data,
            test_size=val_size_adjusted,
            random_state=random_state
        )

    # Save datasets for each target separately
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    for target in target_columns:
        # Create target-specific directory
        target_dir = output_path / target
        target_dir.mkdir(parents=True, exist_ok=True)
        
        # Save complete datasets (features + target)
        train_data.to_csv(target_dir / 'train.csv', index=False)
        val_data.to_csv(target_dir / 'val.csv', index=False)
        test_data.to_csv(target_dir / 'test.csv', index=False)
        
        # Also save feature and target lists for reference
        feature_columns = [col for col in data.columns if col not in target_columns]
        
        # Calculate and save class distributions
        train_dist = train_data[target].value_counts(normalize=True).to_dict()
        val_dist = val_data[target].value_counts(normalize=True).to_dict()
        test_dist = test_data[target].value_counts(normalize=True).to_dict()
        
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
            'stratified': stratify,
            'class_distribution': {
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
        
        if not data[target].dtype.kind in 'ifc' or len(data[target].unique()) <= 10:
            logger.info("\nClass distribution:")
            logger.info("Training set:")
            for cls, prop in train_dist.items():
                logger.info(f"  Class {cls}: {prop:.1%}")

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