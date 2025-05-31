"""
Enhanced feature selection module for MIDAS risk prediction.
Provides Optuna-based feature selection with multiple model options.
"""
import os
import logging
import warnings
from datetime import datetime
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path
import json

# Third-party imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import optuna
from optuna.samplers import TPESampler, GridSampler
from tqdm import tqdm
import shap
import argparse

# Scikit-learn imports
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings('ignore')

@dataclass
class EnhancedFeatureSelectionConfig:
    """Configuration for enhanced feature selection"""
    method: str = 'optuna'  # Default to optuna
    task_type: str = 'classification'  # 'classification' or 'regression'
    target_column: str = None
    n_features: int = 10  # Number of features to select
    random_state: int = 42
    cv_folds: int = 3
    n_trials: int = 50  # Number of Optuna trials
    early_stopping_rounds: int = 10  # Early stopping rounds for Optuna
    model_type: str = 'logistic_regression'  # 'logistic_regression', 'lasso', or 'pca'
    metric: str = 'roc_auc'
    
class PCAFeatureSelector:
    """Helper class to wrap PCA for feature selection"""
    
    def __init__(self, n_components=None):
        self.n_components = n_components
        self.pca = PCA(n_components=n_components)
        self.lr = LogisticRegression(random_state=42, class_weight='balanced')
        
    def fit(self, X, y):
        """Fit PCA and logistic regression"""
        X_pca = self.pca.fit_transform(X)
        self.lr.fit(X_pca, y)
        return self
        
    def predict(self, X):
        """Predict using PCA and logistic regression"""
        X_pca = self.pca.transform(X)
        return self.lr.predict(X_pca)
        
    def predict_proba(self, X):
        """Predict probabilities using PCA and logistic regression"""
        X_pca = self.pca.transform(X)
        return self.lr.predict_proba(X_pca)
        
    def score(self, X, y):
        """Score using PCA and logistic regression"""
        X_pca = self.pca.transform(X)
        return self.lr.score(X_pca, y)

class EnhancedFeatureSelector:
    """Class for enhanced feature selection using Optuna with multiple model options"""
    
    def __init__(self, config: EnhancedFeatureSelectionConfig):
        """Initialize the feature selector with configuration"""
        self.config = config
        self.selected_features = None
        self.feature_importance = None
        self.study = None
        self.available_features = []
        self.feature_scores = []
        self.best_trials = []
    
    def load_data(self, input_file: str) -> Tuple[pd.DataFrame, pd.Series]:
        """Load and prepare data for feature selection"""
        logger.info(f"Loading data from {input_file}")
        
        if not os.path.exists(input_file):
            raise FileNotFoundError(f"Input file not found: {input_file}")
        
        # Load data
        df = pd.read_csv(input_file)
        
        # Check for missing values
        missing_values = df.isnull().sum()
        if missing_values.any():
            missing_cols = missing_values[missing_values > 0]
            error_msg = "Found missing values in the following columns:\n"
            for col, count in missing_cols.items():
                error_msg += f"- {col}: {count} missing values\n"
            raise ValueError(error_msg)
        
        # Check if target column exists
        if self.config.target_column not in df.columns:
            raise ValueError(f"Target column '{self.config.target_column}' not found in data")
        
        # Define all target columns that should be excluded from features
        target_columns = [
            'dd_3month',  # 3-month death/disability
            'dd_6month',  # 6-month death/disability
            'los_target', 
            '180_readmission',
            'icu_admission_date_and_tim', 
            "hospital_los"
        ]
        
        # Get target variable
        y = df[self.config.target_column]
        
        # Separate features and target, excluding all target columns and record_id
        columns_to_drop = [col for col in target_columns if col in df.columns] + ['record_id']
        X = df.drop(columns=columns_to_drop)
        
        logger.info(f"Data loaded. Features shape: {X.shape}")
        logger.info(f"Excluded columns from features: {columns_to_drop}")
        return X, y
    
    def _create_model(self, trial=None):
        """Create model for feature evaluation based on selected model type"""
        if self.config.model_type == 'logistic_regression':
            return LogisticRegression(
                penalty='l2',
                solver='liblinear',
                random_state=self.config.random_state,
                class_weight='balanced'
            )
        elif self.config.model_type == 'lasso':
            return Lasso(
                alpha=0.1,
                random_state=self.config.random_state,
                max_iter=10000,
                selection='random'
            )
        elif self.config.model_type == 'pca':
            return PCAFeatureSelector(n_components=min(len(self.selected_features) + 1, 
                                                      min(self.X.shape[0], self.X.shape[1])))
        elif self.config.model_type == 'random_forest':
            if trial is not None:
                # Define hyperparameter search space
                n_estimators = trial.suggest_int('n_estimators', 100, 300)
                max_depth = trial.suggest_int('max_depth', 5, 15)
                min_samples_split = trial.suggest_int('min_samples_split', 2, 10)
                min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 5)
                max_features = trial.suggest_categorical('max_features', ['sqrt', 'log2', None])
                
                return RandomForestClassifier(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    min_samples_split=min_samples_split,
                    min_samples_leaf=min_samples_leaf,
                    max_features=max_features,
                    random_state=self.config.random_state,
                    class_weight='balanced',
                    n_jobs=-1
                )
            else:
                # Use default parameters if no trial is provided
                return RandomForestClassifier(
                    n_estimators=100,
                    max_depth=None,
                    min_samples_split=2,
                    min_samples_leaf=1,
                    random_state=self.config.random_state,
                    class_weight='balanced',
                    n_jobs=-1
                )
        else:
            logger.warning(f"Unknown model type '{self.config.model_type}', using logistic regression")
            return LogisticRegression(
                penalty='l2',
                solver='liblinear',
                random_state=self.config.random_state,
                class_weight='balanced'
            )
    
    def _objective(self, trial):
        """Objective function for Optuna optimization"""
        # Select a feature from candidates
        add_feature = trial.suggest_categorical('add_feature', self.available_features)
        
        # Current feature set
        current_features = self.selected_features + [add_feature]
        
        # Create and evaluate model
        if self.config.model_type == 'random_forest':
            model = self._create_model(trial)
        else:
            model = self._create_model()
        X_subset = self.X[current_features]
        
        # Perform cross-validation
        cv = StratifiedKFold(n_splits=self.config.cv_folds, shuffle=True, 
                           random_state=self.config.random_state)
        scores = cross_val_score(
            model, X_subset, self.y, 
            cv=cv, 
            scoring=self.config.metric
        )
        
        mean_score = scores.mean()
        std_score = scores.std()
        
        # Record feature set and scores
        trial.set_user_attr("features", current_features)
        trial.set_user_attr("mean_score", mean_score)
        trial.set_user_attr("std_score", std_score)
        
        # For RandomForest, also record the hyperparameters
        if self.config.model_type == 'random_forest':
            trial.set_user_attr("hyperparameters", {
                'n_estimators': trial.params['n_estimators'],
                'max_depth': trial.params['max_depth'],
                'min_samples_split': trial.params['min_samples_split'],
                'min_samples_leaf': trial.params['min_samples_leaf'],
                'max_features': trial.params['max_features']
            })
        
        return mean_score
    
    def select_features(self, X: pd.DataFrame, y: pd.Series) -> List[str]:
        """Select features using Optuna with the specified model type"""
        logger.info(f"Starting enhanced feature selection with {self.config.model_type} model...")
        
        self.X = X
        self.y = y
        self.available_features = list(X.columns)
        self.selected_features = []
        
        # Incrementally add features
        for iteration in range(self.config.n_features):
            if not self.available_features:
                logger.info("No more available features.")
                break
                
            logger.info(f"Iteration {iteration+1}/{self.config.n_features}: "
                       f"Selecting from {len(self.available_features)} candidate features...")
            
            # Choose appropriate sampler based on whether we're doing hyperparameter tuning
            if self.config.model_type == 'random_forest':
                sampler = TPESampler(seed=self.config.random_state)
            else:
                sampler = GridSampler({'add_feature': self.available_features})
            
            # Create Optuna study
            study = optuna.create_study(
                study_name=f"feature_selection_iter{iteration}",
                direction='maximize',
                sampler=sampler
            )
            
            # Run optimization
            study.optimize(
                self._objective, 
                n_trials=min(self.config.n_trials, len(self.available_features))
            )
            
            # If no successful trials, stop search
            if not study.trials:
                logger.warning("No successful trials, stopping search.")
                break
                
            # Get best trial result
            best_trial = study.best_trial
            best_feature = best_trial.params['add_feature']
            best_score = best_trial.values[0]
            
            # Save results
            self.best_trials.append(best_trial)
            self.feature_scores.append(best_score)
            
            logger.info(f"Selected feature: {best_feature}, score: {best_score:.4f}")
            
            # Add to selected features list
            self.selected_features.append(best_feature)
            
            # Remove from available features
            if best_feature in self.available_features:
                self.available_features.remove(best_feature)
            
            # Early stopping check
            if self.config.early_stopping_rounds and len(self.feature_scores) >= self.config.early_stopping_rounds:
                recent_scores = self.feature_scores[(-self.config.early_stopping_rounds+1):]
                previous_scores = self.feature_scores[-self.config.early_stopping_rounds:-1]
                
                if np.mean(recent_scores) < np.mean(previous_scores):
                    logger.info(f"Early stopping triggered: Average score of recent {self.config.early_stopping_rounds} features has not improved.")
                    # Remove the last added feature as it decreased performance
                    self.selected_features.pop()
                    break
        
        # Create feature importance DataFrame
        importances = []
        for i, feature in enumerate(self.selected_features):
            if i < len(self.feature_scores):
                score = self.feature_scores[i]
                importances.append({
                    'Feature': feature,
                    'Importance': score,
                    'Rank': i + 1
                })
        
        self.feature_importance = pd.DataFrame(importances)
        if not self.feature_importance.empty:
            self.feature_importance = self.feature_importance.sort_values('Importance', ascending=False)
        
        logger.info(f"Feature selection complete. Selected {len(self.selected_features)} features.")
        print(self.selected_features)
        return self.selected_features
    
    def save_results(self, X: pd.DataFrame, y: pd.Series, output_dir: str = None, input_dir: str = None):
        """Save selected features, importance scores, and create new train/test/val datasets with selected features"""
        # Create method and target-specific output directory
        target_dir = Path(output_dir) / self.config.method / self.config.target_column
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)
        
        # Create filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_filename = f"{self.config.method}_{self.config.model_type}_{self.config.task_type}_{timestamp}"
        
        # Save feature importance
        if self.feature_importance is not None:
            importance_file = target_dir / f"{base_filename}_importance.csv"
            self.feature_importance.to_csv(importance_file, index=False)
            logger.info(f"Saved feature importance to {importance_file}")
        
        # Save selected features
        selected_features_file = target_dir / f"{base_filename}_selected_features.json"
        with open(selected_features_file, 'w') as f:
            json.dump(self.selected_features, f, indent=4)
        logger.info(f"Saved selected features to {selected_features_file}")
        
        # Process each split file
        for split in ['train', 'val', 'test']:
            input_file = Path(input_dir) / f"{split}.csv"
            if input_file.exists():
                # Read the split file
                split_data = pd.read_csv(input_file)
                
                # Select only the chosen features and target column
                selected_cols = self.selected_features + [self.config.target_column]
                split_data_selected = split_data[selected_cols]
                
                # Save the new split file with selected features
                output_file = target_dir / f"{base_filename}_{split}.csv"
                split_data_selected.to_csv(output_file, index=False)
                logger.info(f"Created {split} data with selected features at {output_file}")
            else:
                logger.warning(f"Could not find {split} data file at {input_file}")
        
        return str(target_dir)

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Select features using enhanced Optuna-based methods')
    parser.add_argument('--input_directory', type=str, required=True,
                      help='Directory containing train.csv, test.csv, and val.csv files')
    parser.add_argument('--method', type=str, default='optuna',
                      help='Feature selection method')
    parser.add_argument('--task_type', type=str, default='classification',
                      choices=['classification', 'regression'],
                      help='Type of task')
    parser.add_argument('--target_column', type=str, required=True,
                      help='Column to predict')
    parser.add_argument('--n_features', type=int, default=10,
                      help='Number of features to select')
    parser.add_argument('--model_type', type=str, default='logistic_regression',
                      choices=['logistic_regression', 'lasso', 'pca', 'random_forest'],
                      help='Model type for feature selection')
    parser.add_argument('--metric', type=str, default='roc_auc',
                      help='Evaluation metric for optimization')
    parser.add_argument('--n_trials', type=int, default=100,
                      help='Number of Optuna trials')
    parser.add_argument('--early_stopping_rounds', type=int, default=10,
                      help='Early stopping rounds for Optuna')
    parser.add_argument('--output_dir', type=str, default='feature_selection_results',
                      help='Directory to save results')
    
    args = parser.parse_args()
    
    # Create feature selection configuration
    config = EnhancedFeatureSelectionConfig(
        method=args.method,
        task_type=args.task_type,
        target_column=args.target_column,
        n_features=args.n_features,
        model_type=args.model_type,
        metric=args.metric,
        n_trials=args.n_trials,
        early_stopping_rounds=args.early_stopping_rounds
    )
    
    # Create feature selector and run
    selector = EnhancedFeatureSelector(config)
    
    # Load train data for feature selection
    train_file = os.path.join(args.input_directory, 'train.csv')
    X, y = selector.load_data(train_file)
    selector.select_features(X, y)
    
    # Save results using the input directory for split files
    selector.save_results(X, y, args.output_dir, args.input_directory)

if __name__ == "__main__":
    main()
