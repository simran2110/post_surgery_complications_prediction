"""
Hyperparameter optimization module for Random Forest, GAM, and Logistic Regression models.
Designed to work with preprocessed and feature-selected data for risk prediction.
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, StratifiedKFold, ParameterGrid
from sklearn.metrics import make_scorer, roc_auc_score
from pygam import LogisticGAM
import json
import os
from datetime import datetime
import logging
import argparse
from sklearn.preprocessing import StandardScaler, RobustScaler
import warnings
from itertools import product


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def find_optimal_rf_hyperparameters(X_train, y_train, cv=5, random_state=42):
    """
    Find optimal hyperparameters for Random Forest model
    """
    # Calculate class weights based on imbalance
    pos_ratio = y_train.mean()
    neg_pos_ratio = (1 - pos_ratio) / pos_ratio if pos_ratio > 0 else 1.0
    
    logger.info(f"Class imbalance ratio (negative/positive): {neg_pos_ratio:.2f}")
    
    # Adapt parameter grid based on feature count
    n_features = X_train.shape[1]
    
    if n_features <= 30:
        param_grid = {
            'n_estimators': [100, 200, 300, 400, 500],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2'],
            'class_weight': ['balanced', 'balanced_subsample'],
            'bootstrap': [True, False]
        }
    else:
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [None, 20, 30],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2],
            'max_features': ['sqrt', 'log2'],
            'class_weight': ['balanced']
        }

    base_model = RandomForestClassifier(random_state=random_state, n_jobs=-1)
    
    search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_grid,
        n_iter=50,
        cv=StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state),
        scoring='roc_auc',
        n_jobs=-1,
        verbose=1,
        random_state=random_state
    )
    
    logger.info("Starting Random Forest hyperparameter search...")
    search.fit(X_train, y_train)
    
    logger.info(f"Best AUC: {search.best_score_:.4f}")
    logger.info(f"Best parameters: {search.best_params_}")
    
    return search.best_params_

def analyze_nan_values(df):
    """
    Analyze and display information about NaN values in a DataFrame
    
    Parameters:
    -----------
    df : DataFrame
        Input DataFrame to analyze
    
    Returns:
    --------
    DataFrame
        Summary of NaN values per column
    """
    # Calculate NaN statistics
    nan_stats = pd.DataFrame({
        'total_nan': df.isnull().sum(),
        'percentage_nan': (df.isnull().sum() / len(df) * 100).round(2),
        'mean_value': df.mean(),
        'median_value': df.median()
    })
    
    # Filter only columns with NaN values
    nan_stats = nan_stats[nan_stats['total_nan'] > 0]
    
    if len(nan_stats) > 0:
        logger.info("\nNaN Analysis Summary:")
        logger.info("=====================")
        for col in nan_stats.index:
            logger.info(f"\nColumn: {col}")
            logger.info(f"Total NaN values: {nan_stats.loc[col, 'total_nan']}")
            logger.info(f"Percentage NaN: {nan_stats.loc[col, 'percentage_nan']}%")
            logger.info(f"Mean value: {nan_stats.loc[col, 'mean_value']:.4f}")
            logger.info(f"Median value: {nan_stats.loc[col, 'median_value']:.4f}")
    else:
        logger.info("No NaN values found in the dataset")
    
    return nan_stats

def find_optimal_gam_hyperparameters(X_train, y_train, cv_folds=5, random_state=42):
    """
    Finds the optimal GAM hyperparameters using a two-stage dynamic grid search.
    """
    warnings.filterwarnings("ignore")

    # Handle missing values
    if isinstance(X_train, pd.DataFrame):
        X_train = X_train.fillna(0)
    elif isinstance(X_train, np.ndarray):
        X_train = np.nan_to_num(X_train)

    if isinstance(y_train, pd.Series):
        y_train = y_train.fillna(0)
    elif isinstance(y_train, np.ndarray):
        y_train = np.nan_to_num(y_train)

    # Scale features
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_train_scaled = np.clip(X_train_scaled, -10, 10)

    # ===== STAGE 1: Coarse Grid Search =====
    coarse_grid = {
        'n_splines': [5, 10],
        'lam': [0.01, 0.1, 1]
    }

    best_score = -np.inf
    best_params = None

    def evaluate_grid(param_grid):
        nonlocal best_score, best_params
        results = []
        for n_splines, lam in product(param_grid['n_splines'], param_grid['lam']):
            scores = []
            cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
            for train_idx, val_idx in cv.split(X_train_scaled, y_train):
                X_tr, X_val = X_train_scaled[train_idx], X_train_scaled[val_idx]
                y_tr, y_val = y_train[train_idx], y_train[val_idx]
                try:
                    gam = LogisticGAM(n_splines=n_splines, lam=lam).fit(X_tr, y_tr)
                    y_pred = gam.predict_proba(X_val)
                    y_pred = np.clip(y_pred, 1e-6, 1 - 1e-6)
                    auc = roc_auc_score(y_val, y_pred)
                    scores.append(auc)
                except:
                    continue

            if scores:
                mean_auc = np.mean(scores)
                results.append((mean_auc, n_splines, lam))
                if mean_auc > best_score:
                    best_score = mean_auc
                    best_params = {'n_splines': n_splines, 'lam': lam}

        return results

    print("🔍 Stage 1: Coarse Grid Search")
    stage1_results = evaluate_grid(coarse_grid)

    # ===== STAGE 2: Focused Grid Around Best =====
    best_n_splines = best_params['n_splines']
    best_lam = best_params['lam']

    refined_grid = {
        'n_splines': sorted(list(set([best_n_splines - 2, best_n_splines, best_n_splines + 2]))),
        'lam': sorted(list(set([
            round(best_lam / 2, 4),
            best_lam,
            round(best_lam * 2, 4)
        ])))
    }

    # Ensure values are within safe bounds
    refined_grid['n_splines'] = [n for n in refined_grid['n_splines'] if n > 2]
    refined_grid['lam'] = [l for l in refined_grid['lam'] if l > 0]

    print("\n🎯 Stage 2: Focused Grid Search")
    stage2_results = evaluate_grid(refined_grid)

    print(f"\n✅ Best AUC: {best_score:.4f} with params: {best_params}")
    return best_params, best_score
    

def find_optimal_lr_hyperparameters(X_train, y_train, cv=5, random_state=42):
    """
    Find optimal hyperparameters for Logistic Regression model
    with valid solver–penalty combinations and tol tuning
    """
    # Split param_grid based on valid solver-penalty pairs
    param_grid = [
        {
            'C': [0.001, 0.01, 0.1, 1.0, 10.0],
            'penalty': ['l1', 'l2'],
            'solver': ['liblinear'],
            'class_weight': ['balanced', None],
            'tol': [1e-4, 1e-3]
        },
        {
            'C': [0.001, 0.01, 0.1, 1.0, 10.0],
            'penalty': ['l1', 'l2'],
            'solver': ['saga'],
            'class_weight': ['balanced', None],
            'tol': [1e-4, 1e-3]
        }
    ]

    base_model = LogisticRegression(
        max_iter=5000,
        random_state=random_state,
        n_jobs=-1
    )

    search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        scoring='roc_auc',
        cv=StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state),
        n_jobs=-1,
        verbose=1
    )

    logger.info("Starting Logistic Regression hyperparameter search...")
    search.fit(X_train, y_train)

    logger.info(f"Best AUC: {search.best_score_:.4f}")
    logger.info(f"Best parameters: {search.best_params_}")

    return search.best_params_

def HyperparameterTuner(X_train, y_train, model_type='random_forest', 
                            cv=5, random_state=42, output_dir='hyperparameter_results'):
    """
    Main function to find optimal hyperparameters for specified model type
    
    Parameters:
    -----------
    X_train : DataFrame
        Training features
    y_train : Series
        Training target
    model_type : str
        One of ['random_forest', 'gam', 'logistic_regression']
    cv : int
        Number of cross-validation folds
    random_state : int
        Random seed
    output_dir : str
        Directory to save results
    
    Returns:
    --------
    dict
        Optimal hyperparameters
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
            
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    logger.info(f"\nOptimizing hyperparameters for {model_type} model...")
    
    # Choose optimization function based on model type
    if model_type == 'random_forest':
        best_params = find_optimal_rf_hyperparameters(X_train, y_train, cv, random_state)
    elif model_type == 'gam':
        best_params = find_optimal_gam_hyperparameters(X_train, y_train, cv, random_state)
    elif model_type == 'logistic_regression':
        best_params = find_optimal_lr_hyperparameters(X_train, y_train, cv, random_state)
    else:
        raise ValueError("Model type must be one of: 'random_forest', 'gam', 'logistic_regression'")
    
    # Save parameters
    output_file = os.path.join(output_dir, f"{model_type}_params.json")
    with open(output_file, 'w') as f:
        json.dump(best_params, f, indent=4)
    
    logger.info(f"Optimal parameters saved to: {output_file}")
    return best_params

def find_hyperparameters_for_feature_selection(X_train, y_train, cv=3, random_state=42):
    """
    Find optimal hyperparameters focused on stable feature importance scores
    Uses a smaller, focused parameter grid and cross-validation
    """
    # Calculate class weights based on imbalance
    pos_ratio = y_train.mean()
    neg_pos_ratio = (1 - pos_ratio) / pos_ratio if pos_ratio > 0 else 1.0
    
    logger.info(f"Class imbalance ratio (negative/positive): {neg_pos_ratio:.2f}")
    
    # Parameter grid focused on feature importance stability
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 20, 30],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2'],
        'class_weight': ['balanced', 'balanced_subsample']
    }

    base_model = RandomForestClassifier(random_state=random_state, n_jobs=-1)
    
    search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_grid,
        n_iter=20,  # Reduced iterations for feature selection phase
        cv=cv,      # Smaller CV for efficiency
        scoring='roc_auc',  # AUC is more stable for feature selection
        n_jobs=-1,
        verbose=1,
        random_state=random_state
    )
    
    logger.info("Finding optimal hyperparameters for feature selection...")
    search.fit(X_train, y_train)
    
    logger.info(f"Best AUC for feature selection: {search.best_score_:.4f}")
    logger.info(f"Best parameters for feature selection: {search.best_params_}")
    
    return search.best_params_

def main():
    """Main entry point"""
    
    
    parser = argparse.ArgumentParser(description='Find optimal hyperparameters for ML models')
    parser.add_argument('--input_file', type=str, 
                        help='Path to preprocessed and feature-selected data')
    parser.add_argument('--model_type', type=str, 
                       choices=['random_forest', 'gam', 'logistic_regression'],
                       required=True, help='Model type to optimize')
    parser.add_argument('--target_column', type=str, required=True,
                       help='Name of target column')
    parser.add_argument('--cv', type=int, default=5,
                       help='Number of cross-validation folds')
    parser.add_argument('--random_state', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--output_dir', type=str, default='hyperparameter_results',
                       help='Directory to save results')
    
    args = parser.parse_args()
    
    # Load data
    data = pd.read_csv(args.input_file)
    X = data.drop(columns=[args.target_column])
    y = data[args.target_column]
    
    # Find optimal hyperparameters
    best_params = HyperparameterTuner(
        X, y,
        model_type=args.model_type,
        cv=args.cv,
        random_state=args.random_state,
        output_dir=args.output_dir
    )

if __name__ == "__main__":
    main() 