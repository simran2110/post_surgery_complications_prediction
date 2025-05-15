"""
Hyperparameter optimization module for Random Forest, GAM, and Logistic Regression models.
Designed to work with preprocessed and feature-selected data for risk prediction.
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, StratifiedKFold
from sklearn.metrics import make_scorer, roc_auc_score
from pygam import LogisticGAM
import json
import os
from datetime import datetime
import logging
import argparse
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
            'class_weight': ['balanced', 'balanced_subsample']
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

def find_optimal_gam_hyperparameters(X_train, y_train, cv=5, random_state=42):
    """
    Find optimal hyperparameters for GAM model
    """
    param_grid = {
        'n_splines': [5, 10, 15, 20],
        'lam': [0.1, 1.0, 10.0],
        'max_iter': [100, 200, 300]
    }
    
    scores = []
    params = []
    
    # Manual grid search as GAM doesn't support sklearn's GridSearchCV
    cv_folds = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    
    for n_splines in param_grid['n_splines']:
        for lam in param_grid['lam']:
            for max_iter in param_grid['max_iter']:
                fold_scores = []
                
                for train_idx, val_idx in cv_folds.split(X_train, y_train):
                    X_fold_train, X_fold_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
                    y_fold_train, y_fold_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
                    
                    try:
                        # Initialize and fit GAM model
                        gam = LogisticGAM(n_splines=n_splines, lam=lam, max_iter=max_iter)
                        gam.fit(X_fold_train, y_fold_train)
                        
                        # Get predictions - handle both single and multi-class cases
                        try:
                            y_pred = gam.predict_proba(X_fold_val)
                            if isinstance(y_pred, np.ndarray) and len(y_pred.shape) == 2:
                                y_pred = y_pred[:, 1]  # Get probabilities for positive class
                            else:
                                y_pred = y_pred.flatten()  # Handle 1D case
                        except AttributeError:
                            # If predict_proba not available, use predict
                            y_pred = gam.predict(X_fold_val)
                        
                        fold_scores.append(roc_auc_score(y_fold_val, y_pred))
                    except Exception as e:
                        logger.warning(f"Error in GAM fitting: {str(e)}")
                        fold_scores.append(0.0)  # Assign poor score for failed fits
                
                mean_score = np.mean(fold_scores)
                scores.append(mean_score)
                params.append({
                    'n_splines': n_splines,
                    'lam': lam,
                    'max_iter': max_iter
                })
                
                logger.info(f"GAM params - n_splines: {n_splines}, lam: {lam}, "
                          f"max_iter: {max_iter}, AUC: {mean_score:.4f}")
    
    best_idx = np.argmax(scores)
    best_params = params[best_idx]
    logger.info(f"Best GAM parameters: {best_params}, AUC: {scores[best_idx]:.4f}")
    
    return best_params

def find_optimal_lr_hyperparameters(X_train, y_train, cv=5, random_state=42):
    """
    Find optimal hyperparameters for Logistic Regression model
    """
    param_grid = {
        'C': [0.001, 0.01, 0.1, 1.0, 10.0],
        'penalty': ['l1', 'l2'],
        'solver': ['liblinear', 'saga'],
        'class_weight': ['balanced', None]
    }
    
    base_model = LogisticRegression(
        max_iter=1000,
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

def find_best_hyperparameters(X_train, y_train, model_type='random_forest', 
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
    output_file = os.path.join(output_dir, f"{model_type}_params_{timestamp}.json")
    with open(output_file, 'w') as f:
        json.dump(best_params, f, indent=4)
    
    logger.info(f"Optimal parameters saved to: {output_file}")
    return best_params

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
    best_params = find_best_hyperparameters(
        X, y,
        model_type=args.model_type,
        cv=args.cv,
        random_state=args.random_state,
        output_dir=args.output_dir
    )

if __name__ == "__main__":
    main() 