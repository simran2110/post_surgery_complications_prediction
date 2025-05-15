"""
Feature selection module for MIDAS risk prediction.
Provides various methods for feature selection including RFE, Random Forest importance, and more.
"""
# Standard library imports
import os
import logging
import warnings
from datetime import datetime
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Union

# Third-party imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import optuna
from optuna.samplers import TPESampler
from tqdm import tqdm
import shap
import argparse

# Scikit-learn imports
from sklearn.feature_selection import (
    SelectKBest, 
    f_classif, 
    f_regression, 
    mutual_info_classif, 
    mutual_info_regression,
    RFE, 
    RFECV
)
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.inspection import permutation_importance

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings('ignore')

@dataclass
class FeatureSelectionConfig:
    """Configuration for feature selection"""
    method: str  # 'univariate', 'rfe', 'rf_importance', 'mutual_info', 'optuna'
    task_type: str  # 'classification' or 'regression'
    target_column: str
    n_features: int = 20  # Number of features to select
    random_state: int = 42
    cv_folds: int = 5
    n_trials: int = 100  # Number of Optuna trials
    early_stopping_rounds: int = 10  # Early stopping rounds for Optuna

class ProgressCallback:
    """Callback class for tracking RFE progress"""
    def __init__(self, total_steps):
        self.pbar = tqdm(total=total_steps, desc="Eliminating features")
        self.current_step = 0
    
    def __call__(self, estimator, X, y):
        self.current_step += 1
        self.pbar.update(1)
        if self.current_step % 5 == 0:  # Log every 5 steps
            remaining_features = X.shape[1]
            logger.info(f"Step {self.current_step}: {remaining_features} features remaining")

class FeatureSelector:
    """Class for selecting features using different methods"""
    
    def __init__(self, config: FeatureSelectionConfig):
        """Initialize the feature selector with configuration"""
        self.config = config
        self.selected_features = None
        self.feature_importance = None
        self.study = None
    
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
            'icu_admission_date_and_tim'
        ]
        
        # Get target variable
        y = df[self.config.target_column]
        
        # Separate features and target, excluding all target columns and record_id
        columns_to_drop = [col for col in target_columns if col in df.columns] + ['record_id']
        X = df.drop(columns=columns_to_drop)
        
        logger.info(f"Data loaded. Features shape: {X.shape}")
        logger.info(f"Excluded columns from features: {columns_to_drop}")
        return X, y
    
    def univariate_selection(self, X: pd.DataFrame, y: pd.Series) -> List[str]:
        """Select features using univariate statistical tests"""
        logger.info("Performing univariate feature selection...")
        
        # Choose scoring function based on task type
        if self.config.task_type == 'classification':
            selector = SelectKBest(score_func=f_classif, k=self.config.n_features)
        else:
            selector = SelectKBest(score_func=f_regression, k=self.config.n_features)
        
        # Fit and transform
        selector.fit(X, y)
        
        # Get selected feature names
        selected_features = X.columns[selector.get_support()].tolist()
        scores = pd.DataFrame({
            'Feature': X.columns,
            'Score': selector.scores_
        }).sort_values('Score', ascending=False)
        
        self.feature_importance = scores
        logger.info(f"Selected {len(selected_features)} features using univariate selection")
        return selected_features
    
    def recursive_feature_elimination(self, X: pd.DataFrame, y: pd.Series) -> List[str]:
        """Select features using Recursive Feature Elimination"""
        logger.info("Performing recursive feature elimination...")
        
        # Choose base estimator based on task type
        if self.config.task_type == 'classification':
            estimator = LogisticRegression(random_state=self.config.random_state)
        else:
            estimator = LinearRegression()
        
        # Calculate total number of features to eliminate
        n_features_to_eliminate = X.shape[1] - self.config.n_features
        logger.info(f"Starting with {X.shape[1]} features, need to eliminate {n_features_to_eliminate} features")
        
        # Create RFE object with step size of 1
        rfe = RFE(
            estimator=estimator,
            n_features_to_select=self.config.n_features,
            step=1
        )
        
        # Add progress tracking
        progress_callback = ProgressCallback(n_features_to_eliminate)
        
        # Fit RFE with progress tracking
        logger.info("Starting feature elimination process...")
        rfe.fit(X, y, callback=progress_callback)
        progress_callback.pbar.close()
        
        # Get selected feature names
        selected_features = X.columns[rfe.support_].tolist()
        rankings = pd.DataFrame({
            'Feature': X.columns,
            'Ranking': rfe.ranking_
        }).sort_values('Ranking')
        
        self.feature_importance = rankings
        logger.info(f"Selected {len(selected_features)} features using RFE")
        logger.info("Feature elimination complete!")
        
        return selected_features
    
    def random_forest_importance(self, X: pd.DataFrame, y: pd.Series) -> List[str]:
        """Select features using Random Forest feature importance"""
        logger.info("Performing Random Forest feature importance selection...")
        
        # Choose Random Forest model based on task type
        if self.config.task_type == 'classification':
            rf = RandomForestClassifier(
                n_estimators=100,
                random_state=self.config.random_state
            )
        else:
            rf = RandomForestRegressor(
                n_estimators=100,
                random_state=self.config.random_state
            )
        
        # Fit Random Forest
        rf.fit(X, y)
        
        # Get feature importance
        importance = pd.DataFrame({
            'Feature': X.columns,
            'Importance': rf.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        # Select top features
        selected_features = importance.head(self.config.n_features)['Feature'].tolist()
        self.feature_importance = importance
        
        logger.info(f"Selected {len(selected_features)} features using Random Forest importance")
        return selected_features
    
    def mutual_information(self, X: pd.DataFrame, y: pd.Series) -> List[str]:
        """Select features using Mutual Information"""
        logger.info("Performing mutual information feature selection...")
        
        # Choose scoring function based on task type
        if self.config.task_type == 'classification':
            mi_scores = mutual_info_classif(X, y)
        else:
            mi_scores = mutual_info_regression(X, y)
        
        # Create importance DataFrame
        importance = pd.DataFrame({
            'Feature': X.columns,
            'MI_Score': mi_scores
        }).sort_values('MI_Score', ascending=False)
        
        # Select top features
        selected_features = importance.head(self.config.n_features)['Feature'].tolist()
        self.feature_importance = importance
        
        logger.info(f"Selected {len(selected_features)} features using mutual information")
        return selected_features
    
    def optuna_feature_selection(self, X: pd.DataFrame, y: pd.Series) -> List[str]:
        """Select features using Optuna optimization"""
        logger.info("Performing Optuna-based feature selection...")
        
        def objective(trial):
            # Suggest which features to use
            feature_mask = []
            for _ in range(len(X.columns)):
                feature_mask.append(trial.suggest_categorical(f'feature_{_}', [True, False]))
            
            # Get selected features
            selected_features = X.columns[feature_mask].tolist()
            
            if not selected_features:  # If no features selected, return worst score
                return float('-inf')
            
            # Choose model based on task type
            if self.config.task_type == 'classification':
                model = RandomForestClassifier(
                    n_estimators=100,
                    random_state=self.config.random_state
                )
            else:
                model = RandomForestRegressor(
                    n_estimators=100,
                    random_state=self.config.random_state
                )
            
            # Evaluate model with selected features
            try:
                scores = cross_val_score(
                    model, 
                    X[selected_features], 
                    y, 
                    cv=self.config.cv_folds,
                    scoring='roc_auc' if self.config.task_type == 'classification' else 'r2'
                )
                return scores.mean()
            except:
                return float('-inf')
        
        # Create Optuna study
        self.study = optuna.create_study(
            direction='maximize',
            sampler=TPESampler(seed=self.config.random_state)
        )
        
        # Run optimization
        self.study.optimize(
            objective, 
            n_trials=self.config.n_trials,
            show_progress_bar=True
        )
        
        # Get best feature set
        best_trial = self.study.best_trial
        feature_mask = [best_trial.params[f'feature_{i}'] for i in range(len(X.columns))]
        selected_features = X.columns[feature_mask].tolist()
        
        # Create feature importance DataFrame - only for selected features
        selected_indices = [i for i, selected in enumerate(feature_mask) if selected]
        importance = pd.DataFrame({
            'Feature': selected_features,
            'Selected': [True] * len(selected_features),
            'Importance': [self.study.best_trial.params.get(f'feature_{i}', 0) for i in selected_indices]
        }).sort_values('Importance', ascending=False)
        
        self.feature_importance = importance
        logger.info(f"Selected {len(selected_features)} features using Optuna")
        return selected_features
    
    def select_features(self, X: pd.DataFrame, y: pd.Series) -> List[str]:
        """Select features using the specified method"""
        if self.config.method == 'optuna':
            return self.optuna_feature_selection(X, y)
        elif self.config.method == 'univariate':
            return self.univariate_selection(X, y)
        elif self.config.method == 'rfe':
            return self.recursive_feature_elimination(X, y)
        elif self.config.method == 'rf_importance':
            return self.random_forest_importance(X, y)
        elif self.config.method == 'mutual_info':
            return self.mutual_information(X, y)
        else:
            raise ValueError(f"Unsupported feature selection method: {self.config.method}")
    
    def create_visualizations(self, X: pd.DataFrame, y: pd.Series, output_dir: str):
        """Create various visualizations for feature selection results"""
        logger.info("Creating feature selection visualizations...")
        
        # Create visualizations directory
        viz_dir = os.path.join(output_dir, f'visualization_{self.config.method}')
        if not os.path.exists(viz_dir):
            os.makedirs(viz_dir)
        
        # 1. Feature Importance Plot
        if self.feature_importance is not None:
            plt.figure(figsize=(12, 6))
            
            # Map method to importance column name
            importance_col_map = {
                'univariate': 'Score',
                'rf_importance': 'Importance',
                'mutual_info': 'MI_Score',
                'rfe': 'Ranking',
                'optuna': 'Importance'
            }
            
            # Get the correct importance column name for the current method
            importance_col = importance_col_map.get(self.config.method, 'Importance')
            
            # Check if the importance column exists in the data
            if importance_col in self.feature_importance.columns:
                importance_plot = sns.barplot(
                    data=self.feature_importance.head(20),
                    x=importance_col,
                    y='Feature'
                )
                plt.title(f'Top 20 Feature Importance Scores ({self.config.method})')
                plt.tight_layout()
                plt.savefig(os.path.join(viz_dir, 'feature_importance.png'))
                plt.close()
            else:
                logger.warning(f"Importance column '{importance_col}' not found in feature importance data")
        
        # 2. Correlation Heatmap for Selected Features
        if self.selected_features:
            plt.figure(figsize=(12, 10))
            corr_matrix = X[self.selected_features].corr()
            sns.heatmap(
                corr_matrix,
                annot=True,
                cmap='coolwarm',
                center=0,
                fmt='.2f'
            )
            plt.title('Correlation Heatmap of Selected Features')
            plt.tight_layout()
            plt.savefig(os.path.join(viz_dir, 'correlation_heatmap.png'))
            plt.close()
        
        # 3. SHAP Values (if using Random Forest or Optuna)
        if self.config.method in ['rf_importance', 'optuna']:
            try:
                # Create and train a model for SHAP values
                if self.config.task_type == 'classification':
                    model = RandomForestClassifier(n_estimators=100, random_state=42)
                else:
                    model = RandomForestRegressor(n_estimators=100, random_state=42)
                
                model.fit(X[self.selected_features], y)
                
                # Calculate SHAP values
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X[self.selected_features])
                
                # Plot SHAP summary
                plt.figure(figsize=(12, 8))
                if self.config.task_type == 'classification':
                    shap.summary_plot(shap_values[1], X[self.selected_features], show=False)
                else:
                    shap.summary_plot(shap_values, X[self.selected_features], show=False)
                plt.title('SHAP Value Summary')
                plt.tight_layout()
                plt.savefig(os.path.join(viz_dir, 'shap_summary.png'))
                plt.close()
                
                # Save SHAP values
                shap_df = pd.DataFrame(
                    shap_values[1] if self.config.task_type == 'classification' else shap_values,
                    columns=self.selected_features
                )
                shap_df.to_csv(os.path.join(viz_dir, 'shap_values.csv'), index=False)
                
            except Exception as e:
                logger.warning(f"Could not create SHAP values: {str(e)}")
        
        # 4. Permutation Importance
        try:
            if self.config.task_type == 'classification':
                model = RandomForestClassifier(n_estimators=100, random_state=42)
            else:
                model = RandomForestRegressor(n_estimators=100, random_state=42)
            
            model.fit(X[self.selected_features], y)
            perm_importance = permutation_importance(
                model, X[self.selected_features], y,
                n_repeats=10,
                random_state=42
            )
            
            # Plot permutation importance
            plt.figure(figsize=(12, 6))
            perm_importance_df = pd.DataFrame({
                'Feature': self.selected_features,
                'Importance': perm_importance.importances_mean
            }).sort_values('Importance', ascending=True)
            
            sns.barplot(
                data=perm_importance_df,
                x='Importance',
                y='Feature'
            )
            plt.title('Permutation Importance')
            plt.tight_layout()
            plt.savefig(os.path.join(viz_dir, 'permutation_importance.png'))
            plt.close()
            
        except Exception as e:
            logger.warning(f"Could not create permutation importance: {str(e)}")
        
        # 5. Feature Selection Progress (for Optuna)
        if self.config.method == 'optuna' and self.study is not None:
            # Plot optimization history
            plt.figure(figsize=(12, 6))
            optuna.visualization.matplotlib.plot_optimization_history(self.study)
            plt.title('Optuna Optimization History')
            plt.tight_layout()
            plt.savefig(os.path.join(viz_dir, 'optuna_history.png'))
            plt.close()
            
            # Plot parameter importance
            plt.figure(figsize=(12, 6))
            optuna.visualization.matplotlib.plot_param_importances(self.study)
            plt.title('Optuna Parameter Importance')
            plt.tight_layout()
            plt.savefig(os.path.join(viz_dir, 'optuna_importance.png'))
            plt.close()
        
        logger.info(f"Visualizations saved to {viz_dir}")

    def save_results(self, X: pd.DataFrame, y: pd.Series, output_dir: str = "feature_selection_results"):
        """Save selected features and importance scores"""
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Create filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_filename = f"{self.config.method}_{self.config.task_type}_{self.config.target_column}_{timestamp}"
        
        # Save selected features - ensure we only save the selected features
        selected_data = X[self.selected_features].copy()
        selected_data[self.config.target_column] = y
        
        output_file = os.path.join(output_dir, f"{base_filename}.csv")
        selected_data.to_csv(output_file, index=False)
        
        # Save feature importance
        if self.feature_importance is not None:
            # For Optuna method, only save selected features and their importance
            if self.config.method == 'optuna':
                importance_df = self.feature_importance[self.feature_importance['Selected'] == True].copy()
                importance_df = importance_df.sort_values('Importance', ascending=False)
            else:
                importance_df = self.feature_importance
                
            importance_file = os.path.join(output_dir, f"{base_filename}_importance.csv")
            importance_df.to_csv(importance_file, index=False)
            
            # If using Optuna, save optimization history
            if self.config.method == 'optuna' and self.study is not None:
                history_file = os.path.join(output_dir, f"{base_filename}_optuna_history.csv")
                history_df = pd.DataFrame({
                    'trial': range(len(self.study.trials)),
                    'value': [t.value for t in self.study.trials],
                    'state': [t.state for t in self.study.trials]
                })
                history_df.to_csv(history_file, index=False)
        
        # Create visualizations
        self.create_visualizations(X, y, output_dir)
        
        logger.info(f"Selected features saved to {output_file}")
        if self.feature_importance is not None:
            logger.info(f"Feature importance saved to {importance_file}")
        
        return output_file
    
    def select_and_save(self, input_file: str, output_dir: str = "feature_selection_results"):
        """Main feature selection and saving pipeline"""
        try:
            # Load data
            X, y = self.load_data(input_file)
            
            # Select features
            self.selected_features = self.select_features(X, y)
            
            # Save results
            return self.save_results(X, y, output_dir)
            
        except Exception as e:
            logger.error(f"Error in feature selection: {str(e)}")
            raise

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Select features using various methods')
    parser.add_argument('input_file', type=str, help='Path to preprocessed data file')
    parser.add_argument('--method', type=str, 
                      choices=['univariate', 'rfe', 'rf_importance', 'mutual_info', 'optuna'],
                      default='optuna', help='Feature selection method')
    parser.add_argument('--task_type', type=str, 
                      choices=['classification', 'regression'],
                      default='classification', help='Type of task')
    parser.add_argument('--target_column', type=str, required=True,
                      help='Column to predict')
    parser.add_argument('--n_features', type=int, default=20,
                      help='Number of features to select')
    parser.add_argument('--n_trials', type=int, default=100,
                      help='Number of Optuna trials (only for optuna method)')
    parser.add_argument('--early_stopping_rounds', type=int, default=10,
                      help='Early stopping rounds for Optuna (only for optuna method)')
    parser.add_argument('--output_dir', type=str, default='feature_selection_results',
                      help='Directory to save results')
    
    args = parser.parse_args()
    
    # Create feature selection configuration
    config = FeatureSelectionConfig(
        method=args.method,
        task_type=args.task_type,
        target_column=args.target_column,
        n_features=args.n_features,
        n_trials=args.n_trials,
        early_stopping_rounds=args.early_stopping_rounds
    )
    
    # Create feature selector and run
    selector = FeatureSelector(config)
    selector.select_and_save(args.input_file, args.output_dir)

if __name__ == "__main__":
    main() 