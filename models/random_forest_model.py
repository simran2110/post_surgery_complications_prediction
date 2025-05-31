"""
Enhanced Random Forest model implementation with comprehensive evaluation capabilities.
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (roc_curve, auc, confusion_matrix, 
                           classification_report, precision_recall_curve)
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
import seaborn as sns
import json
import logging
import pickle
from pathlib import Path
import os
from typing import Dict, Any, Optional, Union, List, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RandomForestModel:
    """
    Random Forest model wrapper with comprehensive evaluation capabilities.
    
    Attributes:
        model: Trained RandomForestClassifier instance
        params: Dictionary of model parameters
        feature_names_: List of feature names used in training
        feature_importances_: DataFrame of feature importances
    """
    
    def __init__(self, params_file: Optional[str] = None, feature_selection_file: Optional[str] = None, **kwargs):
        """
        Initialize the Random Forest model.
        
        Args:
            params_file: Path to JSON file containing model parameters
            feature_selection_file: Path to CSV file containing pre-selected features
            **kwargs: Direct model parameters to use instead of loading from file
        """
        self.model = None
        self.feature_names_ = None
        # Use kwargs if provided, otherwise try to load from file
        self.params = kwargs if kwargs else (self._load_params(params_file) if params_file else {})
        self.feature_importances_ = None
        self.best_threshold_ = None
        self.target_column_ = None
        self.selected_features_ = self._load_selected_features(feature_selection_file) if feature_selection_file else None
        
    def _load_params(self, params_file: str) -> Dict[str, Any]:
        """Load parameters from JSON file"""
        try:
            with open(params_file, 'r') as f:
                params = json.load(f)
            logger.info(f"Loaded parameters from {params_file}: {params}")
            return params
        except Exception as e:
            logger.warning(f"Error loading parameters from {params_file}: {str(e)}")
            return {}
            
    def _load_selected_features(self, feature_selection_file: str) -> List[str]:
        """Load pre-selected features from feature selection results"""
        try:
            if not os.path.exists(feature_selection_file):
                raise FileNotFoundError(f"Feature selection file not found: {feature_selection_file}")
            
            # Try different possible formats of feature selection files
            df = pd.read_csv(feature_selection_file)
            
            # Case 1: Direct list of features
            if len(df.columns) == 1:
                features = df.iloc[:, 0].tolist()
            # Case 2: Feature importance format with 'Feature' column
            elif 'Feature' in df.columns:
                features = df['Feature'].tolist()
            # Case 3: Feature importance format with 'feature' column
            elif 'feature' in df.columns:
                features = df['feature'].tolist()
            else:
                raise ValueError("Could not determine features from the file format")
            
            logger.info(f"Loaded {len(features)} pre-selected features from {feature_selection_file}")
            return features
            
        except Exception as e:
            logger.warning(f"Error loading selected features from {feature_selection_file}: {str(e)}")
            logger.warning("Will proceed with default feature filtering")
            return None
    
    def _filter_features(self, X: pd.DataFrame, target_column: Optional[str] = None) -> pd.DataFrame:
        """
        Filter features based on pre-selected features or exclusion list
        
        Parameters:
        -----------
        X : DataFrame
            Input features
        target_column : str, optional
            Current target column being predicted
            
        Returns:
        --------
        DataFrame
            Filtered features
        """
        # If we have pre-selected features, use those
        if self.selected_features_ is not None:
            missing_cols = set(self.selected_features_) - set(X.columns)
            if missing_cols:
                raise ValueError(f"Missing required pre-selected features: {missing_cols}")
            return X[self.selected_features_]
        
        # Otherwise, use stored feature names from training or filter by exclusion
        if self.feature_names_ is not None:
            # Use stored feature names during prediction
            cols_to_keep = self.feature_names_
            # Verify all required features are present
            missing_cols = set(cols_to_keep) - set(X.columns)
            if missing_cols:
                raise ValueError(f"Missing required features: {missing_cols}")
        else:
            # During training, filter out excluded columns
            exclude_cols = {
                'record_id',  # Identifier column
                'dd_3month',  # 3-month death/disability
                'dd_6month',  # 6-month death/disability
                'los_target',  # Length of stay target
                '180_readmission',  # 180-day readmission
                'icu_admission_date_and_tim'  # ICU admission
            }
            
            # Remove current target from exclusion list if it's there and provided
            if target_column is not None:
                exclude_cols.discard(target_column)
                
            cols_to_keep = [col for col in X.columns if col not in exclude_cols]
        
        if not cols_to_keep:
            raise ValueError("No features remaining after filtering!")
        
        logger.info(f"Using {len(cols_to_keep)} features")
        return X[cols_to_keep]
    
    def _handle_class_weights(self, y: pd.Series) -> Dict[int, float]:
        """Calculate balanced class weights based on class distribution"""
        neg_count = (y == 0).sum()
        pos_count = (y == 1).sum()
        total = len(y)
        
        # Calculate balanced weights
        neg_weight = total / (2 * neg_count) if neg_count > 0 else 1.0
        pos_weight = total / (2 * pos_count) if pos_count > 0 else 1.0
        
        class_weights = {0: neg_weight, 1: pos_weight}
        logger.info(f"Class distribution - Negative: {neg_count}, Positive: {pos_count}")
        logger.info(f"Calculated class weights: {class_weights}")
        return class_weights
    
    def fit(self, X: pd.DataFrame, y: pd.Series, test_data: Optional[Tuple[pd.DataFrame, pd.Series]] = None, validation_data: Optional[Tuple[pd.DataFrame, pd.Series]] = None) -> 'RandomForestModel':
        """
        Fit the random forest model with class weight balancing and optional validation
        
        Parameters:
        -----------
        X : DataFrame
            Training features
        y : Series
            Target variable
        validation_data : tuple, optional
            (X_val, y_val) for monitoring performance during training
        """
        # Store target column name
        self.target_column_ = y.name if hasattr(y, 'name') else None
        
        # Ensure y is binary (0/1)
        y = y.astype(int)
        
        # Filter features
        X_filtered = self._filter_features(X, self.target_column_)
        
        # Store feature names for prediction if not using pre-selected features
        if self.selected_features_ is None:
            self.feature_names_ = X_filtered.columns.tolist()
        
        # Update parameters with class weights and regularization
        model_params = self.params.copy()
        if 'class_weight' not in model_params:
            model_params['class_weight'] = self._handle_class_weights(y)
        
        # # Set default parameters with stronger regularization
        # model_params.setdefault('n_estimators', 200)  # Increased for stability
        # model_params.setdefault('random_state', 42)
        # model_params.setdefault('n_jobs', -1)
        # model_params.setdefault('max_features', 'sqrt')
        # model_params.setdefault('min_samples_leaf', 8)  # Increased for regularization
        # model_params.setdefault('min_samples_split', 10)  # Added for regularization
        # model_params.setdefault('max_depth', 15)  # Added depth limit
        # model_params.setdefault('min_weight_fraction_leaf', 0.1)  # Added for regularization
        
        logger.info(f"Training Random Forest with parameters: {model_params}")
        
        # Initialize and train model
        self.model = RandomForestClassifier(**model_params)
        self.model.fit(X_filtered, y)
        
        # Calculate and log feature importances
        self._calculate_feature_importances()
        self._log_feature_importance(top_n=20)
        
        # # Evaluate on training set
        # train_metrics, train_y_pred, train_y_pred_proba = self.evaluate(X_filtered, y, "Training")
        # print("\n---train_metrics")
        # print(train_metrics)
        # print("\n---train_y_pred")
        # print(train_y_pred)
        # print("\n---train_y_pred_proba")
        # print(train_y_pred_proba)
        # logger.info("\nTraining Metrics:")
        # for metric, value in train_metrics.items():
        #     logger.info(f"{metric}: {value:.4f}")
        
        # # Evaluate on test set if provided
        # if test_data is not None:
        #     X_test, y_test = test_data
        #     y_test = y_test.astype(int)  # Ensure test data is also binary
        #     X_test_filtered = self._filter_features(X_test)  # No target_column needed here as we use stored features
        #     test_metrics = self.evaluate(X_test_filtered, y_test, "Test")
        #     logger.info("\nTest Metrics:")
        #     for metric, value in test_metrics.items():
        #         logger.info(f"{metric}: {value:.4f}")
                
        # # Evaluate on validation set if provided
        # if validation_data is not None:
        #     X_val, y_val = validation_data
        #     y_val = y_val.astype(int)  # Ensure validation data is also binary
        #     X_val_filtered = self._filter_features(X_val)  # No target_column needed here as we use stored features
        #     val_metrics = self.evaluate(X_val_filtered, y_val, "Validation")
        #     logger.info("\nValidation Metrics:")
        #     for metric, value in val_metrics.items():
        #         logger.info(f"{metric}: {value:.4f}")
        
        return self
    
    def fit_old(self, X: pd.DataFrame, y: pd.Series, test_data: Optional[Tuple[pd.DataFrame, pd.Series]] = None, validation_data: Optional[Tuple[pd.DataFrame, pd.Series]] = None) -> 'RandomForestModel':
        """
        Fit the random forest model with class weight balancing and optional validation
        
        Parameters:
        -----------
        X : DataFrame
            Training features
        y : Series
            Target variable
        validation_data : tuple, optional
            (X_val, y_val) for monitoring performance during training
        """
        # Store target column name
        self.target_column_ = y.name if hasattr(y, 'name') else None
        
        # Ensure y is binary (0/1)
        y = y.astype(int)
        
        # Filter features
        X_filtered = self._filter_features(X, self.target_column_)
        
        # Store feature names for prediction if not using pre-selected features
        if self.selected_features_ is None:
            self.feature_names_ = X_filtered.columns.tolist()
        
        # Update parameters with class weights and regularization
        model_params = self.params.copy()
        if 'class_weight' not in model_params:
            model_params['class_weight'] = self._handle_class_weights(y)
        
        # Set default parameters with stronger regularization
        model_params.setdefault('n_estimators', 200)  # Increased for stability
        model_params.setdefault('random_state', 42)
        model_params.setdefault('n_jobs', -1)
        model_params.setdefault('max_features', 'sqrt')
        model_params.setdefault('min_samples_leaf', 8)  # Increased for regularization
        model_params.setdefault('min_samples_split', 10)  # Added for regularization
        model_params.setdefault('max_depth', 15)  # Added depth limit
        model_params.setdefault('min_weight_fraction_leaf', 0.1)  # Added for regularization
        
        logger.info(f"Training Random Forest with parameters: {model_params}")
        
        # Initialize and train model
        self.model = RandomForestClassifier(**model_params)
        self.model.fit(X_filtered, y)
        
        # Calculate and log feature importances
        self._calculate_feature_importances()
        self._log_feature_importance(top_n=20)
        
        # Evaluate on training set
        train_metrics, train_y_pred, train_y_pred_proba = self.evaluate(X_filtered, y, "Training")
        print("\n---train_metrics")
        print(train_metrics)
        print("\n---train_y_pred")
        print(train_y_pred)
        print("\n---train_y_pred_proba")
        print(train_y_pred_proba)
        logger.info("\nTraining Metrics:")
        for metric, value in train_metrics.items():
            logger.info(f"{metric}: {value:.4f}")
        
        # Evaluate on test set if provided
        if test_data is not None:
            X_test, y_test = test_data
            y_test = y_test.astype(int)  # Ensure test data is also binary
            X_test_filtered = self._filter_features(X_test)  # No target_column needed here as we use stored features
            test_metrics = self.evaluate(X_test_filtered, y_test, "Test")
            logger.info("\nTest Metrics:")
            for metric, value in test_metrics.items():
                logger.info(f"{metric}: {value:.4f}")
                
        # Evaluate on validation set if provided
        if validation_data is not None:
            X_val, y_val = validation_data
            y_val = y_val.astype(int)  # Ensure validation data is also binary
            X_val_filtered = self._filter_features(X_val)  # No target_column needed here as we use stored features
            val_metrics = self.evaluate(X_val_filtered, y_val, "Validation")
            logger.info("\nValidation Metrics:")
            for metric, value in val_metrics.items():
                logger.info(f"{metric}: {value:.4f}")
        
        return self
    
    
    def _calculate_feature_importances(self) -> None:
        """Calculate and store feature importances"""
        feature_names = self.selected_features_ if self.selected_features_ is not None else self.feature_names_
        importances = pd.DataFrame({
            'feature': feature_names,
            'importance': self.model.feature_importances_
        })
        self.feature_importances_ = importances.sort_values(
            'importance', ascending=False
        ).reset_index(drop=True)
    
    def predict(self, X: pd.DataFrame, threshold: Optional[float] = None) -> np.ndarray:
        """
        Make predictions using the fitted model.
        
        Args:
            X: Features to make predictions on
            threshold: Custom threshold for binary classification
            
        Returns:
            Array of predictions
        """
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        # Filter features using stored feature names
        X_filtered = self._filter_features(X)
            
        if threshold is None:
            return self.model.predict(X_filtered)
        else:
            probas = self.predict_proba(X_filtered)
            return (probas[:, 1] >= threshold).astype(int)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Get prediction probabilities"""
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        # Filter features using stored feature names
        X_filtered = self._filter_features(X)
        return self.model.predict_proba(X_filtered)
    
    def evaluate(self, X: pd.DataFrame, y: pd.Series, dataset_name: str = "Test") -> Dict[str, float]:
        """
        Comprehensive model evaluation
        
        Parameters:
        -----------
        X : DataFrame
            Features
        y : Series
            True labels
        dataset_name : str
            Name of the dataset for logging
            
        Returns:
        --------
        Dict[str, float]
            Dictionary of evaluation metrics
        """
        # Filter features if not already filtered
        feature_names = self.selected_features_ if self.selected_features_ is not None else self.feature_names_
        if set(X.columns) != set(feature_names):
            X = self._filter_features(X)
            
        y_pred = self.predict(X)
        y_pred_proba = self.predict_proba(X)[:, 1]
        
        # Calculate ROC curve and AUC
        fpr, tpr, thresholds = roc_curve(y, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        # Calculate Precision-Recall curve
        precision, recall, _ = precision_recall_curve(y, y_pred_proba)
        pr_auc = auc(recall, precision)
        
        # Find optimal threshold
        optimal_idx = np.argmax(tpr - fpr)
        self.best_threshold_ = thresholds[optimal_idx]
        
        # Get classification report
        report = classification_report(y, y_pred, output_dict=True)
        
        metrics = {
            'accuracy': report['accuracy'],
            'precision': report['1']['precision'],
            'recall': report['1']['recall'],
            'f1': report['1']['f1-score'],
            'roc_auc': roc_auc,
            'pr_auc': pr_auc
        }
        
        # Plot confusion matrix
        self.plot_confusion_matrix(y, y_pred, dataset_name)
        
        return y_pred, y_pred_proba, metrics
    
    def plot_confusion_matrix(self, y_true: pd.Series, y_pred: np.ndarray, dataset_name: str) -> None:
        """Plot confusion matrix"""
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - {dataset_name} Set')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.show()
    
    def plot_feature_importance(self, top_n: int = 20) -> None:
        """Plot feature importance"""
        importance_df = self.get_feature_importance(top_n)
        
        plt.figure(figsize=(10, 6))
        sns.barplot(x='importance', y='feature', data=importance_df)
        plt.title(f'Top {top_n} Most Important Features')
        plt.xlabel('Importance Score')
        plt.tight_layout()
        plt.show()
    
    def get_feature_importance(self, top_n: Optional[int] = None) -> pd.DataFrame:
        """Get feature importance scores"""
        if self.model is None:
            raise ValueError("Model not trained yet")
            
        feature_names = self.selected_features_ if self.selected_features_ is not None else self.feature_names_
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        if top_n:
            return importance_df.head(top_n)
        return importance_df
    
    def cross_validate(self, X: pd.DataFrame, y: pd.Series, n_splits: int = 5) -> Dict[str, List[float]]:
        """
        Perform stratified cross-validation
        
        Parameters:
        -----------
        X : DataFrame
            Features
        y : Series
            Target variable
        n_splits : int
            Number of cross-validation folds
            
        Returns:
        --------
        Dict[str, List[float]]
            Cross-validation metrics
        """
        # Filter features first
        X_filtered = self._filter_features(X, y.name if hasattr(y, 'name') else None)
        
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        metrics = {
            'accuracy': [], 'precision': [], 'recall': [], 
            'f1': [], 'roc_auc': [], 'pr_auc': []
        }
        
        logger.info(f"\nPerforming {n_splits}-fold cross-validation...")
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X_filtered, y), 1):
            X_train_fold = X_filtered.iloc[train_idx]
            y_train_fold = y.iloc[train_idx]
            X_val_fold = X_filtered.iloc[val_idx]
            y_val_fold = y.iloc[val_idx]
            
            # Train model on fold
            self.fit(X_train_fold, y_train_fold)
            
            # Evaluate on validation fold
            fold_metrics = self.evaluate(X_val_fold, y_val_fold, f"Fold {fold}")
            
            # Store metrics
            for metric, value in fold_metrics.items():
                metrics[metric].append(value)
        
        # Log average metrics
        logger.info("\nCross-validation Results:")
        for metric, values in metrics.items():
            mean_value = np.mean(values)
            std_value = np.std(values)
            logger.info(f"{metric}: {mean_value:.4f} (±{std_value:.4f})")
        
        return metrics
    
    def save_model(self, filepath: str) -> None:
        """Save the model to a file"""
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
        logger.info(f"Model saved to {filepath}")
    
    @classmethod
    def load_model(cls, filepath: str) -> 'RandomForestModel':
        """Load a saved model"""
        with open(filepath, 'rb') as f:
            model = pickle.load(f)
        logger.info(f"Model loaded from {filepath}")
        return model
    
    def _log_feature_importance(self, top_n: int = 10) -> None:
        """Log top feature importances"""
        importance_df = self.get_feature_importance(top_n=top_n)
        logger.info("\nTop %d most important features:", top_n)
        for idx, row in importance_df.iterrows():
            logger.info(f"{row['feature']}: {row['importance']:.4f}")

    @property
    def estimator(self):
        """Get the underlying scikit-learn estimator"""
        if self.model is None:
            raise ValueError("Model not trained yet")
        return self.model