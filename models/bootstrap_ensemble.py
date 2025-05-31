"""
Bootstrap Ensemble implementation for model training.
"""
import numpy as np
import pandas as pd
from typing import List, Optional, Any, Dict
import logging
from sklearn.base import BaseEstimator
from .random_forest_model import RandomForestModel

logger = logging.getLogger(__name__)

class BootstrapEnsemble:
    def __init__(
        self,
        n_models: int = 10,
        base_model_class: Any = RandomForestModel,
        params_file: Optional[str] = None,
        random_state: Optional[int] = None,
        model_params: Optional[Dict] = None,
        **kwargs
    ):
        """
        Initialize Bootstrap Ensemble
        
        Parameters:
        -----------
        n_models : int
            Number of models in the ensemble
        base_model_class : class
            Class of the base model to use
        params_file : str, optional
            Path to parameters file
        random_state : int, optional
            Random seed for reproducibility
        model_params : dict, optional
            Parameters to use for each base model
        **kwargs : dict
            Additional parameters for the base model
        """
        self.n_models = n_models
        self.base_model_class = base_model_class
        self.params_file = params_file
        self.random_state = random_state
        self.model_params = model_params or {}
        self.kwargs = kwargs
        self.models: List[BaseEstimator] = []
        self.feature_names_ = None
        
        np.random.seed(random_state)
        
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'BootstrapEnsemble':
        """
        Fit the ensemble using bootstrap samples
        
        Parameters:
        -----------
        X : DataFrame
            Training features
        y : Series
            Target variable
        """
        self.feature_names_ = X.columns.tolist()
        n_samples = len(X)
        
        logger.info(f"Training bootstrap ensemble with {self.n_models} models...")
        
        for i in range(self.n_models):
            # Create bootstrap sample
            indices = np.random.choice(n_samples, n_samples, replace=True)
            X_boot = X.iloc[indices]
            y_boot = y.iloc[indices]
            
            # Calculate out-of-bag indices
            oob_indices = list(set(range(n_samples)) - set(indices))
            
            logger.info(f"\nTraining model {i+1}/{self.n_models}")
            logger.info(f"Bootstrap sample size: {len(indices)}")
            logger.info(f"Out-of-bag sample size: {len(oob_indices)}")
            
            # Create and train a new model for each bootstrap sample
            model = self.base_model_class(params_file=self.params_file, **self.model_params)
            model.fit(X_boot, y_boot)
            
            # Evaluate on out-of-bag samples if available
            if oob_indices:
                X_oob = X.iloc[oob_indices]
                y_oob = y.iloc[oob_indices]
                oob_score = model.model.score(X_oob, y_oob)
                logger.info(f"Out-of-bag score: {oob_score:.4f}")
            
            self.models.append(model)
        
        logger.info("\nEnsemble training completed")
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions using majority voting
        
        Parameters:
        -----------
        X : DataFrame
            Features to predict
        
        Returns:
        --------
        ndarray
            Predicted classes
        """
        if not self.models:
            raise ValueError("Models not trained yet. Call fit first.")
        
        # Get predictions from all models
        predictions = np.array([model.predict(X) for model in self.models])
        
        # Use majority voting
        return np.apply_along_axis(
            lambda x: np.bincount(x).argmax(),
            axis=0,
            arr=predictions
        )
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict class probabilities by averaging probabilities from all models
        
        Parameters:
        -----------
        X : DataFrame
            Features to predict
        
        Returns:
        --------
        ndarray
            Predicted probabilities
        """
        if not self.models:
            raise ValueError("Models not trained yet. Call fit first.")
        
        # Get probability predictions from all models
        probas = np.array([model.predict_proba(X) for model in self.models])
        
        # Average probabilities across models
        return np.mean(probas, axis=0)
    
    def get_feature_importance(self, top_n: Optional[int] = None) -> pd.DataFrame:
        """
        Get average feature importance across all models
        
        Parameters:
        -----------
        top_n : int, optional
            Number of top features to return
        
        Returns:
        --------
        DataFrame
            Average feature importance scores
        """
        if not self.models:
            raise ValueError("Models not trained yet. Call fit first.")
        
        # Get feature importance from each model
        importances = []
        for model in self.models:
            model_importance = model.get_feature_importance()
            importances.append(model_importance['importance'].values)
        
        # Calculate average importance
        avg_importance = np.mean(importances, axis=0)
        
        # Create DataFrame
        importance_df = pd.DataFrame({
            'feature': self.feature_names_,
            'importance': avg_importance
        }).sort_values('importance', ascending=False)
        
        if top_n:
            return importance_df.head(top_n)
        return importance_df 