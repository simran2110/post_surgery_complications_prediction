# models/gam_model.py
from pygam import LogisticGAM, s, f
import json
import logging
import numpy as np
import pandas as pd
from typing import Dict, Optional, Any
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score

logger = logging.getLogger(__name__)

class GAMModel:
    def __init__(self, params_file: Optional[str] = None, **kwargs):
        """
        Initialize GAM model.
        
        Args:
            params_file: Path to JSON file containing model parameters
            **kwargs: Direct model parameters to use instead of loading from file
        """
        self.model = None
        self.params = self._load_params(params_file) if params_file else {}
        self.feature_names = None
    
    def _load_params(self, params_file):
        with open(params_file, 'r') as f:
            return json.load(f)
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'GAMModel':
        """
        Fit the GAM model.
        
        Args:
            X: Feature matrix
            y: Target variable
        """
        self.feature_names = X.columns.tolist()
        print("feature_names")
        print(self.feature_names)
        print(X.head())
        
        print("y")
        print(y.head())
        # Build the 'terms' specification for pyGAM
        terms = self._build_terms(X)
        print("checkin which is categorical and numerical terms")
        print(terms)
        # Separate out GAM-specific params vs fitting params
        gam_kwargs = dict(self.params)          # copy all
        print("gam_kwargs")
        print(gam_kwargs)
        gam_kwargs.pop('n_splines', None)       # handled in _build_terms
        # Now instantiate with the built terms
        self.model = LogisticGAM(terms=terms, **gam_kwargs)

        # Fit — convert to numpy under the hood
        self.model.fit(X.values, y.values)
        
        return self
    
    def _build_terms(self, X):
        """Auto-detect categorical vs numerical and build term list"""
        terms = []
        for i in range(X.shape[1]):
            col = X.iloc[:, i]
            unique_vals = col.unique()
            if len(unique_vals) < 10 and np.all(unique_vals == unique_vals.astype(int)):
                terms.append(f(i))  # Treat as categorical
            else:
                terms.append(s(i, n_splines=self.params.get('n_splines', 10)))
        return sum(terms[1:], start=terms[0]) if terms else s(0)  # Avoid sum(0 + term)

    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            X: Feature matrix
        """
        if self.model is None:
            raise ValueError("Model not trained yet")
        return self.model.predict(X)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Get prediction probabilities.
        
        Args:
            X: Feature matrix
        """
        if self.model is None:
            raise ValueError("Model not trained yet")
        # GAM returns probabilities directly
        return self.model.predict_proba(X)
    
    def evaluate(self, X: pd.DataFrame, y: pd.Series, dataset_name: str = "Test") -> Dict[str, float]:
        """
        Evaluate the model.
        
        Args:
            X: Feature matrix
            y: True labels
            dataset_name: Name of the dataset being evaluated
            
        Returns:
            Dictionary of evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model not trained yet")
            
        y_pred = self.predict(X)
        proba = self.predict_proba(X)
        y_pred_proba = proba[:, 1] if proba.ndim == 2 else proba
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred),
            'recall': recall_score(y, y_pred),
            'f1': f1_score(y, y_pred),
            'roc_auc': roc_auc_score(y, y_pred_proba)
        }
        
        logger.info(f"\nMetrics for {dataset_name} set:")
        for metric, value in metrics.items():
            logger.info(f"{metric}: {value:.4f}")
            
        return y_pred, y_pred_proba, metrics  

    @property
    def estimator(self):
        """Get the underlying GAM estimator"""
        if self.model is None:
            raise ValueError("Model not trained yet")
        return self.model

def main():
    # Simulate 2 categorical and 3 numerical features
    np.random.seed(42)
    cat1 = np.random.randint(0, 3, size=(100, 1))  # Categorical with 3 levels
    cat2 = np.random.randint(0, 2, size=(100, 1))  # Categorical with 2 levels
    num1 = np.random.rand(100, 1)  # Numerical
    num2 = np.random.rand(100, 1)  # Numerical
    num3 = np.random.rand(100, 1)  # Numerical

    X = np.hstack([cat1, cat2, num1, num2, num3])
    y = np.random.randint(0, 2, size=100)  # Binary target

    # Set hyperparameters (manually or via JSON file)
    params = {'lam': 0.001, 'max_iter': 50, 'n_splines': 4}
    model = GAMModel()
    model.params = params  # override internal params dict

    model.fit(X, y)
    preds = model.predict_proba(X)

    print("Predicted probabilities:\n", preds)
    
if __name__ == "__main__":
    main()