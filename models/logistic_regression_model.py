# models/logistic_regression_model.py
from sklearn.linear_model import LogisticRegression
import json
import logging
from typing import Dict, Optional, Any

logger = logging.getLogger(__name__)

class LogisticRegressionModel:
    def __init__(self, params_file: Optional[str] = None, **kwargs):
        """
        Initialize Logistic Regression model.
        
        Args:
            params_file: Path to JSON file containing model parameters
            **kwargs: Direct model parameters to use instead of loading from file
        """
        self.model = None
        # Use kwargs if provided, otherwise try to load from file
        self.params = kwargs if kwargs else (self._load_params(params_file) if params_file else {})
    
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
    
    def fit(self, X, y):
        """Fit the model with the given parameters"""
        logger.info(f"Training Logistic Regression with parameters: {self.params}")
        self.model = LogisticRegression(**self.params)
        self.model.fit(X, y)
        return self
    
    def predict(self, X):
        """Make predictions"""
        if self.model is None:
            raise ValueError("Model not trained yet")
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """Get prediction probabilities"""
        if self.model is None:
            raise ValueError("Model not trained yet")
        return self.model.predict_proba(X)
    
    def evaluate(self, X, y, dataset_name: str = "Test"):
        """Evaluate the model"""
        if self.model is None:
            raise ValueError("Model not trained yet")
            
        y_pred = self.predict(X)
        y_pred_proba = self.predict_proba(X)[:, 1]
        
        # Calculate metrics
        from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
        
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
        """Get the underlying scikit-learn estimator"""
        if self.model is None:
            raise ValueError("Model not trained yet")
        return self.model