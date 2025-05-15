# models/logistic_regression_model.py
from sklearn.linear_model import LogisticRegression
import json
import logging

logger = logging.getLogger(__name__)

class LogisticRegressionModel:
    def __init__(self, params_file=None):
        self.model = None
        self.params = self._load_params(params_file) if params_file else {}
    
    def _load_params(self, params_file):
        with open(params_file, 'r') as f:
            return json.load(f)
    
    def fit(self, X, y):
        self.model = LogisticRegression(**self.params)
        self.model.fit(X, y)
        return self
    
    def predict(self, X):
        return self.model.predict(X)
    
    def predict_proba(self, X):
        return self.model.predict_proba(X)