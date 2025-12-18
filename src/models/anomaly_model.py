from sklearn.ensemble import IsolationForest
import numpy as np

class AnomalyDetector:
    """
    Isolation Forest based anomaly detector for fraud detection.
    """
    
    def __init__(self, contamination=0.05, random_state=42):
        """
        Initialize the anomaly detector.
        
        Args:
            contamination: Expected proportion of anomalies (default 0.05 = 5%)
            random_state: Random seed for reproducibility
        """
        self.model = IsolationForest(
            contamination=contamination,
            random_state=random_state,
            n_estimators=100
        )
        self.feature_columns = None
        self.is_fitted = False
    
    def fit(self, X):
        """
        Train the anomaly detection model.
        
        Args:
            X: Feature matrix (DataFrame or numpy array)
        """
        if isinstance(X, np.ndarray):
            X_train = X
        else:
            # Assume DataFrame
            self.feature_columns = X.columns.tolist()
            X_train = X.values
        
        self.model.fit(X_train)
        self.is_fitted = True
    
    def predict(self, X):
        """
        Predict anomalies.
        
        Args:
            X: Feature matrix (DataFrame or numpy array)
        
        Returns:
            Array of predictions: -1 for anomaly, 1 for normal
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        if isinstance(X, np.ndarray):
            X_pred = X
        else:
            # Assume DataFrame
            if self.feature_columns:
                X_pred = X[self.feature_columns].values
            else:
                X_pred = X.values
        
        return self.model.predict(X_pred)
    
    def fit_predict(self, X):
        """
        Fit the model and predict on the same data.
        
        Args:
            X: Feature matrix (DataFrame or numpy array)
        
        Returns:
            Array of predictions: -1 for anomaly, 1 for normal
        """
        self.fit(X)
        return self.predict(X)


