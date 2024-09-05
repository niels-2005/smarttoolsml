from sklearn.base import BaseEstimator, TransformerMixin 
from sklearn.preprocessing import LabelEncoder


class CustomLabelEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, variables=None):
        self.variables = variables
        self.encoders = {}
    
    def fit(self, X, y=None):
        for column in self.variables:
            le = LabelEncoder()
            le.fit(X[column].astype(str))
            self.encoders[column] = le
        return self
    
    def transform(self, X):
        X = X.copy() 
        for column in self.variables:
            X[column] = self.encoders[column].transform(X[column].astype(str))
        return X