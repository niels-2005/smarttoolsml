from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder


class CustomLabelEncoder(BaseEstimator, TransformerMixin):
    """_summary_

    Args:
        BaseEstimator (_type_): _description_
        TransformerMixin (_type_): _description_

    Example usage:
            classification_pipeline = Pipeline(
        [
            ('DomainProcessing', pp.DomainProcessing(variable_to_add=config.FEATURE_TO_ADD)),
            ('DropFeatures', pp.DropColumns(variables_to_drop=config.DROP_FEATURES)),

            # LabelEncoder here
            ('LabelEncoder', pp.CustomLabelEncoder(variables=config.FEATURES_TO_ENCODE)),

            ('LogTransform', pp.LogTransforms(variables=config.LOG_FEATURES)),
            ('LogisticClassifier', LogisticRegression(random_state=0))
        ]
    )
    """

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
