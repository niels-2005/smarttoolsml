from sklearn.base import BaseEstimator, TransformerMixin


class DropColumns(BaseEstimator, TransformerMixin):
    """_summary_

    Args:
        BaseEstimator (_type_): _description_
        TransformerMixin (_type_): _description_

    Example usage:
            classification_pipeline = Pipeline(
        [
            ('DomainProcessing',pp.DomainProcessing(variable_to_add = config.FEATURE_TO_ADD)),

            # Drop specific Features
            ('DropFeatures', pp.DropColumns(variables_to_drop=config.DROP_FEATURES)),

            ('LabelEncoder',pp.CustomLabelEncoder(variables=config.FEATURES_TO_ENCODE)),
            ('LogTransform',pp.LogTransforms(variables=config.LOG_FEATURES)),
            ('LogisticClassifier',LogisticRegression(random_state=0))
        ]
    )
    """

    def __init__(self, variables_to_drop=None):
        self.variables_to_drop = variables_to_drop

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.drop(columns=self.variables_to_drop)
        return X
