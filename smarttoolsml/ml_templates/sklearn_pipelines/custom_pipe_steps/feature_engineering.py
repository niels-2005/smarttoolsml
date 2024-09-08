from sklearn.base import BaseEstimator, TransformerMixin


class SumFeaturesToOne(BaseEstimator, TransformerMixin):
    """_summary_

    Args:
        BaseEstimator (_type_): _description_
        TransformerMixin (_type_): _description_

    Example usage:
            classification_pipeline = Pipeline(
        [
            # Sum Features as example
            ('SumFeatures',pp.SumFeaturesToOne(variables_to_sum = config.VARIABLES_TO_SUM, new_column = config.NEW_FEATURE_ADD)),

            ('DropFeatures', pp.DropColumns(variables_to_drop=config.DROP_FEATURES)),
            ('LabelEncoder',pp.CustomLabelEncoder(variables=config.FEATURES_TO_ENCODE)),
            ('LogTransform',pp.LogTransforms(variables=config.LOG_FEATURES)),
            ('LogisticClassifier',LogisticRegression(random_state=0))
        ]
    )

        # Feature Engineering, mehrere Features werden zu einem neuen zusammen summiert
    """

    def __init__(self, variables_to_sum=None, new_column=None):
        self.new_column = new_column
        self.variables_to_sum = variables_to_sum

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X[self.new_column] = X[self.variables_to_sum].sum(axis=1)
        return X
