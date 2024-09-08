from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler


def get_preprocessor_for_pipe(categorical_fea: list, numerical_fea: list):
    """_summary_

    Args:
        categorical_fea (list): _description_
        numerical_fea (list): _description_

    Returns:
        _type_: _description_

    Example usage:
        categorical_fea = ["f1", "f2"]
        numerical_fea = ["f3", "f4"]

        prep = get_preprocessor_for_pipe(categorical_fea, numerical_fea)

        Using in Pipeline:
                classification_pipeline = Pipeline(
        steps=[
            ('preprocessor', prep),
            ('classifier', LogisticRegression(random_state=0))
        ]
    )
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(sparse=False, drop="first"), categorical_fea),
            ("num", StandardScaler(), numerical_fea),
        ]
    )

    # preprocessor = ColumnTransformer(
    #     transformers=[
    #         ("cat", OrdinalEncoder(), categorical_fea),
    #         ("num", StandardScaler(), numerical_fea)
    #     ]
    # )
    return preprocessor
