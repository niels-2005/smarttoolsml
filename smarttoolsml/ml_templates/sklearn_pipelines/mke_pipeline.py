from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def get_pipe(X_train, y_train):
    pipe_lr = make_pipeline(
        StandardScaler(),
        PCA(n_components=2),
        LogisticRegression(random_state=1, solver="lbfgs"),
    )

    pipe_lr.fit(X_train, y_train)
    return pipe_lr
