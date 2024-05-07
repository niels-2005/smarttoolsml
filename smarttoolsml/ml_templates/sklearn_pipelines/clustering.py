from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

pipelines = [
    Pipeline([("scaler", StandardScaler()), ("cluster", KMeans(n_clusters=3))]),
    Pipeline([("scaler", MinMaxScaler()), ("cluster", KMeans(n_clusters=3))]),
    Pipeline([("scaler", RobustScaler()), ("cluster", KMeans(n_clusters=3))]),
]
