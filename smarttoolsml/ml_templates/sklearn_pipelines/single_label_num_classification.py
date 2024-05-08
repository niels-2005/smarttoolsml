from lightgbm import LGBMClassifier
from sklearn.ensemble import (
    AdaBoostClassifier,
    BaggingClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier
from sklearn.naive_bayes import BernoulliNB, GaussianNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

pipelines = [
    Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(n_jobs=-1))]),
    Pipeline([("scaler", MinMaxScaler()), ("clf", LogisticRegression(n_jobs=-1))]),
    Pipeline([("scaler", RobustScaler()), ("clf", LogisticRegression(n_jobs=-1))]),
    Pipeline([("scaler", StandardScaler()), ("clf", SVC())]),
    Pipeline([("scaler", MinMaxScaler()), ("clf", SVC())]),
    Pipeline([("scaler", RobustScaler()), ("clf", SVC())]),
    Pipeline([("scaler", StandardScaler()), ("clf", RidgeClassifier())]),
    Pipeline([("scaler", MinMaxScaler()), ("clf", RidgeClassifier())]),
    Pipeline([("scaler", RobustScaler()), ("clf", RidgeClassifier())]),
    Pipeline([("scaler", StandardScaler()), ("clf", KNeighborsClassifier(n_jobs=-1))]),
    Pipeline([("scaler", MinMaxScaler()), ("clf", KNeighborsClassifier(n_jobs=-1))]),
    Pipeline([("scaler", RobustScaler()), ("clf", KNeighborsClassifier(n_jobs=-1))]),
    Pipeline([("scaler", StandardScaler()), ("clf", DecisionTreeClassifier())]),
    Pipeline([("scaler", MinMaxScaler()), ("clf", DecisionTreeClassifier())]),
    Pipeline([("scaler", RobustScaler()), ("clf", DecisionTreeClassifier())]),
    Pipeline([("scaler", StandardScaler()), ("clf", RandomForestClassifier(n_jobs=-1))]),
    Pipeline([("scaler", MinMaxScaler()), ("clf", RandomForestClassifier(n_jobs=-1))]),
    Pipeline([("scaler", RobustScaler()), ("clf", RandomForestClassifier(n_jobs=-1))]),
    Pipeline([("scaler", StandardScaler()), ("clf", GradientBoostingClassifier())]),
    Pipeline([("scaler", MinMaxScaler()), ("clf", GradientBoostingClassifier())]),
    Pipeline([("scaler", RobustScaler()), ("clf", GradientBoostingClassifier())]),
    Pipeline([("scaler", StandardScaler()), ("clf", AdaBoostClassifier())]),
    Pipeline([("scaler", MinMaxScaler()), ("clf", AdaBoostClassifier())]),
    Pipeline([("scaler", RobustScaler()), ("clf", AdaBoostClassifier())]),
    Pipeline([("scaler", StandardScaler()), ("clf", ExtraTreesClassifier(n_jobs=-1))]),
    Pipeline([("scaler", MinMaxScaler()), ("clf", ExtraTreesClassifier(n_jobs=-1))]),
    Pipeline([("scaler", RobustScaler()), ("clf", ExtraTreesClassifier(n_jobs=-1))]),
    Pipeline([("scaler", StandardScaler()), ("clf", BaggingClassifier(n_jobs=-1))]),
    Pipeline([("scaler", MinMaxScaler()), ("clf", BaggingClassifier(n_jobs=-1))]),
    Pipeline([("scaler", RobustScaler()), ("clf", BaggingClassifier(n_jobs=-1))]),
    Pipeline([("scaler", StandardScaler()), ("clf", BernoulliNB())]),
    Pipeline([("scaler", MinMaxScaler()), ("clf", BernoulliNB())]),
    Pipeline([("scaler", RobustScaler()), ("clf", BernoulliNB())]),
    Pipeline([("scaler", StandardScaler()), ("clf", MultinomialNB())]),
    Pipeline([("scaler", MinMaxScaler()), ("clf", MultinomialNB())]),
    Pipeline([("scaler", RobustScaler()), ("clf", MultinomialNB())]),
    Pipeline([("scaler", StandardScaler()), ("clf", GaussianNB())]),
    Pipeline([("scaler", MinMaxScaler()), ("clf", GaussianNB())]),
    Pipeline([("scaler", RobustScaler()), ("clf", GaussianNB())]),
    Pipeline([("scaler", StandardScaler()), ("clf", MLPClassifier())]),
    Pipeline([("scaler", MinMaxScaler()), ("clf", MLPClassifier())]),
    Pipeline([("scaler", RobustScaler()), ("clf", MLPClassifier())]),
    Pipeline([("scaler", StandardScaler()), ("clf", SGDClassifier(n_jobs=-1))]),
    Pipeline([("scaler", MinMaxScaler()), ("clf", SGDClassifier(n_jobs=-1))]),
    Pipeline([("scaler", RobustScaler()), ("clf", SGDClassifier(n_jobs=-1))]),
    Pipeline([("scaler", StandardScaler()), ("clf", XGBClassifier())]),
    Pipeline([("scaler", MinMaxScaler()), ("clf", XGBClassifier())]),
    Pipeline([("scaler", RobustScaler()), ("clf", XGBClassifier())]),
    Pipeline([("scaler", StandardScaler()), ("clf", LGBMClassifier(n_jobs=-1))]),
    Pipeline([("scaler", MinMaxScaler()), ("clf", LGBMClassifier(n_jobs=-1))]),
    Pipeline([("scaler", RobustScaler()), ("clf", LGBMClassifier(n_jobs=-1))]),
]
