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
    Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression())]),
    Pipeline([("scaler", MinMaxScaler()), ("clf", LogisticRegression())]),
    Pipeline([("scaler", RobustScaler()), ("clf", LogisticRegression())]),
    Pipeline([("scaler", StandardScaler()), ("clf", SVC())]),
    Pipeline([("scaler", MinMaxScaler()), ("clf", SVC())]),
    Pipeline([("scaler", RobustScaler()), ("clf", SVC())]),
    Pipeline([("scaler", StandardScaler()), ("clf", RidgeClassifier())]),
    Pipeline([("scaler", MinMaxScaler()), ("clf", RidgeClassifier())]),
    Pipeline([("scaler", RobustScaler()), ("clf", RidgeClassifier())]),
    Pipeline([("scaler", StandardScaler()), ("clf", KNeighborsClassifier())]),
    Pipeline([("scaler", MinMaxScaler()), ("clf", KNeighborsClassifier())]),
    Pipeline([("scaler", RobustScaler()), ("clf", KNeighborsClassifier())]),
    Pipeline([("scaler", StandardScaler()), ("clf", DecisionTreeClassifier())]),
    Pipeline([("scaler", MinMaxScaler()), ("clf", DecisionTreeClassifier())]),
    Pipeline([("scaler", RobustScaler()), ("clf", DecisionTreeClassifier())]),
    Pipeline([("scaler", StandardScaler()), ("clf", RandomForestClassifier())]),
    Pipeline([("scaler", MinMaxScaler()), ("clf", RandomForestClassifier())]),
    Pipeline([("scaler", RobustScaler()), ("clf", RandomForestClassifier())]),
    Pipeline([("scaler", StandardScaler()), ("clf", GradientBoostingClassifier())]),
    Pipeline([("scaler", MinMaxScaler()), ("clf", GradientBoostingClassifier())]),
    Pipeline([("scaler", RobustScaler()), ("clf", GradientBoostingClassifier())]),
    Pipeline([("scaler", StandardScaler()), ("clf", AdaBoostClassifier())]),
    Pipeline([("scaler", MinMaxScaler()), ("clf", AdaBoostClassifier())]),
    Pipeline([("scaler", RobustScaler()), ("clf", AdaBoostClassifier())]),
    Pipeline([("scaler", StandardScaler()), ("clf", ExtraTreesClassifier())]),
    Pipeline([("scaler", MinMaxScaler()), ("clf", ExtraTreesClassifier())]),
    Pipeline([("scaler", RobustScaler()), ("clf", ExtraTreesClassifier())]),
    Pipeline([("scaler", StandardScaler()), ("clf", BaggingClassifier())]),
    Pipeline([("scaler", MinMaxScaler()), ("clf", BaggingClassifier())]),
    Pipeline([("scaler", RobustScaler()), ("clf", BaggingClassifier())]),
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
    Pipeline([("scaler", StandardScaler()), ("clf", SGDClassifier())]),
    Pipeline([("scaler", MinMaxScaler()), ("clf", SGDClassifier())]),
    Pipeline([("scaler", RobustScaler()), ("clf", SGDClassifier())]),
    Pipeline([("scaler", StandardScaler()), ("clf", XGBClassifier())]),
    Pipeline([("scaler", MinMaxScaler()), ("clf", XGBClassifier())]),
    Pipeline([("scaler", RobustScaler()), ("clf", XGBClassifier())]),
    Pipeline([("scaler", StandardScaler()), ("clf", LGBMClassifier())]),
    Pipeline([("scaler", MinMaxScaler()), ("clf", LGBMClassifier())]),
    Pipeline([("scaler", RobustScaler()), ("clf", LGBMClassifier())]),
]
