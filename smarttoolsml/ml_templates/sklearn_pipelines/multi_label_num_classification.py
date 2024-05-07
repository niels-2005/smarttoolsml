from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier, BaggingClassifier
from sklearn.neural_network import MLPClassifier
import numpy as np


pipelines = [
    Pipeline([("scaler", StandardScaler()), ("clf", MultiOutputClassifier(KNeighborsClassifier()))]),
    Pipeline([("scaler", MinMaxScaler()), ("clf", MultiOutputClassifier(KNeighborsClassifier()))]),
    Pipeline([("scaler", RobustScaler()), ("clf", MultiOutputClassifier(KNeighborsClassifier()))]),
    
    Pipeline([("scaler", StandardScaler()), ("clf", DecisionTreeClassifier())]), 
    Pipeline([("scaler", MinMaxScaler()), ("clf", DecisionTreeClassifier())]),
    Pipeline([("scaler", RobustScaler()), ("clf", DecisionTreeClassifier())]),
    
    Pipeline([("scaler", StandardScaler()), ("clf", RandomForestClassifier())]),  
    Pipeline([("scaler", MinMaxScaler()), ("clf", RandomForestClassifier())]),
    Pipeline([("scaler", RobustScaler()), ("clf", RandomForestClassifier())]),
    
    Pipeline([("scaler", StandardScaler()), ("clf", GradientBoostingClassifier())]),  
    Pipeline([("scaler", MinMaxScaler()), ("clf", GradientBoostingClassifier())]),
    Pipeline([("scaler", RobustScaler()), ("clf", GradientBoostingClassifier())]),
    
    Pipeline([("scaler", StandardScaler()), ("clf", ExtraTreesClassifier())]),  
    Pipeline([("scaler", MinMaxScaler()), ("clf", ExtraTreesClassifier())]),
    Pipeline([("scaler", RobustScaler()), ("clf", ExtraTreesClassifier())]),
    
    Pipeline([("scaler", StandardScaler()), ("clf", MultiOutputClassifier(BaggingClassifier()))]),
    Pipeline([("scaler", MinMaxScaler()), ("clf", MultiOutputClassifier(BaggingClassifier()))]),
    Pipeline([("scaler", RobustScaler()), ("clf", MultiOutputClassifier(BaggingClassifier()))]),
    
    Pipeline([("scaler", StandardScaler()), ("clf", MultiOutputClassifier(MLPClassifier()))]),
    Pipeline([("scaler", MinMaxScaler()), ("clf", MultiOutputClassifier(MLPClassifier()))]),
    Pipeline([("scaler", RobustScaler()), ("clf", MultiOutputClassifier(MLPClassifier()))])
]

def get_probabilites(pipeline, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray):
    pipeline.fit(X_train, y_train)
    probs = pipeline.predict_proba(X_test)
    return probs
