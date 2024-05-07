from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor

pipelines = [
    Pipeline([("scaler", StandardScaler()), ("reg", MultiOutputRegressor(Ridge()))]),
    Pipeline([("scaler", MinMaxScaler()), ("reg", MultiOutputRegressor(Ridge()))]),
    Pipeline([("scaler", RobustScaler()), ("reg", MultiOutputRegressor(Ridge()))]),
    
    Pipeline([("scaler", StandardScaler()), ("reg", MultiOutputRegressor(Lasso()))]),
    Pipeline([("scaler", MinMaxScaler()), ("reg", MultiOutputRegressor(Lasso()))]),
    Pipeline([("scaler", RobustScaler()), ("reg", MultiOutputRegressor(Lasso()))]),
    
    Pipeline([("scaler", StandardScaler()), ("reg", MultiOutputRegressor(ElasticNet()))]),
    Pipeline([("scaler", MinMaxScaler()), ("reg", MultiOutputRegressor(ElasticNet()))]),
    Pipeline([("scaler", RobustScaler()), ("reg", MultiOutputRegressor(ElasticNet()))]),
    
    Pipeline([("scaler", StandardScaler()), ("reg", MultiOutputRegressor(SVR()))]),
    Pipeline([("scaler", MinMaxScaler()), ("reg", MultiOutputRegressor(SVR()))]),
    Pipeline([("scaler", RobustScaler()), ("reg", MultiOutputRegressor(SVR()))]),
    
    Pipeline([("scaler", StandardScaler()), ("reg", RandomForestRegressor())]),  
    Pipeline([("scaler", MinMaxScaler()), ("reg", RandomForestRegressor())]),
    Pipeline([("scaler", RobustScaler()), ("reg", RandomForestRegressor())]),
    
    Pipeline([("scaler", StandardScaler()), ("reg", GradientBoostingRegressor())]),  
    Pipeline([("scaler", MinMaxScaler()), ("reg", GradientBoostingRegressor())]),
    Pipeline([("scaler", RobustScaler()), ("reg", GradientBoostingRegressor())]),
    
    Pipeline([("scaler", StandardScaler()), ("reg", KNeighborsRegressor())]),  
    Pipeline([("scaler", MinMaxScaler()), ("reg", KNeighborsRegressor())]),
    Pipeline([("scaler", RobustScaler()), ("reg", KNeighborsRegressor())]),
    
    Pipeline([("scaler", StandardScaler()), ("reg", DecisionTreeRegressor())]), 
    Pipeline([("scaler", MinMaxScaler()), ("reg", DecisionTreeRegressor())]),
    Pipeline([("scaler", RobustScaler()), ("reg", DecisionTreeRegressor())])
]