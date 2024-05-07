from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor

pipelines = [
    Pipeline([("scaler", StandardScaler()), ("reg", LinearRegression())]),
    Pipeline([("scaler", MinMaxScaler()), ("reg", LinearRegression())]),
    Pipeline([("scaler", RobustScaler()), ("reg", LinearRegression())]),
    
    Pipeline([("scaler", StandardScaler()), ("reg", Ridge())]),
    Pipeline([("scaler", MinMaxScaler()), ("reg", Ridge())]),
    Pipeline([("scaler", RobustScaler()), ("reg", Ridge())]),
    
    Pipeline([("scaler", StandardScaler()), ("reg", Lasso())]),
    Pipeline([("scaler", MinMaxScaler()), ("reg", Lasso())]),
    Pipeline([("scaler", RobustScaler()), ("reg", Lasso())]),
    
    Pipeline([("scaler", StandardScaler()), ("reg", ElasticNet())]),
    Pipeline([("scaler", MinMaxScaler()), ("reg", ElasticNet())]),
    Pipeline([("scaler", RobustScaler()), ("reg", ElasticNet())]),
    
    Pipeline([("scaler", StandardScaler()), ("reg", SVR())]),
    Pipeline([("scaler", MinMaxScaler()), ("reg", SVR())]),
    Pipeline([("scaler", RobustScaler()), ("reg", SVR())]),
    
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
