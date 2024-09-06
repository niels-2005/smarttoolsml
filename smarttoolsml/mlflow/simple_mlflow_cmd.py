import mlflow
import argparse
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
import pandas as pd 
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# excecute before: mlflow ui


# set tracking uri
mlflow.set_tracking_uri("http://localhost:5000")


def load_data():
    URL = "http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
    try:
        df = pd.read_csv(URL, sep=";")
        return df
    except Exception as e:
        raise e


# calculate some metrics
def eval_function(actual,pred):
    rmse = mean_squared_error(actual,pred,squared=False)
    mae = mean_absolute_error(actual,pred)
    r2 = r2_score(actual,pred)
    return rmse,mae,r2


def main(alpha, l1_ratio):
    df = load_data()
    TARGET = "quality"
    X = df.drop(columns=TARGET)
    y = df[TARGET]
    X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=6, test_size=0.2)

    # set_experiment Name
    mlflow.set_experiment("ML-Model-1")
    
    # start mlflow run
    with mlflow.start_run():

        # log hyperparameters in mlflow
        mlflow.log_param("alpha",alpha)
        mlflow.log_param("l1_ratio",l1_ratio)

        # Model
        model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=6)
        model.fit(X_train,y_train)
        y_pred = model.predict(X_test)
        rmse,mae,r2 = eval_function(y_test,y_pred)

        # log metrics in mlflow
        mlflow.log_metric("rmse",rmse)
        mlflow.log_metric("mae",mae)
        mlflow.log_metric("r2-score",r2)

        # log model in mlflow
        mlflow.sklearn.log_model(model,"trained_model") # model, foldername
    
    # end mlflow run
    mlflow.end_run()


# execute on terminal commando
# python simple_mlflow_cmd.py -a 0.5 -l1 0.5
if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--alpha","-a", type=float, default=0.2)
    args.add_argument("--l1_ratio","-l1", type=float, default=0.3)
    parsed_args = args.parse_args()
    # parsed_args.param1
    main(parsed_args.alpha, parsed_args.l1_ratio)