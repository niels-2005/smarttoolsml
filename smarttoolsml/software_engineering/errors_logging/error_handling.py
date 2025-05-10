import pandas as pd
from scipy.stats import linregress


def fit_trendline(year_timesteps, data):
    try:
        # works if all is fine
        result = linregress(year_timesteps, data)
    except TypeError:
        # if "TypeError" occurs, handling the error and return default values (other functions might crash)
        print(
            f"Both lists must contain only float or integers, got {data.dtype} and {year_timesteps.dtype} instead."
        )
        return 0.0, 0.0
    else:
        # if no error occurs
        slope = round(result.slope, 3)
        r_squared = round(result.rvalue ** 2, 3)
        return slope, r_squared
    finally:
        # after else statement delete for memory
        del slope, r_squared


def load_data(path):
    try:
        df = pd.read_csv(path)
        return df
    except Exception as e:
        raise e
