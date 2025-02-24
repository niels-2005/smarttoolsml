import logging

from scipy.stats import linregress

# LOG LEVELS
# DEBUG: Detailed information, typically of interest only when diagnosing problems.

# INFO: Confirmation that things are working as expected

# WARNING: An indication that something unexpected happened, or indicative of some problem in the near future (for example, "dis space low")
# The software is still working as expected

# ERROR: Due to a more seroius problem, the software has not been able to perform some function.

# CRITICAL: A serious error, indicating that the program itself may be unable to continue running


# needs to be specified at the beginning
logging.basicConfig(filename="logs.log", filemode="w", level=logging.DEBUG)
# filename, where to save the logs (needs .log)
# filemode "w", overwrite existing logs instead appending
# level (above levels)

# logs with date, time at beginning because "format="
logging.basicConfig(
    filename="logs.log",
    filemode="w",
    level=logging.DEBUG,
    format="%(asctime)s %(message)s",
)


def fit_trendline(year_timesteps, data):
    logging.info("Running fit_trendline function")
    try:
        # works if all is fine
        result = linregress(year_timesteps, data)
    except TypeError as e:
        logging.error(
            f"Both lists must contain floats or integers, got {year_timesteps.dtype} and {data.dtype} instead."
        )
        logging.exception(e)
        return 0.0, 0.0
    else:
        # if no error occurs
        slope = round(result.slope, 3)
        r_squared = round(result.rvalue**2, 3)
        logging.info(
            f"Completed fit_trendline function, Slope is {slope}, r_squared is {r_squared}"
        )
        return slope, r_squared
    finally:
        # after else statement delete for memory
        logging.info("Deleted slope and r_squared")
        del slope, r_squared
