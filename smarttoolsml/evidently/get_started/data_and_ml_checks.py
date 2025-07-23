import pandas as pd
from sklearn import datasets

from evidently import Dataset
from evidently import DataDefinition
from evidently import Report
from evidently.presets import DataDriftPreset, DataSummaryPreset

# Load the adult dataset from OpenML
adult_data = datasets.fetch_openml(name="adult", version=2, as_frame="auto")
adult = adult_data.frame


# define reference and production datasets (reference is the training set, production is the test set)
adult_ref = adult[~adult.education.isin(["Some-college", "HS-grad", "Bachelors"])]
adult_prod = adult[adult.education.isin(["Some-college", "HS-grad", "Bachelors"])]

# map the column types
schema = DataDefinition(
    numerical_columns=[
        "education-num",
        "age",
        "capital-gain",
        "hours-per-week",
        "capital-loss",
        "fnlwgt",
    ],
    categorical_columns=[
        "education",
        "occupation",
        "native-country",
        "workclass",
        "marital-status",
        "relationship",
        "race",
        "sex",
        "class",
    ],
)

# Create Evidently Datasets to work with:
eval_data_1 = Dataset.from_pandas(pd.DataFrame(adult_prod), data_definition=schema)

eval_data_2 = Dataset.from_pandas(pd.DataFrame(adult_ref), data_definition=schema)

# define the report
report = Report([DataDriftPreset()])

# run the report
my_eval = report.run(eval_data_1, eval_data_2)


# local preview with my_eval
my_eval

# or with single dataframe (only DataSummaryPreset will work)
report = Report([DataSummaryPreset()])
my_eval = report.run(eval_data_1)


# my_eval.json()
# my_eval.dict()
# my_report.save_html("file.html")


# Alternatively, try DataSummaryPreset that will generate a summary of all columns in the dataset,
# and run auto-generated Tests to check for data quality and core descriptive stats.
report = Report([DataSummaryPreset()], include_tests="True")
my_eval = report.run(eval_data_1, eval_data_2)
