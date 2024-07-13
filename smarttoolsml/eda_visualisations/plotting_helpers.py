import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def custom_xlabels_ax(data: pd.DataFrame):
    """How to make custom xticks on plots with ax plots

    Args:
        data (pd.DataFrame): _description_
    """
    plt.figure(figsize=(8, 5))
    ax = sns.countplot(data=data, x="sex", hue="target_names")
    plt.title("Sex Balance")
    ax.set_xticks([0, 1])  #
    ax.set_xticklabels(["female", "male"])

    plt.show()


def custom_xlabels_facetgrid(data: pd.DataFrame):
    """How to make custom xlabels on plot with facetgrid

    Args:
        data (pd.DataFrame): _description_
    """
    g = sns.catplot(x="sex", y="age", hue="target_names", data=data)
    plt.title("Balance between Age and Target")
    g.set_xticklabels(["female", "male"])
    plt.show()


def custom_hue_names(df: pd.DataFrame):
    plt.figure(figsize=(10, 5))
    ax = sns.countplot(data=df, x="sex", hue="output")
    plt.title("Heart Failure between Male and Female")
    ax.set_xticks([0, 1]) 
    ax.set_xticklabels(["Female", "Male"])
    # get legend
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, ["Heart Failure", "Heart Health"])
    plt.show()
