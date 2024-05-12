import torch 
from torch import nn 

def binary_classification():
    loss = nn.BCEWithLogitsLoss()


def categorical_classification():
    loss = nn.CrossEntropyLoss