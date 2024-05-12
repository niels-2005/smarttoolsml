import torch.optim as optim 
from torch.optim.lr_scheduler import ReduceLROnPlateau 

def reduce_lr_on_plateau(model):
    opt = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = ReduceLROnPlateau(opt, mode="min", factor=0.1, patience=5, verbose=True)
    return scheduler