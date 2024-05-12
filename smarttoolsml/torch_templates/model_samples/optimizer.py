from torch import optim 

def adam(model):
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    return optimizer