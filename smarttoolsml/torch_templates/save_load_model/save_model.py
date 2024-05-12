from pathlib import Path

import torch

device = "cuda" if torch.cuda.is_available() else "cpu"


def save_model_state_dict(model, MODEL_NAME: str = "model.pth"):
    """_summary_

    Args:
        model (_type_): _description_
        MODEL_NAME (str, optional): _description_. Defaults to "model.pth".

    Example usage:
        model = Model()
        MODEL_NAME = "mymodel.pth"
        save_model(model=model, MODEL_NAME=MODEL_NAME)
    """
    MODEL_PATH = Path("models")
    MODEL_PATH.mkdir(
        parents=True,  # create parent directories if needed
        exist_ok=True,  # if models directory already exists, don't error
    )
    MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME
    print(f"Saving model to: {MODEL_SAVE_PATH}")
    torch.save(
        obj=model.state_dict(),  # only saving the state_dict() only saves the learned parameters
        f=MODEL_SAVE_PATH,
    )


def load_model_state_dict(model, MODEL_SAVE_PATH: str):
    """_summary_

    Args:
        model (_type_): _description_
        MODEL_SAVE_PATH (str): _description_

    Returns:
        _type_: _description_

    Example usage:
        model = model()
        MODEL_SAVE_PATH = "./models/model.pth"
        load_model_state_dict(model=model, MODEL_SAVE_PATH=MODEL_SAVE_PATH)
    """
    model.load_state_dict(torch.load(f=MODEL_SAVE_PATH))
    loaded_model = model.to(device)
    return loaded_model
