from base_functions import load_model
from cfg import CFG
from ultralytics import YOLO


def finetune_pipeline():
    print("1. Loading Model...")
    model = load_model()
    print("2. Start Finetuning...")
    train_yolo_model(model=model)


def train_yolo_model(
    model: YOLO,
    yaml_file_path: str = CFG.yaml_file_path,
    epochs: int = CFG.epochs,
    imgsz: int = CFG.imgsz,
    device: int = CFG.device,
    patience: int = CFG.patience,
    batch: int = CFG.batch,
    optimizer: str = CFG.optimizer,
    lr0: float = CFG.lr0,
    lrf: float = CFG.lrf,
    dropout: float = CFG.dropout,
    seed: int = CFG.seed,
) -> None:
    """
    Trains a YOLO model with specified parameters.

    Args:
        model (YOLO): The YOLO model to be trained.
        yaml_file_path (str): The file path to the dataset configuration YAML file.
        epochs (int, optional): Number of epochs to train for. Defaults to 100.
        imgsz (int, optional): Input image size. Defaults to 640.
        device (int, optional): Device to run the training on. Defaults to 0 (for CUDA device).
        patience (int, optional): Patience for early stopping. Defaults to 50.
        batch (int, optional): Batch size. Defaults to 32.
        optimizer (str, optional): Type of optimizer to use. Defaults to "auto".
        lr0 (float, optional): Initial learning rate. Defaults to 0.0001.
        lrf (float, optional): Final learning rate factor. Defaults to 0.1.
        dropout (float, optional): Dropout rate. Defaults to 0.1.
        seed (int, optional): Seed for random number generators. Defaults to 0.

    Returns:
        None
    """

    results = model.train(
        data=yaml_file_path,
        epochs=epochs,
        imgsz=imgsz,
        device=device,
        patience=patience,
        batch=batch,
        optimizer=optimizer,
        lr0=lr0,
        lrf=lrf,
        dropout=dropout,
        seed=seed,
    )
