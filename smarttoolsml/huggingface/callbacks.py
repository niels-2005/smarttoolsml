import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from transformers import EarlyStoppingCallback, Trainer, TrainingArguments


# Eigener Callback für ReduceLROnPlateau-Scheduler
class ReduceLROnPlateauCallback(Trainer):
    def __init__(self, patience=3, factor=0.1):
        super().__init__()
        self.patience = patience
        self.factor = factor
        self.scheduler = None

    def on_train_begin(self, args, state, control, **kwargs):
        # ReduceLROnPlateau-Scheduler erstellen
        self.scheduler = ReduceLROnPlateau(
            optimizer=self.model.optimizer,
            mode="min",
            patience=self.patience,
            factor=self.factor,
            verbose=True,
        )

    def on_evaluate(self, args, state, control, metrics, **kwargs):
        # Scheduler basierend auf den Validierungs-Loss anpassen
        val_loss = metrics["eval_loss"]
        self.scheduler.step(val_loss)


def trainer_with_callbacks():
    early_stopping = EarlyStoppingCallback(early_stopping_patience=10)
    lr_scheduler = ReduceLROnPlateauCallback(patience=5)
    # trainer = Trainer(
    #     model,
    #     args,
    #     ...
    #     compute_metrics=compute_metrics,
    #     callbacks = [early_stopping, lr_scheduler]
    # )
