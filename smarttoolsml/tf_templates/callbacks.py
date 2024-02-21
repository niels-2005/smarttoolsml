import tensorflow as tf
from tensorflow.keras import callbacks


def create_mc_es_rlop_callbacks(
    checkpoint_path: str, patience: int = 5, monitor: str = "val_loss"
) -> tuple:
    """
    Creates a list of common callbacks used during the training of a Keras model.

    This function generates three callbacks:
    - EarlyStopping: Monitors a specified metric and stops training when it stops improving after a certain number of epochs.
    - ReduceLROnPlateau: Reduces learning rate when a metric has stopped improving, which helps the model to get out of plateaus.
    - ModelCheckpoint: Saves the model after every epoch where the monitored metric has improved.

    Args:
        checkpoint_path (str): Path to save the model file.
        patience (int, optional): Number of epochs with no improvement after which training will be stopped or learning rate will be reduced. Defaults to 5.
        monitor (str, optional): Metric to be monitored by the callbacks. Defaults to 'val_loss'.

    Returns:
        tuple: A tuple containing the configured EarlyStopping, ReduceLROnPlateau, and ModelCheckpoint callbacks.

    Example usage:
        callbacks = create_mc_es_rlop_callbacks('./model_checkpoint.h5', patience=10, monitor='val_accuracy')
        model.fit(x_train, y_train, callbacks=callbacks)
    """
    early_stopping = callbacks.EarlyStopping(
        patience=patience * 2, monitor=monitor, restore_best_weights=True
    )

    reduce_lr_on_plateau = callbacks.ReduceLROnPlateau(
        patience=patience, monitor=monitor, verbose=1
    )

    model_checkpoint = callbacks.ModelCheckpoint(
        filepath=checkpoint_path, monitor=monitor, save_best_only=True
    )

    return early_stopping, reduce_lr_on_plateau, model_checkpoint
