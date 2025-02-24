import keras_tuner
import tensorflow as tf
from tensorflow.keras import layers  # type: ignore


# build your model here with different hyperparameter choices.
def build_model(hp):
    model = tf.keras.Sequential()
    # Tune the number of layers.
    for i in range(hp.Int("num_layers", 1, 3)):
        model.add(
            layers.Dense(
                units=hp.Int(f"units_{i}", min_value=32, max_value=512, step=32),
                activation=hp.Choice("activation", ["relu", "tanh"]),
            )
        )
        # dropout or not with tuneable dropout rate
        if hp.Boolean(f"dropout_{i}"):
            model.add(
                layers.Dropout(
                    rate=hp.Float(
                        f"dropout_rate_{i}", min_value=0.2, max_value=0.5, step=0.1
                    )
                )
            )

    # output layer
    model.add(layers.Dense(10, activation="softmax"))

    # learning rate for optimizer
    learning_rate = hp.Float("lr", min_value=1e-4, max_value=1e-2, sampling="log")
    # or optimizer = hp.Choice("optimizer", ["adam", "rmsprop"])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def get_data():
    return 0, 0, 0, 0


if __name__ == "__main__":
    build_model(keras_tuner.HyperParameters())
    tuner = keras_tuner.RandomSearch(
        hypermodel=build_model,
        objective="val_accuracy",
        max_trials=3,
        executions_per_trial=2,
        overwrite=True,
        directory="my_dir",
        project_name="helloworld",
    )

    X_train, X_val, y_train, y_val = get_data()
    # example callbacks: earlystopping, reducelronplateau, tensorboard
    tuner.search(
        X_train,
        y_train,
        epochs=50,
        batch_size=32,
        validation_data=(X_val, y_val),
        callbacks=[],
    )

    # get specific models
    models = tuner.get_best_models(num_models=2)
    best_model = models[0]
    best_model.summary()

    # if interested in tuning summary
    tuner.results_summary()
