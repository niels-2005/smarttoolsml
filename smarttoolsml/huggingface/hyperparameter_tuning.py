# Example for Hyperparameter Search with OPTUNA Backend

trainer = Trainer(
    args=training_args,
    tokenizer=tokenizer,
    train_dataset=encoded_dataset["train"],
    eval_dataset=encoded_dataset["validation"],
    model_init=model_init,
    compute_metrics=compute_metrics,
)


def optuna_hp_space(trial):
    return {
        "learning_rate": trial.suggest_float("learning_rate", 1e-6, 1e-4, log=True),
        "per_device_train_batch_size": trial.suggest_categorical(
            "per_device_train_batch_size", [16, 32, 64, 128]
        ),
    }


best_trials = trainer.hyperparameter_search(
    direction=["maximize"],
    backend="optuna",
    hp_space=optuna_hp_space,
    n_trials=20,
)
