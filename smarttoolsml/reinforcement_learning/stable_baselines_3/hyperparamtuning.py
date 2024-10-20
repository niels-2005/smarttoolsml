import gymnasium as gym
import optuna
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack, VecTransposeImage

# Zielumgebung
env_id = "Breakout-v4"


def optimize_ppo(trial):
    n_steps = trial.suggest_int("n_steps", 128, 2048, log=True)
    gamma = trial.suggest_float("gamma", 0.8, 0.9999, log=True)
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    ent_coef = trial.suggest_float("ent_coef", 0.00001, 0.1, log=True)
    clip_range = trial.suggest_float("clip_range", 0.1, 0.4, log=True)
    vf_coef = trial.suggest_float("vf_coef", 0.1, 1.0, log=True)
    max_grad_norm = trial.suggest_float("max_grad_norm", 0.3, 5.0, log=True)

    # Die Batch-Size muss ein Teiler von n_steps * n_envs sein, um die Warnung zu vermeiden
    batch_size_options = [i for i in range(32, 256) if (n_steps * 4) % i == 0]
    batch_size = trial.suggest_categorical("batch_size", batch_size_options)

    # Atari-Umgebung erstellen
    env = make_atari_env(env_id, n_envs=4, seed=42)
    env = VecFrameStack(env, n_stack=4)
    env = VecTransposeImage(env)

    model = PPO(
        "CnnPolicy",
        env,
        n_steps=n_steps,
        gamma=gamma,
        learning_rate=learning_rate,
        ent_coef=ent_coef,
        clip_range=clip_range,
        vf_coef=vf_coef,
        max_grad_norm=max_grad_norm,
        batch_size=batch_size,
        verbose=0,
    )

    eval_env = make_atari_env(env_id, n_envs=1, seed=42)
    eval_env = VecFrameStack(eval_env, n_stack=4)
    eval_env = VecTransposeImage(eval_env)
    eval_callback = EvalCallback(
        eval_env, eval_freq=10000, n_eval_episodes=5, verbose=0
    )

    model.learn(total_timesteps=200000, callback=eval_callback)

    mean_reward, _ = model.evaluate_policy(eval_env, n_eval_episodes=5)

    return mean_reward


study = optuna.create_study(direction="maximize")
study.optimize(optimize_ppo, n_trials=50)

# Die besten Hyperparameter anzeigen
print("Beste Hyperparameter:", study.best_params)
